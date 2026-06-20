import sys

import numpy as np
import pandas as pd
from scipy import stats


OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    reddit_counts = sys.argv[1]
    
    #from assignment description: "It turns out Pandas (≥0.21) can handle the compression automatically and we don't need to explicitly uncompress:"
    counts = pd.read_json(reddit_counts, lines=True)
    #print(counts)
#             date        subreddit  comment_count
# 0     2012-02-20     newfoundland              7
# 1     2015-01-26         Manitoba              1    

    # For this question, we will look only at values (1) in 2012 and 2013, and (2) in the /r/canada subreddit.
    # first i will keep in canada subreddit
    counts = counts[counts['subreddit'] == 'canada']
    #print(counts)
    
    counts = counts[counts['date'].dt.year.isin([2012, 2013])] #.dt means: “Use date/time properties from this column.”

    # separate the weekdays from the weekends. Hint: check for datetime.date.weekday either 5 or 6.
    # date.weekday() Return the day of the week as an integer, where Monday is 0 and Sunday is 6 
    #source: https://docs.python.org/3/library/datetime.html#datetime.date.weekday

    is_weekend = counts['date'].dt.weekday >= 5
    # print(is_weekend)
    # so its printing the row numbers and if the row is a weekend or not. True means its a weekend, False means its a weekday.
    weekday = counts[counts['date'].dt.weekday < 5] # a smaller table(dataframe) from counts with only the weekdays
    # print("weekday:")
    # print(weekday)
    weekend = counts[is_weekend] # smaller table from counts with only the weekends



    #for the answer.txt q4
    # print("weekday average:", weekday['comment_count'].mean())
    # print("weekend average:", weekend['comment_count'].mean())
    # #weekday average: 1823.5785440613026
    # #weekend average: 1269.5071770334928

    # --------Student's T-Test
    initial_ttest = stats.ttest_ind(weekday['comment_count'], weekend['comment_count'])
    initial_weekday_normality = stats.normaltest(weekday['comment_count'])
    initial_weekend_normality = stats.normaltest(weekend['comment_count'])
    initial_levene = stats.levene(weekday['comment_count'], weekend['comment_count'])

    # so no i cannot draw a conclusion from this t-test and I also got "Original data equal-variance p-value: 0.0438"





    # --------- Fix 1: transforming data might save us.
    # import matplotlib.pyplot as plt

    # plt.hist(weekday['comment_count'])
    # plt.show()
    # # so its a right-skewed 

    # plt.hist(weekend['comment_count'])
    # plt.show()
    # # also right-skewed, but more skewed than the weekday one

    # so The data is skewed, not perfectly normal



    trans_weekday = np.sqrt(weekday['comment_count'])
    trans_weekend = np.sqrt(weekend['comment_count'])

    transformed_weekday_normality = stats.normaltest(trans_weekday)
    transformed_weekend_normality = stats.normaltest(trans_weekend)
    transformed_levene = stats.levene(trans_weekday, trans_weekend)

    # import matplotlib.pyplot as plt
    # plt.hist(trans_weekday)
    # plt.show()

    # plt.hist(trans_weekend)
    # plt.show()
    # # now they are not skwed they are normal





    # ---------- Fix 2: the Central Limit Theorem might save us.
    weekday = weekday.copy()
    weekend = weekend.copy()

    #Hints: you can get a “year” and “week number” from the first two values returned by date.isocalendar().
    weekday_iso = weekday['date'].dt.isocalendar()
    weekend_iso = weekend['date'].dt.isocalendar()
    weekday['year'] = weekday_iso['year']
    weekday['week'] = weekday_iso['week']
    weekend['year'] = weekend_iso['year']
    weekend['week'] = weekend_iso['week']

    weekly_weekday = weekday.groupby(['year', 'week'])['comment_count'].mean()
    weekly_weekend = weekend.groupby(['year', 'week'])['comment_count'].mean()

    weekly_weekday_normality = stats.normaltest(weekly_weekday)
    weekly_weekend_normality = stats.normaltest(weekly_weekend)
    weekly_levene = stats.levene(weekly_weekday, weekly_weekend)
    weekly_ttest = stats.ttest_ind(weekly_weekday, weekly_weekend)





    # -------------- Fix 3: a non-parametric test might save us.
    utest = stats.mannwhitneyu(weekday['comment_count'], weekend['comment_count'],
                               alternative='two-sided')

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest.pvalue,
        initial_weekday_normality_p=initial_weekday_normality.pvalue,
        initial_weekend_normality_p=initial_weekend_normality.pvalue,
        initial_levene_p=initial_levene.pvalue,
        transformed_weekday_normality_p=transformed_weekday_normality.pvalue,
        transformed_weekend_normality_p=transformed_weekend_normality.pvalue,
        transformed_levene_p=transformed_levene.pvalue,
        weekly_weekday_normality_p=weekly_weekday_normality.pvalue,
        weekly_weekend_normality_p=weekly_weekend_normality.pvalue,
        weekly_levene_p=weekly_levene.pvalue,
        weekly_ttest_p=weekly_ttest.pvalue,
        utest_p=utest.pvalue,
    ))


if __name__ == '__main__':
    main()
