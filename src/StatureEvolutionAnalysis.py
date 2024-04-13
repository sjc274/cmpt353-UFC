import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, normaltest

fight_data = pd.read_csv('fight_data_cleaned.csv')

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        return pd.to_datetime(date_str, format='%m/%d/%Y')

fight_data['date'] = fight_data['date'].apply(parse_date) # Convert to datetime
fight_data.dropna(subset=['date'], inplace=True) # Drop rows with invalid dates
fight_data['year'] = fight_data['date'].dt.year

fighters = fight_data[['R_fighter', 'B_fighter', 'R_Height_cms', 'B_Height_cms', 'R_Reach_cms', 'B_Reach_cms', 'R_Weight_lbs', 'B_Weight_lbs', 'year']].copy()
R_fighters = fighters[['R_fighter', 'R_Height_cms', 'R_Reach_cms', 'R_Weight_lbs', 'year']]
B_fighters = fighters[['B_fighter', 'B_Height_cms', 'B_Reach_cms', 'B_Weight_lbs', 'year']]
R_fighters.columns = ['fighter', 'Height_cms', 'Reach_cms', 'Weight_lbs', 'year']
B_fighters.columns = ['fighter', 'Height_cms', 'Reach_cms', 'Weight_lbs', 'year']


combined_fighters = pd.concat([R_fighters, B_fighters], ignore_index=True)
combined_fighters = combined_fighters.drop_duplicates()

mean_values_by_year = pd.DataFrame()
mean_values_by_year['Height_cms_mean'] = combined_fighters.groupby('year')['Height_cms'].mean()
mean_values_by_year['Weight_lbs_mean'] = combined_fighters.groupby('year')['Weight_lbs'].mean()
mean_values_by_year['Reach_cms_mean'] = combined_fighters.groupby('year')['Reach_cms'].mean()
mean_values_by_year['#Fighter_in_the_year'] = combined_fighters.groupby('year')['fighter'].nunique()

mean_values_by_year

# Plot Height
plt.figure(figsize=(10, 6))
plt.plot(mean_values_by_year.index, mean_values_by_year['Height_cms_mean'], marker='o', linestyle='-')
plt.title('Mean Height Change Over Years')
plt.xlabel('Year')
plt.ylabel('Mean Height (cms)')
plt.grid(True)
plt.show()

# Plot Weight
plt.figure(figsize=(10, 6))
plt.plot(mean_values_by_year.index, mean_values_by_year['Weight_lbs_mean'], marker='o', linestyle='-')
plt.title('Mean Weight Change Over Years')
plt.xlabel('Year')
plt.ylabel('Mean Weight (lbs)')
plt.grid(True)
plt.show()

# Plot Reach
plt.figure(figsize=(10, 6))
plt.plot(mean_values_by_year.index, mean_values_by_year['Reach_cms_mean'], marker='o', linestyle='-')
plt.title('Mean Reach Change Over Years')
plt.xlabel('Year')
plt.ylabel('Mean Reach (cms)')
plt.grid(True)
plt.show()

# Plot Number of Fighters
plt.figure(figsize=(10, 6))
plt.bar(mean_values_by_year.index, mean_values_by_year['#Fighter_in_the_year'])
plt.title('Number of Unique Fighters Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Fighters')
plt.grid(axis='y')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mean_values_by_year.index, mean_values_by_year['Height_cms_mean'], marker='o', linestyle='-', label='Mean Height')
ax.plot(mean_values_by_year.index, mean_values_by_year['Weight_lbs_mean'], marker='o', linestyle='-', label='Mean Weight')
ax.plot(mean_values_by_year.index, mean_values_by_year['Reach_cms_mean'], marker='o', linestyle='-', label='Mean Reach')
ax.set_title('Mean Anthropometric Changes Over Years')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Value')
ax.grid(True)
ax.legend()

plt.show()



from scipy.stats import ttest_ind, mannwhitneyu, levene, normaltest

def normality_test(data):
    _, p = normaltest(data)
    return p

def equal_variance_test(data1, data2):
    _, p = levene(data1, data2)
    return p

# Lets compare the fighters of 2010 vs 2021 

fighters_2010 = combined_fighters.loc[combined_fighters['year'] == 2010, ['fighter', 'Height_cms', 'Reach_cms', 'Weight_lbs']]
fighters_2021 = combined_fighters.loc[combined_fighters['year'] == 2021, ['fighter', 'Height_cms', 'Reach_cms', 'Weight_lbs']]
fighters_2014 = combined_fighters.loc[combined_fighters['year'] == 2014, ['fighter', 'Height_cms', 'Reach_cms', 'Weight_lbs']]




from scipy.stats import ttest_ind, mannwhitneyu, levene, normaltest

# Check for normality using normaltest
def normality_test(data):
    t,p = normaltest(data)
    if p < 0.05:
        return False
    return True

# Check for homogeneity of variances using Levene's test
def equal_variance_test(data1, data2):
    t,p = levene(data1, data2)
    if p < 0.05:
        return False
    return True

print("\n\nComparing year 2010 and 2021 height weight and reach")
# Perform appropriate statistical test based on normality and homogeneity results for each column
for column in fighters_2010.columns:
    if column!='fighter':
        if normality_test(fighters_2010[column]) and normality_test(fighters_2021[column]) and equal_variance_test(fighters_2010[column], fighters_2021[column]):
            # If both test passed, use independent t-test
            _, p_value = ttest_ind(fighters_2010[column], fighters_2021[column])
            print(f"T-test p-value for {column}: {p_value}")
        else:
            # Otherwise, use Mann-Whitney U test
            _, p_value = mannwhitneyu(fighters_2010[column], fighters_2021[column])
            print(f"Mann-Whitney U test p-value for {column}: {p_value}")

# Lets compare the fighters of 2014 vs 2021 

print("\n\n\nComparing year 2014 and 2021 height weight and reach")

from scipy.stats import ttest_ind, mannwhitneyu, levene, normaltest

def normality_test(data):
    _, p = normaltest(data)
    return p

def equal_variance_test(data1, data2):
    _, p = levene(data1, data2)
    return p

for column in fighters_2014.columns:
    if column != 'fighter':
        print(f"Column: {column}")
        p_value_2014 = normality_test(fighters_2014[column])
        p_value_2021 = normality_test(fighters_2021[column])
        print(f"Normality test p-value for 2014: {p_value_2014}")
        print(f"Normality test p-value for 2021: {p_value_2021}")
        
        p_value_var = equal_variance_test(fighters_2014[column], fighters_2021[column])
        print(f"Equal variance test p-value: {p_value_var}")
        
        if p_value_2014 > 0.05 and p_value_2021 > 0.05 and p_value_var > 0.05:
            # If both tests passed, use independent t-test
            t_stat, p_value = ttest_ind(fighters_2014[column], fighters_2021[column])
            print(f"T-test p-value for {column}: {p_value}")
        else:
            # Otherwise, use Mann-Whitney U test
            _, p_value = mannwhitneyu(fighters_2014[column], fighters_2021[column], alternative='two-sided')
            print(f"Mann-Whitney U test p-value for {column}: {p_value}")