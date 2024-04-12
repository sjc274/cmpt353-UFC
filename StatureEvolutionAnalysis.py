import pandas as pd
import matplotlib.pyplot as plt

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

# Plot Height
plt.figure(figsize=(10, 6))
plt.plot(mean_values_by_year.index, mean_values_by_year['Height_cms_mean'], marker='o', linestyle='-')
plt.title('Mean Height Change Over Years')
plt.xlabel('Year')
plt.ylabel('Mean Height (cms)')
plt.grid(True)
plt.savefig('height_plot.png')
plt.close()

# Plot Weight
plt.figure(figsize=(10, 6))
plt.plot(mean_values_by_year.index, mean_values_by_year['Weight_lbs_mean'], marker='o', linestyle='-')
plt.title('Mean Weight Change Over Years')
plt.xlabel('Year')
plt.ylabel('Mean Weight (lbs)')
plt.grid(True)
plt.savefig('weight_plot.png')
plt.close()

# Plot Reach
plt.figure(figsize=(10, 6))
plt.plot(mean_values_by_year.index, mean_values_by_year['Reach_cms_mean'], marker='o', linestyle='-')
plt.title('Mean Reach Change Over Years')
plt.xlabel('Year')
plt.ylabel('Mean Reach (cms)')
plt.grid(True)
plt.savefig('reach_plot.png')
plt.close()

# Plot Number of Fighters
plt.figure(figsize=(10, 6))
plt.bar(mean_values_by_year.index, mean_values_by_year['#Fighter_in_the_year'])
plt.title('Number of Unique Fighters Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Fighters')
plt.grid(axis='y')
plt.savefig('fighter_plot.png')
plt.close()

# Plot all in one
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mean_values_by_year.index, mean_values_by_year['Height_cms_mean'], marker='o', linestyle='-', label='Mean Height')
ax.plot(mean_values_by_year.index, mean_values_by_year['Weight_lbs_mean'], marker='o', linestyle='-', label='Mean Weight')
ax.plot(mean_values_by_year.index, mean_values_by_year['Reach_cms_mean'], marker='o', linestyle='-', label='Mean Reach')
ax.set_title('Mean Anthropometric Changes Over Years')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Value')
ax.grid(True)
plt.savefig('combined.png')
plt.close()

from scipy.stats import norm, normaltest

# Filter out the data for years 2010, 2011, 2012, and 2013
filtered_mean_values = mean_values_by_year[~mean_values_by_year.index.isin([2010, 2011, 2012, 2013])]

plt.figure(figsize=(12, 8))
for idx, feature in enumerate(['Height_cms_mean', 'Weight_lbs_mean', 'Reach_cms_mean'], start=1):
    plt.subplot(2, 2, idx)
    data = filtered_mean_values[feature]
    mu, std = data.mean(), data.std()
    xmin, xmax = data.min(), data.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.hist(data, bins=10, density=True, alpha=0.6, color='g')
    plt.title(f'Normal Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Probability Density')
    plt.grid(True)

    # Perform normality test
    stat, p_value = normaltest(data)
    if p_value > 0.05:
        plt.text(0.05, 0.95, 'Normally distributed (p={:.3f})'.format(p_value), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    else:
        plt.text(0.05, 0.95, 'Not normally distributed (p={:.3f})'.format(p_value), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', color='red')

plt.tight_layout()
plt.savefig('normal_distribution.png')

