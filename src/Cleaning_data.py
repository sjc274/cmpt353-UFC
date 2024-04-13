import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_ufc_data = pd.read_csv('fight_data.csv')

# Removing unwanted columns
irrelevant_columns = ['R_odds', 'B_odds', 'R_ev', 'B_ev', 'B_current_lose_streak', 'B_current_win_streak',
                      'B_longest_win_streak', 'R_current_lose_streak', 'R_current_win_streak', 'R_longest_win_streak',
                      'empty_arena', 'constant_1', 'B_match_weightclass_rank', 'R_match_weightclass_rank',
                      "R_Women's Flyweight_rank", "R_Women's Featherweight_rank", "R_Women's Strawweight_rank",
                      "R_Women's Bantamweight_rank", "R_Heavyweight_rank", "R_Light Heavyweight_rank",
                      'R_Middleweight_rank', 'R_Welterweight_rank', 'R_Lightweight_rank', 'R_Featherweight_rank',
                      'R_Bantamweight_rank', 'R_Flyweight_rank', 'R_Pound-for-Pound_rank',
                      "B_Women's Flyweight_rank", "B_Women's Featherweight_rank", "B_Women's Strawweight_rank",
                      "B_Women's Bantamweight_rank", 'B_Heavyweight_rank', 'B_Light Heavyweight_rank',
                      'B_Middleweight_rank', 'B_Welterweight_rank', 'B_Lightweight_rank', 'B_Featherweight_rank',
                      'B_Bantamweight_rank', 'B_Flyweight_rank', 'B_Pound-for-Pound_rank', 'r_dec_odds', 'b_dec_odds',
                      'r_sub_odds', 'b_sub_odds', 'r_ko_odds', 'b_ko_odds']

raw_ufc_data.drop(columns=irrelevant_columns, inplace=True)

# Find the missing rows
missing_rows = {}
for column in raw_ufc_data.columns:
    missing_count = raw_ufc_data[column].isnull().sum()
    missing_rows[column] = missing_count

print("Missing rows for each column:")
for column, missing_count in missing_rows.items():
    print(f"{column}: {missing_count} missing rows")

# Fill missing values with mean
columns_to_fill = ['B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_avg_TD_pct',
                   'R_avg_SIG_STR_landed', 'R_avg_SIG_STR_pct', 'R_avg_SUB_ATT', 'R_avg_TD_landed', 'R_avg_TD_pct']
for column in columns_to_fill:
    raw_ufc_data[column].fillna(raw_ufc_data[column].mean(), inplace=True)

# Fill missing values for 'finish' column based on specified distribution
finish_distribution = {'DQ': 0.3, 'KO/TKO': 32, 'M-Dec': 0.6, 'Overturned': 0.04, 'S-dec': 10.4, 'Sub': 18.3, 'U-Dec': 38.17}
for finish_type, percentage in finish_distribution.items():
    num_missing = int(missing_rows['finish'] * percentage / 100)
    raw_ufc_data.loc[raw_ufc_data['finish'].isnull(), 'finish'] = finish_type
    missing_rows['finish'] -= num_missing

# Replace missing values for 'finish_details' with 'blank'
raw_ufc_data['finish_details'].fillna('blank', inplace=True)

# Distribute missing values for 'finish_round' based on specified percentages
round_distribution = {1: 25.8, 2: 15.7, 3: 54.1, 4: 0.6, 5: 3.7}
for round_num, percentage in round_distribution.items():
    num_missing = int(missing_rows['finish_round'] * percentage / 100)
    raw_ufc_data.loc[raw_ufc_data['finish_round'].isnull(), 'finish_round'] = round_num
    missing_rows['finish_round'] -= num_missing

# Replace missing values for 'finish_round_time' with '5:00'
raw_ufc_data['finish_round_time'].fillna('5:00', inplace=True)

# Calculate and replace missing values for 'total_fight_time_secs' based on (finish round * 5 * 60)
raw_ufc_data['total_fight_time_secs'].fillna(raw_ufc_data['finish_round'] * 5 * 60, inplace=True)

# Additional cleaning for B_Stance
raw_ufc_data.loc[raw_ufc_data['B_fighter'] == 'Juancamilo Ronderos', 'B_Stance'] = 'Southpaw'
raw_ufc_data.loc[raw_ufc_data['B_fighter'] == 'Juan Espino', 'B_Stance'] = 'Orthodox'

# Function to round numbers to two decimal places
def round_to_two_decimals(value):
    return round(value, 2)

# Convert all numerical columns to two decimal places
numerical_columns = raw_ufc_data.select_dtypes(include='number').columns
raw_ufc_data[numerical_columns] = raw_ufc_data[numerical_columns].applymap(round_to_two_decimals)

# Fix spacing issue in the 'Country' column
raw_ufc_data['country'] = raw_ufc_data['country'].str.strip()

# Calculate weight difference
raw_ufc_data['weight_dif'] = raw_ufc_data['R_Weight_lbs'] - raw_ufc_data['B_Weight_lbs']

# Write the updated data to a new file
raw_ufc_data.to_csv('fight_data_cleaned.csv', index=False)
