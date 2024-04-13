import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

raw_ufc_data = pd.read_csv('fight_data.csv')

# Removing irrelavant columns (More details in report)
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
    raw_ufc_data[column] = raw_ufc_data[column].fillna(raw_ufc_data[column].mean())

category_counts = raw_ufc_data['finish'].value_counts()


# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Categories')
plt.axis('equal')
plt.savefig('Finish_ditribution_piechart.png')


# Fill missing values for 'finish' column based on specified distribution
finish_distribution = {'DQ': 0.3, 'KO/TKO': 32, 'M-Dec': 0.6, 'Overturned': 0.04, 'S-dec': 10.4, 'Sub': 18.3, 'U-Dec': 38.17}
for finish_type, percentage in finish_distribution.items():
    num_missing = int(missing_rows['finish'] * percentage / 100)
    raw_ufc_data.loc[raw_ufc_data['finish'].isnull(), 'finish'] = finish_type
    missing_rows['finish'] -= num_missing


# Replace missing values for 'finish_details' with 'blank'
raw_ufc_data['finish_deatils']=raw_ufc_data['finish_details'].fillna('blank')

category_counts = raw_ufc_data['finish_round'].value_counts()


# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Categories')
plt.axis('equal')  
plt.savefig('Finish_round_ditribution_piechart.png')


# Distribute missing values for 'finish_round' based on specified percentages
round_distribution = {1: 25.8, 2: 15.7, 3: 54.1, 4: 0.6, 5: 3.7}
for round_num, percentage in round_distribution.items():
    num_missing = int(missing_rows['finish_round'] * percentage / 100)
    raw_ufc_data.loc[raw_ufc_data['finish_round'].isnull(), 'finish_round'] = round_num
    missing_rows['finish_round'] -= num_missing


# Replace missing values for 'finish_round_time' with '5:00'
raw_ufc_data['finish_round_time']=raw_ufc_data['finish_round_time'].fillna('5:00')
# Calculate and replace missing values for 'total_fight_time_secs' based on (finish round * 5 * 60)
raw_ufc_data['total_fight_time_secs']=raw_ufc_data['total_fight_time_secs'].fillna(raw_ufc_data['finish_round'] * 5 * 60).astype(int)

# Additional cleaning for B_Stance
raw_ufc_data.loc[raw_ufc_data['B_fighter'] == 'Juancamilo Ronderos', 'B_Stance'] = 'Southpaw'
raw_ufc_data.loc[raw_ufc_data['B_fighter'] == 'Juan Espino', 'B_Stance'] = 'Orthodox'

# Function to round numbers to two decimal places
def round_to_two_decimals(value):
    return round(value, 2)

# Convert all numerical columns to two decimal places
numerical_columns = raw_ufc_data.select_dtypes(include='number').columns
raw_ufc_data[numerical_columns] = raw_ufc_data[numerical_columns].map(round_to_two_decimals)
# Fix spacing issue in the 'Country' column
raw_ufc_data['country'] = raw_ufc_data['country'].str.strip()

# Write the updated data to a new file
data=raw_ufc_data

# Finding numerical and categorical data
numrical_columns=data.dtypes[(data.dtypes=='int64')|(data.dtypes=='float64')].index.tolist()
categorical_columns=data.dtypes[(data.dtypes=='object')|(data.dtypes=='bool')].index.tolist()

# Using standard scaler for numerical values
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Use label encoding for categorical values
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for col in categorical_columns:
    data[col] = LE.fit_transform(data[col])

# Separating features and label
y=data['Winner']
X=data.drop(['Winner'], axis=1)

# Creating training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2, test_size = 0.25)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

all_accuracies = []

# Define the models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators = 400, max_depth = 12, random_state = 2),
    'K Nearest Neighbour': KNeighborsClassifier(n_neighbors=100),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 128, 64), max_iter=300),
    'Support Vector Machine': SVC(),
}

print("Using manual data cleaning method")
# Train and evaluate each model
accuracies = []
for name, model in models.items():
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred)
    print(name, accuracy)
    accuracies.append((name, accuracy))
all_accuracies.append(accuracies)

    

data = pd.read_csv('fight_data.csv')
data['Winner']=data['Winner'].map({'Red': 1, 'Blue': 0})

# Drop last 3 error causing columns for calculating correlation coeffcient
data.drop(["R_Women's Featherweight_rank","B_Women's Featherweight_rank","constant_1"],axis=1, inplace=True)

# Drop columns with more than 40 percent null values
threshold = len(data) * 0.4
columns_to_drop = data.columns[data.isnull().sum() > threshold]
data.drop(columns=columns_to_drop, inplace=True)

# Finding numerical and categorical data
numerical_columns=data.dtypes[(data.dtypes=='int64')|(data.dtypes=='float64')].index.tolist()
categorical_columns=data.dtypes[(data.dtypes=='object')|(data.dtypes=='bool')].index.tolist()

# Calculate correlation coefficients between numerical columns and label
corr_dict = abs(data[numerical_columns].corrwith(data['Winner']))
corr_dict= corr_dict.sort_values(ascending=False)

for col, corr in corr_dict.items():
    print(f"{col}: {corr}")


# Visualising correlation coefficients with KMeans
from sklearn.cluster import KMeans
X = corr_dict.values.reshape(-1, 1)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(range(len(corr_dict)), corr_dict.values, c=cluster_labels, cmap='viridis', alpha=0.5)
plt.xlabel('Numerical Column Index')
plt.ylabel('Absolute Correlation Coefficient')
plt.title('KMeans Clustering of Correlation Coefficients')
plt.colorbar(label='Cluster')
plt.savefig('Clustering_of_correlation.png')


# Selecting columns with correlation greater than>10 percent
selected_columns = {col: correlation for col, correlation in corr_dict.items() if correlation > 0.05}

# Print the selected columns and their correlation values
for col, correlation in selected_columns.items():
    print(f"{col}: {correlation}")


# Creating the new dataset with the selected columns
column_list=list(selected_columns.keys())
for c in categorical_columns:
    if (c not in column_list):
        column_list.append(c)
new_data=data.loc[:,list(column_list)]
data=new_data

# Finding numerical and categorical data
numerical_columns=data.dtypes[(data.dtypes=='int64')|(data.dtypes=='float64')].index.tolist()
categorical_columns=data.dtypes[(data.dtypes=='object')|(data.dtypes=='bool')].index.tolist()

data[categorical_columns].isnull().sum().sort_values()

# Fill null values in column with the mode (most frequent value)
data.loc[:, 'finish'] = data['finish'].fillna(data['finish'].mode()[0])
data.loc[:, 'finish_round_time'] = data['finish_round_time'].fillna(data['finish_round_time'].mode()[0])
data.loc[:, 'B_Stance'] = data['B_Stance'].fillna(data['B_Stance'].mode()[0])

# Fill remaining numerical values with mean
data.loc[:,numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Finding numerical and categorical data
int_columns=data.dtypes[(data.dtypes=='int64')].index.tolist()
float_columns=data.dtypes[(data.dtypes=='float64')].index.tolist()
categorical_columns=data.dtypes[(data.dtypes=='object')].index.tolist()
boolean_columns=data.dtypes[data.dtypes=='bool'].index.tolist()


data[categorical_columns].describe()

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
for col in categorical_columns:
    # Convert column dtype to string (if necessary)
    data.loc[:, col] = data[col].astype(str)
    data.loc[:, col] = LE.fit_transform(data[col])

from sklearn.preprocessing import StandardScaler
# Applying MinMax Scaler
scaler = StandardScaler()
for col in int_columns:
    column_data = data[col].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(column_data)
    scaled_data = scaled_data.astype(data[col].dtype)
    data.loc[:,col] = scaled_data


data[int_columns].describe()

# Separating features and label
y=data['Winner']
X=data.drop(['Winner'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2, test_size = 0.25)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Define the models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators = 400, max_depth = 12, random_state = 2),
    'K Nearest Neighbour': KNeighborsClassifier(n_neighbors=100),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 128, 64), max_iter=300),
    'Support Vector Machine': SVC(),
}

print("\n\nUsing most correlated columns")
# Train and evaluate each model
accuracies=[]
for name, model in models.items():
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred)
    print(name, accuracy)
    accuracies.append((name, accuracy))
all_accuracies.append(accuracies)
    

# Using PCA+ Machine Learning

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('fight_data.csv')
label=data['Winner']
data.drop(['Winner','date'],axis=1, inplace=True)

# Step 1: Data Cleaning
threshold = len(data) * 0.4
columns_to_drop = data.columns[data.isnull().sum() > threshold]
data.drop(columns=columns_to_drop, inplace=True)


# Impute missing values for numerical columns with mean
numerical_columns = data.select_dtypes(include='number').columns
imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Impute missing values for categorical columns with most frequent value
categorical_columns = data.select_dtypes(include='object').columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = imputer_categorical.fit_transform(data[categorical_columns])


# Step 2: Feature Engineering
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for col in categorical_columns:
    data[col] = LE.fit_transform(data[col])

# Standardize numerical features
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Apply PCA
pca = PCA(n_components=0.99)  # Retain 99% of the variance
pca_result = pca.fit_transform(data)

print("Data shape ",data.shape)
print("PCA shape ",pca_result.shape)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pca_result, label, test_size=0.25, random_state=2)



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Define the models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators = 400, max_depth = 12, random_state = 2),
    'K Nearest Neighbour': KNeighborsClassifier(n_neighbors=100),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 128, 64), max_iter=300),
    'Support Vector Machine': SVC()
}

print('\n\nUsing PCA+Machine Learning')
# Train and evaluate each model
accuracies=[]
for name, model in models.items():
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred)
    print(name, accuracy)
    accuracies.append((name, accuracy))
all_accuracies.append(accuracies)
    

# Plot the comparison
plt.figure(figsize=(12, 6))
bar_width = 0.15
index = np.arange(len(models))

for i, run_accuracies in enumerate(all_accuracies):
    model_names = [name for name, _ in run_accuracies]
    accuracies = [accuracy for _, accuracy in run_accuracies]
    plt.bar(index + i * bar_width, accuracies, bar_width, label=f'Run {i+1}')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies for Different Runs')
plt.xticks(index + bar_width, model_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('Compared models.png')
