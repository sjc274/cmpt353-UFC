import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

import seaborn as sns


fight_data = pd.read_csv('fight_data_cleaned.csv')

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        return pd.to_datetime(date_str, format='%m/%d/%Y')

fight_data['date'] = fight_data['date'].apply(parse_date) # Convert to datetime
fight_data.dropna(subset=['date'], inplace=True) # Drop rows with invalid dates
fight_data['year'] = fight_data['date'].dt.year
fight_data.drop(columns=['date'])
fighters = fight_data[['R_fighter', 'B_fighter', 'height_dif', 'reach_dif','age_dif', 'weight_dif', 'Winner']].copy()


# Define features and target
features = ['height_dif', 'reach_dif','age_dif', 'weight_dif']
target = ['Winner']

new_features = ['height_dif', 'reach_dif','age_dif', 'weight_dif']
X = fighters[new_features]
y = fighters[target]
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def print_result(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Output the accuracies
    print(model)
    print(classification_report(y_test, y_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred))

    # Output the confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels=['Red', 'Blue'])
    
def ShowFeatureImportance(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = model.fit(X_train, y_train)

    hist_df = pd.DataFrame({'Feature': X.columns, 'Feature importance': model.feature_importances_})
    hist_df = hist_df.sort_values(by='Feature importance', ascending=True)
    plt.figure(figsize=(10, 4))
    plt.barh(hist_df['Feature'], hist_df['Feature importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(model)
    plt.tight_layout()
    plt.show()
    
# DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
print_result(X, y, model_dt)
ShowFeatureImportance(X, y, model_dt)

# GaussianNB
model_nb = GaussianNB()
print_result(X, y, model_nb)

# RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=200)
print_result(X, y, model_rf)
ShowFeatureImportance(X, y, model_rf)

# KNeighborsClassifier
model_knn = KNeighborsClassifier(5)
print_result(X, y, model_knn)

fight_data = pd.read_csv('fight_data_cleaned.csv')

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        return pd.to_datetime(date_str, format='%m/%d/%Y')

fight_data['date'] = fight_data['date'].apply(parse_date) # Convert to datetime
fight_data.dropna(subset=['date'], inplace=True) # Drop rows with invalid dates
fight_data['year'] = fight_data['date'].dt.year
fight_data.drop(columns=['date'])
fighters = fight_data[['R_fighter', 'B_fighter', 'lose_streak_dif', 'loss_dif', 'win_dif', 'win_streak_dif', 'longest_win_streak_dif', 'total_round_dif', 'total_title_bout_dif', 'ko_dif', 'sub_dif', 'Winner']].copy()

# Define features and target
features = ['lose_streak_dif', 'loss_dif', 'win_dif', 'win_streak_dif', 'longest_win_streak_dif', 'total_round_dif', 'total_title_bout_dif', 'ko_dif', 'sub_dif']
target = ['Winner']

new_features = ['lose_streak_dif', 'loss_dif', 'win_dif', 'win_streak_dif', 'longest_win_streak_dif', 'total_round_dif', 'total_title_bout_dif', 'ko_dif', 'sub_dif']
X = fighters[new_features]
y = fighters[target]
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def print_result(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Output the accuracies
    print(model)
    print(classification_report(y_test, y_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred))

    # Output the confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels=['Red', 'Blue'])
    
def ShowFeatureImportance(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = model.fit(X_train, y_train)

    hist_df = pd.DataFrame({'Feature': X.columns, 'Feature importance': model.feature_importances_})
    hist_df = hist_df.sort_values(by='Feature importance', ascending=True)
    plt.figure(figsize=(10, 4))
    plt.barh(hist_df['Feature'], hist_df['Feature importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(model)
    plt.tight_layout()
    plt.show()
    
# DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
print_result(X, y, model_dt)
ShowFeatureImportance(X, y, model_dt)

# GaussianNB
model_nb = GaussianNB()
print_result(X, y, model_nb)

# RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=200)
print_result(X, y, model_rf)
ShowFeatureImportance(X, y, model_rf)

# KNeighborsClassifier
model_knn = KNeighborsClassifier(5)
print_result(X, y, model_knn)