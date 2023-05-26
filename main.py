# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from joblib import dump, load

# Loading csv file
df = pd.read_csv('students_performance.csv')

#generating student id
ids = np.linspace(1, 481, num=480, dtype=int)
df['Student_ID'] = ids[:len(df)]

# Adding new columns
df['mid1'] = np.random.randint(0, 15, size=len(df))
df['mid2'] = np.random.randint(0, 15, size=len(df))
df['quiz'] = np.random.randint(0, 15, size=len(df))
df['final_score'] = np.random.randint(0, 100, size=len(df)) 

# Drop unnecessary columns
df.drop(['PlaceofBirth', 'StageID', 'GradeID', 'SectionID' , 'Topic', 'Semester', 'Relation', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction'], axis = 1, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Filter values
df = df[df['mid1'].between(0,15)]
df = df[df['mid2'].between(0,15)]
df = df[df['quiz'].between(0,15)]
df = df[df['final_score'].between(0,100)] 

# Identify categorical columns and convert them to numerical
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols)


# Normalize 
scaler = MinMaxScaler()
df[['mid1', 'mid2', 'quiz']] = scaler.fit_transform(df[['mid1', 'mid2', 'quiz']]) 



# Convert final_score to discrete values
bins = [0, 20, 40, 60, 80, 100]
labels = [1, 2, 3, 4, 5]
df['final_score'] = pd.cut(df['final_score'], bins=bins, labels=labels)

# 'final_score' is your target column
X = df[['mid1', 'mid2', 'quiz']]  # Only use these columns
y = df['final_score']

# Fill NaN values with the most frequent category in each column
for col in df.select_dtypes(include=['category']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)



# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train the model
model = LinearRegression()
model.fit(X_train, y_train)

#predict the response for test dataset
y_pred = model.predict(X_test)

#calculating the mean squared error
mse = mean_squared_error(y_test, y_pred)

#calculating the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

#calculating the root mean squared error
rmse = sqrt(mse)

#calculating the R-squared
r2 = r2_score(y_test, y_pred)

# Now applying RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions using RandomForestRegressor
rf_pred = rf.predict(X_test)

# Calculate the errors for RandomForestRegressor
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Now applying RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions using RandomForestClassifier
rf_pred = rf.predict(X_test)

# Calculate the metrics for Logistic Regression
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Calculate the metrics for RandomForestClassifier
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average='macro')
rf_recall = recall_score(y_test, rf_pred, average='macro')
rf_f1 = f1_score(y_test, rf_pred, average='macro')

# Choose the best model
if f1 > rf_f1:
    print("\nLogistic Regression performs better.")
    best_model = model
else:
    print("\nRandom Forest performs better.")
    best_model = rf

# Train the best model on the entire dataset
best_model.fit(X, y)

# Save the model to a file
dump(best_model, 'model.joblib')

# Save the scaler to a file
dump(scaler, 'scaler.joblib')

# Define a function to map scores to grades
def score_to_grade(score):
    if score == 5:
        return 'A'
    elif score == 4:
        return 'B'
    elif score == 3:
        return 'C'
    elif score == 2:
        return 'D'
    else:
        return 'F'

# # Create the 'Final_grade' column
# df['Final_grade'] = df['final_score'].apply(score_to_grade)

# # Output the relevant columns
# print(df[['mid1', 'mid2', 'quiz', 'final_score', 'Final_grade']].iloc[200:231])
