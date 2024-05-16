# Python_project
Python Project - Salaries dataset
import sqlite3
import csv

# Connect to SQLite database
conn = sqlite3.connect('salaries.db')
cursor = conn.cursor()

# Create tables with full forms
cursor.execute('''
CREATE TABLE IF NOT EXISTS ExperienceLevel (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experience_level TEXT UNIQUE,
    full_form TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS EmploymentType (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employment_type TEXT UNIQUE,
    full_form TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS JobTitle (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title TEXT UNIQUE
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS EmployeeResidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_residence TEXT UNIQUE
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS RemoteRatio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    remote_ratio INTEGER UNIQUE
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS CompanyLocation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_location TEXT UNIQUE
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS CompanySize (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_size TEXT UNIQUE,
    full_form TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS Salaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_year INTEGER,
    experience_level_id INTEGER,
    employment_type_id INTEGER,
    job_title_id INTEGER,
    salary INTEGER,
    salary_currency TEXT,
    salary_in_usd INTEGER,
    employee_residence_id INTEGER,
    remote_ratio_id INTEGER,
    company_location_id INTEGER,
    company_size_id INTEGER,
    FOREIGN KEY (experience_level_id) REFERENCES ExperienceLevel (id),
    FOREIGN KEY (employment_type_id) REFERENCES EmploymentType (id),
    FOREIGN KEY (job_title_id) REFERENCES JobTitle (id),
    FOREIGN KEY (employee_residence_id) REFERENCES EmployeeResidence (id),
    FOREIGN KEY (remote_ratio_id) REFERENCES RemoteRatio (id),
    FOREIGN KEY (company_location_id) REFERENCES CompanyLocation (id),
    FOREIGN KEY (company_size_id) REFERENCES CompanySize (id)
)
''')

# Helper function to insert or get id from a table with full forms
def insert_or_get_id(cursor, table, column, value, full_form=None):
    cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
    row = cursor.fetchone()
    if row:
        return row[0]
    if full_form:
        cursor.execute(f"INSERT INTO {table} ({column}, full_form) VALUES (?, ?)", (value, full_form))
    else:
        cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
    return cursor.lastrowid

# Define full forms for relevant columns
experience_level_full_forms = {
    'EN': 'Entry-level',
    'MI': 'Mid-level',
    'SE': 'Senior-level',
    'EX': 'Executive-level'
}

employment_type_full_forms = {
    'FT': 'Full-time',
    'PT': 'Part-time',
    'CT': 'Contract',
    'FL': 'Freelance'
}

company_size_full_forms = {
    'S': 'Small',
    'M': 'Medium',
    'L': 'Large'
}

# Insert unique values into individual tables
with open('salaries.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        experience_level_id = insert_or_get_id(cursor, 'ExperienceLevel', 'experience_level', row['experience_level'], experience_level_full_forms.get(row['experience_level']))
        employment_type_id = insert_or_get_id(cursor, 'EmploymentType', 'employment_type', row['employment_type'], employment_type_full_forms.get(row['employment_type']))
        job_title_id = insert_or_get_id(cursor, 'JobTitle', 'job_title', row['job_title'])
        employee_residence_id = insert_or_get_id(cursor, 'EmployeeResidence', 'employee_residence', row['employee_residence'])
        remote_ratio_id = insert_or_get_id(cursor, 'RemoteRatio', 'remote_ratio', row['remote_ratio'])
        company_location_id = insert_or_get_id(cursor, 'CompanyLocation', 'company_location', row['company_location'])
        company_size_id = insert_or_get_id(cursor, 'CompanySize', 'company_size', row['company_size'], company_size_full_forms.get(row['company_size']))

        cursor.execute('''
            INSERT INTO Salaries (
                work_year, experience_level_id, employment_type_id, job_title_id, salary,
                salary_currency, salary_in_usd, employee_residence_id, remote_ratio_id,
                company_location_id, company_size_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['work_year'], experience_level_id, employment_type_id, job_title_id,
            row['salary'], row['salary_currency'], row['salary_in_usd'],
            employee_residence_id, remote_ratio_id, company_location_id, company_size_id
        ))

# Commit the changes and close the connection
conn.commit()
conn.close()

import pandas as pd
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('salaries.db')

# SQL query with JOINs to fetch the data
query = '''
SELECT
    s.id,
    s.work_year,
    e.experience_level,
    e.full_form AS experience_level_full_form,
    et.employment_type,
    et.full_form AS employment_type_full_form,
    j.job_title,
    s.salary,
    s.salary_currency,
    s.salary_in_usd,
    er.employee_residence,
    rr.remote_ratio,
    cl.company_location,
    cs.company_size,
    cs.full_form AS company_size_full_form
FROM
    Salaries s
JOIN
    ExperienceLevel e ON s.experience_level_id = e.id
JOIN
    EmploymentType et ON s.employment_type_id = et.id
JOIN
    JobTitle j ON s.job_title_id = j.id
JOIN
    EmployeeResidence er ON s.employee_residence_id = er.id
JOIN
    RemoteRatio rr ON s.remote_ratio_id = rr.id
JOIN
    CompanyLocation cl ON s.company_location_id = cl.id
JOIN
    CompanySize cs ON s.company_size_id = cs.id
'''

# Load data into Pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

relevant_columns = ['work_year', 'experience_level', 'employment_type',
       'salary_currency', 'employee_residence', 'remote_ratio',
       'company_location', 'company_size', 'salary_in_usd', ]

df = df[relevant_columns]

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=42)

!pip install ydata-profiling

from ydata_profiling import ProfileReport

# Generate profile report for train data
profile = ProfileReport(X_train, title='Train Data Profile Report', explorative=True)
profile.to_file("train_data_profile_report.html")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Selecting only numerical columns for correlation matrix
numerical_data = train.select_dtypes(include=['int64', 'float64'])

correlation_matrix = numerical_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')

%matplotlib inline
plt.show()

# Violin plot for Experience Level
plt.figure(figsize=(12, 8))
sns.violinplot(x=train['experience_level'], y=train['salary_in_usd'])
plt.title('Salary Distribution by Experience Level')
plt.show()

# Violin plot for Employment Type
plt.figure(figsize=(12, 8))
sns.violinplot(x=train['employment_type'], y=train['salary_in_usd'])
plt.title('Salary Distribution by Employment Type')
plt.show()

# Violin plot for Company Size
plt.figure(figsize=(12, 8))
sns.violinplot(x=train['company_size'], y=train['salary_in_usd'])
plt.title('Salary Distribution by Company Size')
plt.show()

train.columns

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming X_train is your DataFrame
columns = ['work_year', 'experience_level', 'employment_type',
           'salary_currency', 'employee_residence', 'remote_ratio',
           'company_location', 'company_size']

# Loop through each column and plot the histogram
for column in columns:
    plt.figure(figsize=(20, 10))
    sns.histplot(data=train, x=column, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)  # Rotate x labels if necessary
    plt.show()
df = df.drop(columns=['salary_currency'])
df = df[(df['employee_residence'] == 'US') & (df['company_location'] == 'US') & (df['employment_type'] == 'FT')]
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=42)

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    # Train our custom preprocessors
    numerical_columns = [
        'remote_ratio'
    ]
    categorical_columns = [
        'work_year', 'experience_level', 'employment_type',
        'employee_residence', 'company_location', 'company_size'
    ]

    def fit(self, X, y=None):

        # Create and fit simple imputer
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(X[self.numerical_columns])

        # Create and fit Standard Scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.numerical_columns])
        return self

     def transform(self, X):

        # Filter rows where 'employee_residence' and 'company_location' are 'US'
        # X = X[(X['employee_residence'] == 'US') & (X['company_location'] == 'US') & (X['employment_type'] == 'FT')]

        # Apply simple imputer
        imputed_cols = self.imputer.transform(X[self.numerical_columns])
        onehot_cols = self.onehot.transform(X[self.categorical_columns])

        # Copy the df
        transformed_df = X.copy()

        # Apply transformed columns
        transformed_df[self.numerical_columns] = imputed_cols
        transformed_df[self.numerical_columns] = self.scaler.transform(transformed_df[self.numerical_columns])

        # Drop existing categorical columns and replace with one hot equivalent
        transformed_df = transformed_df.drop(self.categorical_columns, axis=1)
        transformed_df[self.onehot.get_feature_names_out()] = onehot_cols.toarray().astype(int)

        return transformed_df

preprocessor = Preprocessor()
preprocessor.fit(train)
train_fixed = preprocessor.transform(train)

train_fixed.info()

!pip install mlflow

import os
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/gourimenon8/Python_project.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'gourimenon8'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '8ab2cd5b45a51508f012ed29fd3d7ba0f26545f6'

# Import MLFlow
import mlflow
import mlflow.sklearn

# Set the tracking URI to Dagshub
mlflow.set_tracking_uri('https://dagshub.com/gourimenon8/Python_project.mlflow ')
mlflow.set_experiment('Salary Prediction Experiments')

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
rfr = make_pipeline(Preprocessor(), RandomForestRegressor(n_estimators=50))
rfr

y_train = train['salary_in_usd']
X_train = train.drop('salary_in_usd', axis=1)
rfr.fit(X_train, y_train)

params = rfr.get_params()
params

from sklearn.metrics import mean_squared_error, mean_absolute_error
y_train_hat=rfr.predict(X_train)
print(mean_squared_error(y_train, y_train_hat))
print(mean_absolute_error(y_train, y_train_hat))

y_test = test['salary_in_usd']
X_test = test.drop('salary_in_usd', axis=1)

y_test_hat=rfr.predict(X_test)
rm2e = mean_squared_error(y_test, y_test_hat)
mae = mean_absolute_error(y_test, y_test_hat)
import mlflow
from mlflow.models import infer_signature

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metric("root_mean_squared_error", rm2e)
    mlflow.log_metric("mean_absolute_error", mae)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "RandomForestRegressor model for housing data, n_estimators=50")

    # Infer the model signature
    signature = infer_signature(X_train, rfr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=rfr,
        artifact_path="housing_model",
        signature=signature,
        input_example=preprocessor.transform(X_train),
        registered_model_name="rfr_moodel_n_estimators=50",
    )

!pip install lazypredict

from lazypredict.Supervised import LazyRegressor  # Change to LazyClassifier for classification

# Initialize LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Display the top models
models

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set the tracking URI to your MLFlow server
mlflow.set_tracking_uri('https://dagshub.com/gourimenon8/Python_project.mlflow')
mlflow.set_experiment('Salary Prediction Experiments')

# Prepare data
preprocessor = Preprocessor()
preprocessor.fit(train)

X_train = preprocessor.transform(train.drop('salary_in_usd', axis=1))
y_train = train['salary_in_usd']
X_test = preprocessor.transform(test.drop('salary_in_usd', axis=1))
y_test = test['salary_in_usd']
def log_model(model, model_name):
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Predict on train and test data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)

        print(f"Logged model {model_name} to MLFlow")
        print(f"Metrics:\n{metrics}")
    # Initialize models
rf = RandomForestRegressor(n_estimators=50, random_state=42)
gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
xgb = XGBRegressor(n_estimators=50, random_state=42)

# Log models to MLFlow
log_model(rf, "RandomForestRegressor")
log_model(gb, "GradientBoostingRegressor")
log_model(xgb, "XGBRegressor")



