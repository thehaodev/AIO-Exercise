import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# 1. Read dataset
dataset_path = 'Problem3.csv'
data_df = pd.read_csv(dataset_path)

# 2. Numeric string data

categorical_cols = data_df.select_dtypes(include=['object', 'bool']).columns.to_list()
for col_name in categorical_cols:
    n_categories = data_df[col_name].nunique()
    print(f'Number of categories in {col_name}: {n_categories}')

ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(data_df[categorical_cols])

encoded_categorical_df = pd.DataFrame(encoded_categorical_cols, columns=categorical_cols)

numerical_df = data_df.drop(categorical_cols, axis=1)
encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)

# 3. Define feature(X) and target(y)
X = encoded_df.drop(columns='area')
y = encoded_df['area']

# 4.  Split data to train and test with ratio 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# 5. Define model
xg_reg = xgb.XGBRegressor(seed=7, learning_rate=0.01, n_estimators=102, max_depth=3)
xg_reg.fit(X_train, y_train)

# 6. Avoid over fitting
preds = xg_reg.predict(X_test)

# 7. Measure predict base on mae and mse
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print('Evaluation results on test set:')
print(f'Mean Absolute Error :{mae}')
print(f"Mean Squared Error : {mse}")
