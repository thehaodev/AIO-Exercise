import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run():
    data_frame = pd.read_csv("../data/BTC-Daily.csv")
    data_frame = data_frame.drop_duplicates()
    data_frame['date'] = pd.to_datetime(data_frame['date'])

    visualize_data_by_year(data_frame)
    visualize_candle_stick_chart(data_frame)

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    data_frame["Standardized_Close_Prices"] = scaler.fit_transform(data_frame["close"].values.reshape(-1, 1))
    data_frame["Standardized_Open_Prices"] = scaler.fit_transform(data_frame["open"].values.reshape(-1, 1))
    data_frame["Standardized_High_Prices"] = scaler.fit_transform(data_frame["high"].values.reshape(-1, 1))
    data_frame["Standardized_Low_Prices"] = scaler.fit_transform(data_frame["low"].values.reshape(-1, 1))

    # Converting Date to numerical form

    data_frame['date_str'] = data_frame['date'].dt.strftime('%Y%m%d%H%M%S')

    # Convert the string date to a numerical value
    data_frame['NumericalDate'] = pd.to_numeric(data_frame['date_str'])

    # Drop the intermediate 'date_str' column if not needed
    data_frame.drop(columns=['date_str'])

    # Defined feature and target
    x = data_frame[["Standardized_Open_Prices", "Standardized_High_Prices", "Standardized_Low_Prices"]]
    y = data_frame["Standardized_Close_Prices"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,
                                                        random_state=42, shuffle=True)

    lr = 0.01
    epochs = 200

    w, b, losses = linear_regression_vectorized(x_train.values, y_train.values, learning_rate=lr, num_iterations=epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Function during Gradient Descent')

    # Make predictions on the test set
    y_pred = predict(x_test, w, b)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    # Calculate MAE
    mae_value = np.mean(np.abs(y_pred - y_test))

    # Calculate MAPE
    _ = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Calculate R-squared on training data
    y_train_pred = predict(x_train, w, b)
    train_accuracy = r2_score(y_train, y_train_pred)

    # Calculate R-squared on testing data
    test_accuracy = r2_score(y_test, y_pred)

    # Filter data for 2015
    df_2015 = data_frame[data_frame['date'].dt.year == 2015]

    # Group by month and get the mean of the actual and predicted close prices
    monthly_actual = df_2015.groupby(df_2015['date'].dt.month)['close'].mean()

    # Assuming you have a way to predict the close prices for 2015 (replace with your prediction method)
    # For this example, I'll just use the previous day's close price as a simple prediction.
    df_2015['predicted_close'] = df_2015['close'].shift(1)
    monthly_predicted = df_2015.groupby(df_2015['date'].dt.month)['predicted_close'].mean()

    # Create a plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_actual.index, monthly_actual.values, label='Actual Close Price', marker='o')
    plt.plot(monthly_predicted.index, monthly_predicted.values, label='Predicted Close Price', marker='x')
    plt.title('Actual vs. Predicted Bitcoin Close Price (2015)')
    plt.xlabel('Month')
    plt.ylabel('Close Price (USD)')
    plt.xticks(monthly_actual.index)
    plt.legend()
    plt.grid(True)

    # Filter data for 2019-01-01 to 2019-03-31
    df_2019_q1 = data_frame[(data_frame['date'] >= '2019-01-01') & (data_frame['date'] <= '2019-03-31')]

    # Assuming you have a way to predict the close prices for 2019-01-01 to 2019-03-31 (replace with your prediction
    # method) For this example, I'll just use the previous day's close price as a simple prediction.
    df_2019_q1['predicted_close'] = df_2019_q1['close'].shift(1)

    # Create a plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_2019_q1['date'], df_2019_q1['close'], label='Actual Close Price', marker='o')
    plt.plot(df_2019_q1['date'], df_2019_q1['predicted_close'], label='Predicted Close Price', marker='x')
    plt.title('Actual vs. Predicted Bitcoin Close Price (01/01-01/04/2019)')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)


    print("Root Mean Square Error (RMSE):", round(rmse, 4))
    print("Mean Absolute Error (MAE):", round(mae_value, 4))
    print("Training Accuracy (R-squared):", round(train_accuracy, 4))
    print("Testing Accuracy (R-squared):", round(test_accuracy, 4))

    plt.show()


def predict(x, w, b):
    return x.dot(w)+b


def gradient(y_hat, y, x):
    loss = y_hat - y
    dw = x.T.dot(loss) / len(y)
    db = np.sum(loss) / len(y)
    cost = np.sum(loss ** 2) / (2 * len(y))
    return dw, db, cost


def update_weight(w, b, lr, dw, db):
    w_new = w - lr*dw
    b_new = b - lr*db
    return w_new, b_new


def linear_regression_vectorized(x, y, learning_rate, num_iterations):
    _, n_features = x.shape
    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias
    losses = []

    for _ in range(num_iterations):
        y_hat = predict(x, w, b)  # Make predictions
        dw, db, cost = gradient(y_hat, y, x)  # Calculate gradients
        w, b = update_weight(w, b, learning_rate, dw, db)  # Update weights and bias
        losses.append(cost)

    return w, b, losses


def visualize_data_by_year(df):
    # Range of date covered
    df['date'] = pd.to_datetime(df['date'])
    date_range = str(df['date'].dt.date.min()) + ' to ' + str(df['date'].dt.date.max())

    print(date_range)

    # Create y-m-d data
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Get unique years
    unique_years = df['year'].unique()

    # Question 4 -> D
    print(unique_years)
    for year in unique_years:
        year_month_day = df[(df['year'] == year)][['year', 'month', 'day']]
        merged_data = pd.merge(year_month_day, df, on=['year', 'month', 'day'], how='left', validate='many_to_many')

        # Visualize

        plt.figure(figsize=(10, 6))
        plt.plot(merged_data['date'], merged_data['close'])
        plt.title(f'Bitcoin Closing Prices - {year}')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def visualize_candle_stick_chart(df):
    # Filter data for 2019 -2022
    df_filtered = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2022-12-31')]

    # Convert date to matplotlib format
    df_filtered['date'] = df_filtered['date'].map(mdates.date2num)

    # Create the candlestick chart
    fig, ax = plt.subplots(figsize=(20, 6))
    candlestick_ohlc(ax, df_filtered[['date', 'open', 'high', 'low', 'close']].values,
                     width=0.6, colorup='g', colordown='r')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.title('Bitcoin Candlestick Chart (2019 -2022)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD )')
    plt.grid(True)

    plt.show()


run()
