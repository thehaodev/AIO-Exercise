import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def run():
    df = pd.read_csv("../data/SalesPrediction.csv")

    # Check null
    df.info()

    # One hot encoding Influencer
    encode_species = pd.get_dummies(df.Influencer, prefix='Influencer')

    # Label Encoding
    df["Influencer"] = df["Influencer"].astype('category')
    df = pd.concat([df, encode_species], axis='columns')

    # Handle Null values
    mean_values = df.select_dtypes(include='number').mean().to_dict()
    df.fillna(mean_values, inplace=True)
    df.drop(columns=["Influencer"], inplace=True)
    print(df)

    # Get features
    x = df[['TV', 'Radio', 'Social Media', 'Influencer_Macro',
            'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']]
    y = df[['Sales']]

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=0
    )

    # Feature Scaling
    scaler = StandardScaler()
    x_train_processed = scaler.fit_transform(x_train)
    print(scaler.mean_[0])

    # Polynomial Features
    x_test_processed = scaler.fit_transform(x_test)
    poly_features = PolynomialFeatures(degree=2, interaction_only=False)
    x_train_poly = poly_features.fit_transform(x_train_processed)
    x_test_poly = poly_features.transform(x_test_processed)

    poly_model = LinearRegression()
    poly_model.fit(x_train_poly, y_train)
    preds = poly_model.predict(x_test_poly)
    r2_score(y_test, preds)


run()
