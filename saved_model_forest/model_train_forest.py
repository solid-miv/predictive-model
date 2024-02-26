import os

import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from joblib import dump


# load and preprocess the data
def load_preprocess_data():
    charges_data = pd.read_csv(os.path.join(os.getcwd(), "data/medical_cost.csv"))

    # drop unnecessary column
    X2 = charges_data.drop("Id", axis=1)

    # get rid of string with label encoder
    label_encoder = LabelEncoder()
    X2['sex'] = label_encoder.fit_transform(X2['sex'])
    X2['smoker'] = label_encoder.fit_transform(X2['smoker'])

    # get rid of string with one hot encoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(X2[['region']]).toarray())
    enc_df.rename(columns={3:'region_southwest', 2:'region_southeast', 
                           1:'region_northwest', 0:'region_northeast'},
                           inplace=True)
    
    # drop the region-column (string variant)
    X2 = X2.drop('region', axis=1)

    # join two dataframes
    X2 = enc_df.join(X2)

    # extract the values from a dataframe
    X, y = X2.iloc[:, :-1].values, X2.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test

# build and evaluate the model
def build_model(X_train, X_test, y_train, y_test):
    # Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=5, random_state=0, max_depth=5)
    regressor.fit(X_train, y_train)

    Y_pred = regressor.predict(X_test)     

    # evaluate the model
    mse = mean_squared_error(y_test, Y_pred)
    print(f"mse: {mse}")

    mae = mean_absolute_error(y_test, Y_pred)
    print(f"mae: {mae}")

    r2 = r2_score(y_test, Y_pred)
    print(f"r2: {r2}")

    return regressor

# save the model
def save_model(regressor, filename=os.path.join(os.getcwd(), f"saved_model_forest/forest_model.joblib")):
    dump(regressor, filename)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_preprocess_data()
    regressor = build_model(X_train, X_test, y_train, y_test)
    save_model(regressor)