#applied onehot encoding cause using label encoding may get model biased as it uses number and model may consider it as series

import os as os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def datapreprocessing(dataset):
    X = dataset.drop(['selling_price'],axis=1)
    Y = dataset['selling_price']
    print(f"features:{X.columns}")
    print(f"targets:{Y}")

    scaler=MinMaxScaler()
    X.drop(['car_name'],axis=1,inplace=True)
    X.drop(X.columns[0], axis=1, inplace=True)
    print(f"Columns after removing:{X.columns}")

    print("Data preprocessing started ✅")
    categorical_features = [ 'brand', 'model', 'seller_type', 'fuel_type','transmission_type']

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    print(X.head())

    num_cols = ['vehicle_age', 'km_driven','mileage', 'engine', 'max_power','seats']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print(X.head())
    print("Data preprocessing Completed✅")

    x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle=True,test_size=0.2)
    return x_train, x_test, y_train, y_test






