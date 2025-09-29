import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def exploratory_data_analysis(dataset):
    print("Exploratory Data Analysis Started✅")

    print(dataset.columns)
    print(dataset.info())
    print(dataset.describe())
    print(dataset.describe(include='object'))
    print(f"Null Values:{dataset.isnull().sum()}")

    print("Exploratory Data Analysis Completed✅")



