import os as os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

dataset_path = os.getenv("DATASET_PATH")
def load_dataset(dataset_path):
    print("Data Loading Started✅")

    dataset=pd.read_csv(dataset_path)
    print(dataset.head())
    print("Data Loading Successfull✅")

    return dataset
