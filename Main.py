import os as os
from dotenv import load_dotenv
from DataPreprocessing import datapreprocessing
from Dataloading import load_dataset
from EDA import exploratory_data_analysis
from Evaluations import model_evaluation
from Models import model_selection, model_training

load_dotenv()


if __name__=="__main__":
    dataset_path=os.getenv("DATASET_PATH")

    #Data Loading
    dataset=load_dataset(dataset_path)

    #EDA
    exploratory_data_analysis(dataset)

    #Data Preprocessing
    x_train,x_test,y_train,y_test=datapreprocessing(dataset)

    #Model Selection
    model_choice = int(input("Please Select The Model:\t1.Linear Regression\t2.Ridge Regression\t3.Lasso Regression\t4.Logistic Regression\n"))
    model=model_selection(model_choice)
    if model==None:
        print("Model Not Compiled")

    #Model Training
    y_pred = model_training(model,x_train,x_test,y_train,y_test)

    #Model Evaluation
    model_evaluation(model,x_train,x_test,y_train,y_test,y_pred)



