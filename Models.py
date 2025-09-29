from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
import os as os
import numpy as np
import pickle as pkl

def model_selection(model_choice):
    model = None
    if model_choice==1:
        model = LinearRegression()
    elif model_choice==2:
        model = Ridge(alpha=0.1)
    elif model_choice==3:
        model = Lasso(alpha=40)
    elif model_choice==4:
        model = LogisticRegression(max_iter=1000)
    else:
        print("Option not available")
    return model

def model_training(model,x_train,x_test,y_train,y_test):
    if isinstance(model,LogisticRegression):
        model_save_path = os.path.join(f"models",f"{model.__class__.__name__}.pkl")
        avg_price = np.mean(y_train)
        y_train = np.where(y_train < avg_price, 0, 1)
        model.fit(x_train, y_train)
        # prediction
        y_pred = model.predict(x_test)
        with open(model_save_path, "wb") as f:
            pkl.dump(model, f)
        print(f"✅ Model saved at: {model_save_path}")
        return y_pred
    else:
        model.fit(x_train, y_train)
        model_save_path = os.path.join(f"models",f"{model.__class__.__name__}.pkl")
        with open(model_save_path, "wb") as f:
            pkl.dump(model, f)
        print(f"✅ Model saved at: {model_save_path}")
        # prediction
        y_pred = model.predict(x_test)
        return y_pred


