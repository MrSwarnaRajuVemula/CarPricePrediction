from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import numpy as np

def model_evaluation(model, x_train, x_test, y_train, y_test, y_pred):
    if isinstance(model, LogisticRegression):
        avg_price = np.mean(y_train)
        y_train_bin = np.where(y_train < avg_price, 0, 1)
        y_test_bin = np.where(y_test < avg_price, 0, 1)
        accuracy = accuracy_score(y_test_bin, y_pred)
        cm = confusion_matrix(y_test_bin, y_pred)
        report = classification_report(y_test_bin, y_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
    else:
        # Regression Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2_scr = r2_score(y_test, y_pred)
        n = len(y_test)
        p = x_train.shape[1]
        adjusted_r2 = 1 - (1 - r2_scr) * (n - 1) / (n - p - 1)
        mean_mse_k_fold = -np.mean(cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=5))

        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R2_score: {r2_scr}")
        print(f"Adjusted_R2_score: {adjusted_r2}")
        print(f"Mean_MSE_K_Fold: {mean_mse_k_fold}")
