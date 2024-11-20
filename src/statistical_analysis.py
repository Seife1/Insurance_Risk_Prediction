import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelPipeline:
    def __init__ (self):
        
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
        self.results = {}     # Store the results of the models
        
    # Function to train the models
    def train_evaluate_models(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.predictions = model.predict(self.X_test)
            
            # Use MSE and R2 for regression tasks
            self.mse = mean_squared_error(self.y_test, self.predictions)
            self.r2 = r2_score(self.y_test, self.predictions)
            
            self.results[model_name] = (self.mse, self.r2)  # Store both MSE and R2
        print('Training and Evaluation complete')
        return self.results
    
    # Function to plot the evaluation results
    def plot_evaluation_results(self):
        self.results_df = pd.DataFrame(self.results, index=['MSE', 'R2'])
        self.results_df = self.results_df.T
        self.results_df.plot(kind='bar')
        plt.title('Model Evaluation Results')
        plt.ylabel('Value')
        plt.xlabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.show()
        
    # Function to get the best model
    def get_best_model(self):
        self.best_model = max(self.results, key=self.results.get)
        return self.best_model
    
    # Function to get the best model evaluation results
    def get_best_model_evaluation(self):
        self.best_model = self.get_best_model()
        self.best_model_evaluation = self.results[self.best_model]
        print(f'The best model is {self.best_model} with an evaluation of {self.best_model_evaluation}')
        return self.best_model_evaluation
    
    
    def get_best_model_feature_importances(self):
        # Check if the best model is tree-based
        best_model_instance = self.models[self.best_model]
        if hasattr(best_model_instance, "feature_importances_"):
            self.best_model_feature_importances = best_model_instance.feature_importances_
            return self.best_model_feature_importances
        else:
            # Fallback to a tree-based model (Random Forest)
            tree_model_name = "Random Forest Regressor"
            tree_model_instance = self.models[tree_model_name]
            tree_model_instance.fit(self.X_train, self.y_train)  # Ensure the model is trained
            self.best_model_feature_importances = tree_model_instance.feature_importances_
            print(f"Feature importances computed using {tree_model_name}.")
            return self.best_model_feature_importances

    
    def feature_importance(self):
        try:
            self.best_model_feature_importances = self.get_best_model_feature_importances()
            self.feature_importances_df = pd.DataFrame(
                self.best_model_feature_importances, 
                index=self.X_train.columns, 
                columns=['Feature Importance']
            )
            self.feature_importances_df = self.feature_importances_df.sort_values(by='Feature Importance', ascending=False)
            
            # Plot feature importances
            self.feature_importances_df.plot(kind='bar', figsize=(10, 6))
            plt.title("Feature Importances")
            plt.ylabel("Importance")
            plt.xlabel("Features")
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()
            
            return self.feature_importances_df
        except Exception as e:
            print(f"Error in computing feature importances: {e}")

    
    # Function to save the best model
    def save_best_model(self):
        self.best_model = self.get_best_model()
        self.best_model = self.models[self.best_model]
        self.name = self.best_model.replace(" ", "_").lower()
        joblib.dump(self.best_model, f'../models/{self.name}_best_model.pkl')
        print(f'Model saved as {self.name}_best_model.pkl')
        