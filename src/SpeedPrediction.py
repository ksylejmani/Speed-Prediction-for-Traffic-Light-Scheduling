# speed_prediction.py
# =======================================================
# Speed Prediction for Traffic Light Scheduling
# This script uses machine learning (LightGBM) to predict 
# car speed based on data from a fleet of taxi vehicles.
# Author: Kadri Sylejmani
# License: MIT
# =======================================================
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap
import lime
import lime.lime_tabular
import matplotlib.patches as mpatches


class SpeedPrediction:
    """Class for predicting car speed based on vehicle fleet data."""
    
    def __init__(self, path):
        """
        Initializes the SpeedPrediction model pipeline.
        
        Parameters:
            path (str): The directory path containing CSV data files.
        """
        self.dataset = self.load_data(path)
        self.convert_device_datetime()
        self.add_time_features()
        self.label_encode_features()
        self.round_coordinates()
        self.drop_zero_speed_rows()
        self.drop_device_datetime()
        self.train_data, self.validation_data = self.split_data()
        self.print_main_features()
        self.model = self.train_lightgbm_model()
        self.evaluate_model()
        self.visualize_feature_correlations(self.dataset)
        self.show_permutation_importance(self.model, self.validation_data)
        self.show_partial_dependence(self.model, self.validation_data, 'Longitude')
        # Uncomment the next line to display 2D partial dependence
        # self.show_2d_partial_dependence(self.model, self.validation_data, ['Longitude', 'Direction'])
        self.explain_with_shap(self.model, self.validation_data)
        self.explain_with_lime(self.model, self.validation_data)

    def load_data(self, path):
        """
        Loads and combines datasets from all CSV files in the specified directory.
        
        Parameters:
            path (str): Directory containing CSV files.
        
        Returns:
            pd.DataFrame: Combined dataset.
        """
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df_list = [pd.read_csv(file, encoding="utf8") for file in all_files]
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    def convert_device_datetime(self):
        """Converts DeviceDateTime to datetime format (handles both 12-hour and 24-hour formats)."""
        for index, row in self.dataset.iterrows():
            try:
                dt = pd.to_datetime(row['DeviceDateTime'], format='%m/%d/%Y %H:%M', errors='coerce')
                if pd.isna(dt):
                    dt = pd.to_datetime(row['DeviceDateTime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                self.dataset.at[index, 'DeviceDateTime'] = dt
            except Exception as e:
                print(f"Error converting index {index}: {row['DeviceDateTime']} -> {e}")
        self.dataset['DeviceDateTime'] = pd.to_datetime(self.dataset['DeviceDateTime'], errors='coerce')

    def add_time_features(self):
        """Adds month, day, hour, and minute features from DeviceDateTime."""
        self.dataset['Month'] = self.dataset['DeviceDateTime'].dt.month
        self.dataset['Day'] = self.dataset['DeviceDateTime'].dt.day
        self.dataset['Hour'] = self.dataset['DeviceDateTime'].dt.hour
        self.dataset['Minute'] = self.dataset['DeviceDateTime'].dt.minute

    def label_encode_features(self):
        """Label encodes the boolean features Di1, Di2, and Di3."""
        label_encoder = LabelEncoder()
        for column in ['Di1', 'Di2', 'Di3']:
            self.dataset[column] = label_encoder.fit_transform(self.dataset[column])

    def round_coordinates(self):
        """Rounds the Longitude and Latitude features to five decimal places."""
        self.dataset['Longitude'] = self.dataset['Longitude'].round(5)
        self.dataset['Latitude'] = self.dataset['Latitude'].round(5)

    def drop_zero_speed_rows(self):
        """Drops rows where the speed is 0."""
        self.dataset = self.dataset[self.dataset['Speed'] != 0]

    def drop_device_datetime(self):
        """Drops the DeviceDateTime column."""
        self.dataset.drop(columns=['DeviceDateTime'], inplace=True)

    def split_data(self):
        """Splits the dataset into training and validation sets (80/20 ratio)."""
        return train_test_split(self.dataset, test_size=0.2, random_state=42)

    def train_lightgbm_model(self):
        """Trains an LGBMRegressor model."""
        X_train = self.train_data.drop(columns=['Speed'])
        y_train = self.train_data['Speed']
        model = LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            n_estimators=1000
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self):
        """Evaluates the model using Mean Squared Error and R^2 Score."""
        X_val = self.validation_data.drop(columns=['Speed'])
        y_val = self.validation_data['Speed']
        predictions = self.model.predict(X_val, num_iteration=self.model.best_iteration_)
        print(f"MSE: {mean_squared_error(y_val, predictions)}")
        print(f"R^2 Score: {r2_score(y_val, predictions)}")

    def print_main_features(self):
        """Displays main dataset features and statistics."""
        print("Dataset Head:\n", self.dataset.head())
        print("\nDataset Info:\n", self.dataset.info())
        print("\nDataset Description:\n", self.dataset.describe())

    def visualize_feature_correlations(self, dataset):
        """Displays a heatmap of feature correlations."""
        plt.figure(figsize=(12, 10))
        correlation_matrix = dataset.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 10}, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

    def show_permutation_importance(self, model, validation_set):
        """Displays permutation feature importance."""
        X_val = validation_set.drop(columns=['Speed'])
        y_val = validation_set['Speed']
        perm = PermutationImportance(model, random_state=42).fit(X_val, y_val)
        print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=X_val.columns.tolist())))

    def show_partial_dependence(self, model, validation_set, feature):
        """Displays partial dependence plot for a given feature."""
        X_val = validation_set.drop(columns=['Speed'])
        PartialDependenceDisplay.from_estimator(model, X_val, [feature])
        plt.title(f'Partial Dependence of {feature}')
        plt.show()

    def show_2d_partial_dependence(self, model, validation_set, features):
        """Displays 2D partial dependence plot for two features."""
        X_val = validation_set.drop(columns=['Speed'])
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(model, X_val, [tuple(features)], ax=ax)
        plt.show()

    def explain_with_shap(self, model, validation_set):
        """Visualizes SHAP values for model interpretation."""
        X_val = validation_set.drop(columns=['Speed'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap.initjs()
        shap.summary_plot(shap_values, X_val)

    def explain_with_lime(self, model, validation_set, instance_index=0):
        """
        Uses LIME to explain an individual prediction from the validation set
        and displays it as a plot with a legend explaining the colors.
        """
        # Drop the target variable 'Speed' to get feature data
        X_val = validation_set.drop(columns=['Speed'])
        feature_names = X_val.columns.tolist()
        
        # Create a LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_val.values, 
            feature_names=feature_names, 
            mode="regression"
        )
        
        # Select an instance to explain
        instance = X_val.iloc[instance_index].values.reshape(1, -1)
        
        # Get the model's prediction function
        predict_fn = lambda x: model.predict(x)
        
        # Generate explanation
        explanation = explainer.explain_instance(instance.flatten(), predict_fn)
        
        # Display explanation as a matplotlib plot
        fig = explanation.as_pyplot_figure()
        
        # Add a legend explaining the colors
        red_patch = mpatches.Patch(color='red', label='Negative Impact on Prediction (Lowers Speed)')
        green_patch = mpatches.Patch(color='green', label='Positive Impact on Prediction (Increases Speed)')
        
        plt.legend(handles=[green_patch, red_patch], loc='best', fontsize=10, frameon=True)
        plt.title(f'LIME Explanation for Instance {instance_index}')
        plt.show()





if __name__ == "__main__":
    path = "data"  # Adjust the path as needed
    speed_prediction = SpeedPrediction(path)
