import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """
    Load preprocessed training and test data.
    """
    # Assuming data/train_data.csv and data/test_data.csv are present
    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/test_data.csv')

    X_train = train_data.drop('fertilizer_requirement', axis=1)
    y_train = train_data['fertilizer_requirement']
    X_test = test_data.drop('fertilizer_requirement', axis=1)
    y_test = test_data['fertilizer_requirement']

    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Train multiple regression models with simplified parameters for speed.
    """
    models = {}

    # Random Forest (Simplified parameters - NO GridSearchCV)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    print("RandomForest trained with fixed, fast parameters.")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['LinearRegression'] = lr

    # Support Vector Regression (Simplified parameters - NO GridSearchCV)
    svr = SVR(C=10, gamma='scale', kernel='rbf')
    svr.fit(X_train, y_train)
    models['SVR'] = svr
    print("SVR trained with fixed, fast parameters.")

    return models

# The rest of the functions (evaluate_models, extract_feature_importance, plot_model_comparison, plot_predictions_vs_actual, save_best_model) remain the same.

def evaluate_models(models, X_test, y_test):
    # ... (Original evaluation code) ...
    results = []
    # ... (Original evaluation code) ...
    for name, model in models.items():
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='neg_mean_squared_error') # Reduced CV to 3 for speed
        cv_rmse = np.sqrt(-cv_scores.mean())

        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV_RMSE': cv_rmse
        })

    return pd.DataFrame(results)

def extract_feature_importance(models, feature_names, save_path=None):
    # ... (Original feature importance code) ...
    if 'RandomForest' in models:
        rf_model = models['RandomForest']

        # Get feature importances
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances (Random Forest)")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

        # Print feature importance ranking
        print("\nFeature Importance Ranking:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")

def plot_model_comparison(results_df, save_path=None):
    # ... (Original model comparison code) ...
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # RMSE comparison
    sns.barplot(x='Model', y='RMSE', data=results_df, ax=axes[0,0])
    axes[0,0].set_title('RMSE Comparison')
    axes[0,0].tick_params(axis='x', rotation=45)

    # R2 comparison
    sns.barplot(x='Model', y='R2', data=results_df, ax=axes[0,1])
    axes[0,1].set_title('R² Score Comparison')
    axes[0,1].tick_params(axis='x', rotation=45)

    # MAE comparison
    sns.barplot(x='Model', y='MAE', data=results_df, ax=axes[1,0])
    axes[1,0].set_title('MAE Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)

    # CV RMSE comparison
    sns.barplot(x='Model', y='CV_RMSE', data=results_df, ax=axes[1,1])
    axes[1,1].set_title('Cross-Validation RMSE Comparison')
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")

    plt.show()

def plot_predictions_vs_actual(models, X_test, y_test, save_path=None):
    # ... (Original prediction vs actual plot code) ...
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))

    if len(models) == 1:
        axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)

        axes[i].scatter(y_test, y_pred, alpha=0.5)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual Fertilizer Requirement')
        axes[i].set_ylabel('Predicted Fertilizer Requirement')
        axes[i].set_title(f'{name}: Predicted vs Actual')
        axes[i].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Predictions vs Actual plot saved to {save_path}")

    plt.show()

def save_best_model(models, results_df, model_path='models/best_model.pkl'):
    # ... (Original save best model code) ...
    best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
    best_model = models[best_model_name]

    joblib.dump(best_model, model_path)
    print(f"Best model ({best_model_name}) saved to {model_path}")

    return best_model_name, best_model


if __name__ == "__main__":
    # Ensure all directories exist before saving
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('agri_fertilizer_prediction/static/images', exist_ok=True)
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    feature_names = X_train.columns.tolist()

    # Train models
    print("Training models...")
    models = train_models(X_train, y_train)

    # Evaluate models
    print("Evaluating models...")
    results_df = evaluate_models(models, X_test, y_test)
    print("\nModel Evaluation Results:")
    print(results_df)

    # Extract feature importance
    print("Extracting feature importance...")
    extract_feature_importance(models, feature_names, save_path='agri_fertilizer_prediction/static/images/feature_importance.png')

    # Create comparison plots
    plot_model_comparison(results_df, save_path='agri_fertilizer_prediction/static/images/model_comparison.png')
    plot_predictions_vs_actual(models, X_test, y_test, save_path='agri_fertilizer_prediction/static/images/predictions_vs_actual.png')

    # Save best model
    best_model_name, best_model = save_best_model(models, results_df)

    # Save results to CSV
    results_df.to_csv('models/model_evaluation_results.csv', index=False)
    print("Model evaluation results saved to 'models/model_evaluation_results.csv'")

    print(f"\nBest performing model: {best_model_name}")
    print("Training and evaluation completed!")