import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(file_path='data/synthetic_agricultural_data.csv'):
    """
    Load and perform initial cleaning of the agricultural dataset.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    pd.DataFrame: Cleaned dataset
    """
    # Load the data
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
    print(f"Initial shape: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    # Handle missing values (if any)
    if df.isnull().sum().sum() > 0:
        print("Missing values found:")
        print(df.isnull().sum())
        # Fill missing values with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        print("Missing values filled with median values")
    else:
        print("No missing values found")

    return df

def handle_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Handle outliers in specified columns using IQR or Z-score method.

    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to handle outliers for
    method (str): Method to use ('iqr' or 'zscore')
    threshold (float): Threshold for outlier detection

    Returns:
    pd.DataFrame: Dataframe with outliers handled
    """
    df_clean = df.copy()

    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Cap outliers
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])

        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean[col] = np.where(z_scores > threshold,
                                   df_clean[col].median(),
                                   df_clean[col])

    print(f"Outliers handled for columns: {columns} using {method} method")
    return df_clean

def preprocess_data(df):
    """
    Perform comprehensive data preprocessing for model training.

    Parameters:
    df (pd.DataFrame): Input dataframe

    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Select features and target
    feature_cols = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
                   'temperature', 'humidity', 'rainfall',
                   'soil_texture_encoded', 'crop_type_encoded', 'irrigation_type_encoded']
    target_col = 'fertilizer_requirement'

    X = df[feature_cols]
    y = df[target_col]

    # Handle outliers for numerical features
    numerical_features = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
                         'temperature', 'humidity', 'rainfall']
    X = handle_outliers(X, numerical_features, method='iqr', threshold=1.5)

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Feature columns: {list(X_train.columns)}")

    return X_train, X_test, y_train, y_test, scaler

def create_visualizations(df, save_path='agri_fertilizer_prediction/static/images/'):
    """
    Create exploratory data visualizations.

    Parameters:
    df (pd.DataFrame): Input dataframe
    save_path (str): Path to save plots
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Distribution of fertilizer requirement
    sns.histplot(df['fertilizer_requirement'], kde=True, ax=axes[0,0])
    axes[0,0].set_title('Distribution of Fertilizer Requirement')
    axes[0,0].set_xlabel('Fertilizer Requirement (kg/ha)')

    # Fertilizer requirement by crop type
    sns.boxplot(x='crop_type', y='fertilizer_requirement', data=df, ax=axes[0,1])
    axes[0,1].set_title('Fertilizer Requirement by Crop Type')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Soil pH distribution
    sns.histplot(df['soil_ph'], kde=True, ax=axes[0,2])
    axes[0,2].set_title('Soil pH Distribution')
    axes[0,2].set_xlabel('Soil pH')

    # Correlation heatmap
    numerical_cols = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
                     'temperature', 'humidity', 'rainfall', 'fertilizer_requirement']
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,0])
    axes[1,0].set_title('Correlation Matrix')

    # Temperature vs Fertilizer requirement
    sns.scatterplot(x='temperature', y='fertilizer_requirement', data=df, ax=axes[1,1])
    axes[1,1].set_title('Temperature vs Fertilizer Requirement')
    axes[1,1].set_xlabel('Temperature (°C)')
    axes[1,1].set_ylabel('Fertilizer Requirement (kg/ha)')

    # Rainfall vs Fertilizer requirement
    sns.scatterplot(x='rainfall', y='fertilizer_requirement', data=df, ax=axes[1,2])
    axes[1,2].set_title('Rainfall vs Fertilizer Requirement')
    axes[1,2].set_xlabel('Rainfall (mm)')
    axes[1,2].set_ylabel('Fertilizer Requirement (kg/ha)')

    plt.tight_layout()
    plt.savefig(f'{save_path}data_visualization.png', dpi=300, bbox_inches='tight')
    print(f"Data visualization saved to {save_path}data_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Load and clean data
    df = load_and_clean_data()

    # Create visualizations
    create_visualizations(df)

    # Preprocess data for modeling
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Save preprocessed data
    train_data = X_train.copy()
    train_data['fertilizer_requirement'] = y_train
    test_data = X_test.copy()
    test_data['fertilizer_requirement'] = y_test

    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Data preprocessing completed!")
    print("Train and test data saved to 'data/' directory")
    print("Scaler saved to 'models/scaler.pkl'")
