import sqlite3
import pandas as pd
import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'data', 'agricultural_data.db')
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, 'data', 'synthetic_agricultural_data.csv')


def create_database(db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    """
    Create SQLite database and tables for agricultural data.

    Parameters:
    db_path (str): Path to the database file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create main data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agricultural_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil_ph REAL,
            soil_texture TEXT,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            organic_matter REAL,
            crop_type TEXT,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            irrigation_type TEXT,
            fertilizer_requirement REAL,
            soil_texture_encoded INTEGER,
            crop_type_encoded INTEGER,
            irrigation_type_encoded INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil_ph REAL,
            soil_texture TEXT,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            organic_matter REAL,
            crop_type TEXT,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            irrigation_type TEXT,
            predicted_fertilizer REAL,
            model_used TEXT,
            feature_importance TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create model_performance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            mse REAL,
            rmse REAL,
            mae REAL,
            r2 REAL,
            cv_rmse REAL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database created successfully at {db_path}")


def insert_data_from_csv(csv_path=None, db_path=None):
    csv_path = csv_path or DEFAULT_CSV_PATH
    db_path = db_path or DEFAULT_DB_PATH
    """
    Insert data from CSV file into the database.

    Parameters:
    csv_path (str): Path to the CSV file
    db_path (str): Path to the database file
    """
    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)

    # Insert data in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch.to_sql('agricultural_data', conn, if_exists='append', index=False)
        print(f"Inserted batch {i//batch_size + 1} of {len(df)//batch_size + 1}")

    conn.close()
    print(f"Data inserted successfully from {csv_path}")


def fetch_all_data(db_path=None, limit=None):
    db_path = db_path or DEFAULT_DB_PATH
    """
    Fetch all agricultural data from the database.

    Parameters:
    db_path (str): Path to the database file
    limit (int): Maximum number of records to fetch (optional)

    Returns:
    pd.DataFrame: Agricultural data
    """
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM agricultural_data"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def fetch_data_by_crop(crop_type, db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    """
    Fetch data for a specific crop type.

    Parameters:
    crop_type (str): Crop type to filter by
    db_path (str): Path to the database file

    Returns:
    pd.DataFrame: Filtered agricultural data
    """
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM agricultural_data WHERE crop_type = ?"
    df = pd.read_sql_query(query, conn, params=(crop_type,))

    conn.close()
    return df


def get_statistics(db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    """
    Get basic statistics from the agricultural data.

    Parameters:
    db_path (str): Path to the database file

    Returns:
    dict: Statistics dictionary
    """
    conn = sqlite3.connect(db_path)

    # Get count by crop type
    crop_stats = pd.read_sql_query("""
        SELECT crop_type, COUNT(*) as count,
               AVG(fertilizer_requirement) as avg_fertilizer,
               MIN(fertilizer_requirement) as min_fertilizer,
               MAX(fertilizer_requirement) as max_fertilizer
        FROM agricultural_data
        GROUP BY crop_type
    """, conn)

    # Get overall statistics
    overall_stats = pd.read_sql_query("""
        SELECT COUNT(*) as total_samples,
               AVG(fertilizer_requirement) as avg_fertilizer_req,
               AVG(soil_ph) as avg_soil_ph,
               AVG(temperature) as avg_temperature,
               AVG(rainfall) as avg_rainfall
        FROM agricultural_data
    """, conn)

    conn.close()

    return {
        'crop_statistics': crop_stats.to_dict('records'),
        'overall_statistics': overall_stats.to_dict('records')[0]
    }



def insert_prediction(prediction_data, db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    """
    Insert a prediction record into the database.

    Parameters:
    prediction_data (dict): Prediction data dictionary
    db_path (str): Path to the database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Convert feature importance to JSON string if it exists
    feature_importance_json = None
    if 'feature_importance' in prediction_data:
        import json
        feature_importance_json = json.dumps(prediction_data['feature_importance'])

    cursor.execute('''
        INSERT INTO predictions (
            soil_ph, soil_texture, nitrogen, phosphorus, potassium,
            organic_matter, crop_type, temperature, humidity, rainfall,
            irrigation_type, predicted_fertilizer, model_used, feature_importance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_data['soil_ph'],
        prediction_data['soil_texture'],
        prediction_data['nitrogen'],
        prediction_data['phosphorus'],
        prediction_data['potassium'],
        prediction_data['organic_matter'],
        prediction_data['crop_type'],
        prediction_data['temperature'],
        prediction_data['humidity'],
        prediction_data['rainfall'],
        prediction_data['irrigation_type'],
        prediction_data['predicted_fertilizer'],
        prediction_data['model_used'],
        feature_importance_json
    ))

    conn.commit()
    conn.close()


def insert_model_performance(performance_data, db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    """
    Insert model performance metrics into the database.

    Parameters:
    performance_data (dict): Performance metrics dictionary
    db_path (str): Path to the database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO model_performance (
            model_name, mse, rmse, mae, r2, cv_rmse
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        performance_data['model_name'],
        performance_data['mse'],
        performance_data['rmse'],
        performance_data['mae'],
        performance_data['r2'],
        performance_data['cv_rmse']
    ))

    conn.commit()
    conn.close()



def get_recent_predictions(limit=10, db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    """
    Get recent predictions from the database.

    Parameters:
    limit (int): Number of recent predictions to fetch
    db_path (str): Path to the database file

    Returns:
    pd.DataFrame: Recent predictions
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT * FROM predictions
        ORDER BY prediction_date DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))

    # Parse feature_importance JSON if it exists
    if 'feature_importance' in df.columns:
        import json
        df['feature_importance'] = df['feature_importance'].apply(
            lambda x: json.loads(x) if x and x != 'None' else None
        )

    conn.close()
    return df

if __name__ == "__main__":
    # Create database
    create_database()


    # Insert data from CSV (assuming CSV exists)
    csv_path = DEFAULT_CSV_PATH
    if os.path.exists(csv_path):
        insert_data_from_csv()
        print("Data inserted into database")
    else:
        print(f"CSV file not found at {csv_path}. Please run data_generator.py first.")

    # Test database functions
    stats = get_statistics()
    print("Database statistics:")
    print(f"Total samples: {stats['overall_statistics']['total_samples']}")
    print(f"Average fertilizer requirement: {stats['overall_statistics']['avg_fertilizer_req']:.2f} kg/ha")

    print("\nCrop-wise statistics:")
    for crop in stats['crop_statistics']:
        print(f"{crop['crop_type']}: {crop['count']} samples, Avg fertilizer: {crop['avg_fertilizer']:.2f} kg/ha")
