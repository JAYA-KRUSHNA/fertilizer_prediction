import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def generate_synthetic_agricultural_data(num_samples=1000): # REDUCED TO 1000
    """
    Generate synthetic agricultural dataset for fertilizer requirement prediction.
    """
    np.random.seed(42)

    # Define possible values for categorical features
    soil_textures = ['Sandy', 'Loamy', 'Clay']
    crop_types = ['Wheat', 'Rice', 'Maize', 'Soybean', 'Cotton']
    irrigation_types = ['Rainfed', 'Drip', 'Sprinkler', 'Flood']

    # Generate features
    data = {
        'soil_ph': np.random.uniform(4.0, 9.0, num_samples),
        'soil_texture': np.random.choice(soil_textures, num_samples),
        'nitrogen': np.random.uniform(0, 200, num_samples),
        'phosphorus': np.random.uniform(0, 150, num_samples),
        'potassium': np.random.uniform(0, 200, num_samples),
        'organic_matter': np.random.uniform(0, 10, num_samples),
        'crop_type': np.random.choice(crop_types, num_samples),
        'temperature': np.random.uniform(10, 40, num_samples),
        'humidity': np.random.uniform(20, 90, num_samples),
        'rainfall': np.random.uniform(200, 2000, num_samples),
        'irrigation_type': np.random.choice(irrigation_types, num_samples)
    }

    # Create base fertilizer requirement based on crop type
    crop_base_fertilizer = {
        'Wheat': 120,
        'Rice': 100,
        'Maize': 150,
        'Soybean': 80,
        'Cotton': 110
    }

    # Calculate fertilizer requirement with some randomness and feature influences
    fertilizer_requirements = []
    for i in range(num_samples):
        base = crop_base_fertilizer[data['crop_type'][i]]

        # Adjust based on soil conditions
        ph_adjustment = 1 + (data['soil_ph'][i] - 6.5) * 0.1
        nutrient_adjustment = 1 - (data['nitrogen'][i] + data['phosphorus'][i] + data['potassium'][i]) / 550 * 0.5
        organic_adjustment = 1 - data['organic_matter'][i] / 10 * 0.3

        # Climate adjustments
        temp_adjustment = 1 + (data['temperature'][i] - 25) * 0.01
        humidity_adjustment = 1 - (data['humidity'][i] - 50) * 0.005
        rainfall_adjustment = 1 - (data['rainfall'][i] - 1000) * 0.0001

        # Irrigation adjustment
        irrigation_multipliers = {'Rainfed': 1.2, 'Drip': 0.9, 'Sprinkler': 1.0, 'Flood': 1.1}
        irrigation_adjustment = irrigation_multipliers[data['irrigation_type'][i]]

        # Calculate final requirement with some randomness
        requirement = base * ph_adjustment * nutrient_adjustment * organic_adjustment * \
                     temp_adjustment * humidity_adjustment * rainfall_adjustment * irrigation_adjustment
        requirement += np.random.normal(0, 10)  # Add some noise
        requirement = max(0, min(requirement, 300))  # Clamp between 0 and 300

        fertilizer_requirements.append(requirement)

    data['fertilizer_requirement'] = fertilizer_requirements

    df = pd.DataFrame(data)

    # Encode categorical variables
    le_texture = LabelEncoder()
    le_crop = LabelEncoder()
    le_irrigation = LabelEncoder()

    df['soil_texture_encoded'] = le_texture.fit_transform(df['soil_texture'])
    df['crop_type_encoded'] = le_crop.fit_transform(df['crop_type'])
    df['irrigation_type_encoded'] = le_irrigation.fit_transform(df['irrigation_type'])

    return df

if __name__ == "__main__":
    # Ensure all directories exist before saving
    import os
    os.makedirs('agri_fertilizer_prediction/data', exist_ok=True)
    
    # Generate and save the dataset
    df = generate_synthetic_agricultural_data()
    df.to_csv('agri_fertilizer_prediction/data/synthetic_agricultural_data.csv', index=False)
    print(f"Generated {len(df)} samples of synthetic agricultural data.")
    print("Dataset saved to 'agri_fertilizer_prediction/data/synthetic_agricultural_data.csv'")
    print("\nFeature summary:")
    print(df.describe())
    print("\nCategorical feature distributions:")
    print(df[['soil_texture', 'crop_type', 'irrigation_type']].value_counts())