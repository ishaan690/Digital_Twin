import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# HOW TO LOAD AND USE YOUR TRAINED MODEL
# ==============================================================================
def load_and_use_model(model_path, new_data_sample):
    """
    Loads a trained model from a .joblib file and uses it to make a prediction.
    """
    try:
        # 1. Load the trained model from the file
        model = joblib.load(model_path)
        print(f"Successfully loaded model from '{model_path}'")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please make sure the .joblib file is in the same folder as this script.")
        return

    # 2. "View" the model by inspecting its parameters
    print("\n--- Model Parameters ---")
    print(model.get_params())

    # 3. Use the model to make a prediction on new data
    print("\n--- Making a Prediction on New Data ---")
    
    # --- THIS IS THE FIX ---
    # The new data sample must have the exact same columns as the training data.
    # We get the expected feature names directly from the trained model.
    expected_features = model.feature_names_in_
    
    # Create a DataFrame from the new data sample with the correct columns
    new_data_df = pd.DataFrame([new_data_sample], columns=expected_features)
    
    print("New data sample (formatted for model):")
    print(new_data_df)
    
    # Use the loaded model to predict
    predicted_soh = model.predict(new_data_df)
    
    print(f"\nPredicted SOH (Capacity): {predicted_soh[0]:.4f} Ah")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    
    model_filename = "soh_prediction_model.joblib"

    # --- Create a Hypothetical New Data Sample ---
    # The keys MUST match the feature names from your training data.
    # Your training data had features up to peak 5, so we will match that.
    hypothetical_new_cycle_features = {
        'peak_1_voltage_V': 3.27,
        'peak_1_height_Ah_V': 35.0,
        'peak_1_area_Ah': 0.8,
        'peak_2_voltage_V': 3.35,
        'peak_2_height_Ah_V': 20.0,
        'peak_2_area_Ah': 0.5,
        'peak_3_voltage_V': 0,
        'peak_3_height_Ah_V': 0,
        'peak_3_area_Ah': 0,
        'peak_4_voltage_V': 0,
        'peak_4_height_Ah_V': 0,
        'peak_4_area_Ah': 0,
        'peak_5_voltage_V': 0,
        'peak_5_height_Ah_V': 0,
        'peak_5_area_Ah': 0,
        # The model was not trained on peak 6, so we do not include it.
        
        # Add the load features
        'mean_power_W': 15000.0,
        'peak_power_W': 45000.0,
        'std_dev_power_W': 11000.0,
        'total_energy_Wh': 260.0
    }

    # Call the function to load the model and make a prediction
    load_and_use_model(model_filename, hypothetical_new_cycle_features)
