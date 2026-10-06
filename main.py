# ==============================================================================
# FINAL SCRIPT: EV BATTERY SOH PREDICTION
# ==============================================================================
import matplotlib
matplotlib.use('Agg') # Use the non-interactive 'Agg' backend for saving files

import pandas as pd
import numpy as np
import scipy.io
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
import joblib

# ==============================================================================
# DATA PARSING FUNCTIONS
# ==============================================================================
def parse_characterization_mat_file(file_path):
    """
    Loads and parses a CALCE battery characterization (charge/discharge) .mat file.
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        print(f"Successfully loaded '{os.path.basename(file_path)}'.")
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None

    data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    if not data_key: return None
    print(f"Found main data key: '{data_key}'")
    
    try:
        all_cycles_raw = mat_data[data_key][0]
    except IndexError: return None

    cycle_dataframes = []
    for i, cycle_raw in enumerate(all_cycles_raw):
        try:
            voltage = cycle_raw['voltage'].flatten()
            current = cycle_raw['current'].flatten()
            temperature = cycle_raw['Ts'].flatten()
            time = cycle_raw['time'].flatten()
            cycle_type = 'charge' if np.mean(current) > 0 else 'discharge'
            chg_ah = cycle_raw['chgAh'].flatten()[0] if 'chgAh' in cycle_raw.dtype.names else np.nan
            dis_ah = cycle_raw['disAh'].flatten()[0] if 'disAh' in cycle_raw.dtype.names else np.nan

            df = pd.DataFrame({
                'time': time, 'voltage': voltage, 'current': current, 'temperature': temperature,
            })
            df['cycle_number'] = i + 1
            df['cycle_type'] = cycle_type
            df['charge_capacity_Ah'] = chg_ah
            df['discharge_capacity_Ah'] = dis_ah
            cycle_dataframes.append(df)
        except Exception as e:
            print(f"Warning: Could not parse cycle {i+1}. Error: {e}. Available fields: {cycle_raw.dtype.names}. Skipping.")
    print(f"Successfully parsed {len(cycle_dataframes)} cycles from this file.")
    return cycle_dataframes

def parse_dynamic_mat_file(file_path):
    """
    Loads and parses the CALCE dynamic test .mat file by concatenating the data
    from script1, script2, and script3.
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        print(f"Successfully loaded dynamic data file '{os.path.basename(file_path)}'.")
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None

    data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    if not data_key: return None
    print(f"Found main data key: '{data_key}'")

    try:
        main_struct = mat_data[data_key][0, 0]
        script_fields = ['script1', 'script2', 'script3']
        
        all_script_dfs = []
        for field in script_fields:
            print(f"Extracting data from '{field}'...")
            script_data = main_struct[field][0, 0]
            df = pd.DataFrame({
                'time': script_data['time'].flatten(),
                'voltage': script_data['voltage'].flatten(),
                'current': script_data['current'].flatten(),
            })
            all_script_dfs.append(df)
        
        # Concatenate all parts into a single DataFrame
        dynamic_df = pd.concat(all_script_dfs, ignore_index=True)
        print(f"Successfully concatenated dynamic data. Total rows: {len(dynamic_df)}")
        return dynamic_df

    except Exception as e:
        print(f"An error occurred while parsing the dynamic file: {e}")
        return None

# ==============================================================================
# FEATURE ENGINEERING AND MODELING FUNCTIONS
# ==============================================================================
def perform_ica(cycle_df, cycle_number, script_dir):
    """
    Performs Incremental Capacity Analysis and saves a plot for diagnostics.
    """
    time_h = cycle_df['time'] / 3600.0
    dt = np.diff(time_h, prepend=0)
    dQ = np.abs(cycle_df['current']) * dt
    cycle_df['capacity_ah'] = dQ.cumsum()

    window_length, polyorder = 51, 2
    if len(cycle_df) < window_length:
        window_length = len(cycle_df) - 1 if len(cycle_df) % 2 == 0 else len(cycle_df)
        if window_length < 3: return None
    
    cycle_df['voltage_filtered'] = savgol_filter(cycle_df['voltage'], window_length, polyorder)
    v_filtered = cycle_df['voltage_filtered']
    q_raw = cycle_df['capacity_ah']
    v_unique, idx_unique = np.unique(v_filtered, return_index=True)
    q_unique = q_raw.iloc[idx_unique]

    if len(v_unique) < 2: return None

    interp_func = interp1d(v_unique, q_unique, kind='linear', fill_value="extrapolate", bounds_error=False)
    voltage_uniform = np.linspace(v_unique.min(), v_unique.max(), num=2000)
    capacity_interpolated = interp_func(voltage_uniform)

    dQdV = np.gradient(capacity_interpolated, voltage_uniform)
    dQdV_filtered = savgol_filter(dQdV, 41, 2)
    max_dqdv = np.max(dQdV_filtered)
    adaptive_height = max_dqdv * 0.05
    peaks, properties = find_peaks(dQdV_filtered, height=adaptive_height, prominence=0.01, width=5)
    
    if cycle_number <= 5:
        print(f"Diagnostic for Cycle {cycle_number}: Max dQ/dV value is {max_dqdv:.4f}")
        plt.figure(figsize=(12, 7))
        plt.plot(voltage_uniform, dQdV_filtered, label='Filtered dQ/dV Curve')
        if len(peaks) > 0:
            plt.plot(voltage_uniform[peaks], dQdV_filtered[peaks], 'x', color='red', markersize=10, label=f'Found {len(peaks)} Peaks')
        plt.title(f'Incremental Capacity Analysis (dQ/dV) for Cycle {cycle_number}')
        plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah/V)'); plt.grid(True); plt.legend()
        plot_filename = f"ica_curve_cycle_{cycle_number}.png"
        full_plot_path = os.path.join(script_dir, plot_filename)
        plt.savefig(full_plot_path); plt.close()
        print(f"Saved ICA plot to '{full_plot_path}'")

    if len(peaks) == 0: return None

    peak_features = []
    for i, peak_index in enumerate(peaks):
        peak_height = properties['peak_heights'][i]
        peak_voltage = voltage_uniform[peak_index]
        start_idx, end_idx = int(properties['left_ips'][i]), int(properties['right_ips'][i])
        if end_idx >= len(voltage_uniform): end_idx = len(voltage_uniform) - 1
        peak_area = trapezoid(dQdV_filtered[start_idx:end_idx+1], voltage_uniform[start_idx:end_idx+1])
        peak_features.append({
            f'peak_{i+1}_voltage_V': peak_voltage, f'peak_{i+1}_height_Ah_V': peak_height, f'peak_{i+1}_area_Ah': peak_area,
        })
    return {k: v for d in peak_features for k, v in d.items()}

def calculate_load_features_from_dynamic_profile(dynamic_df):
    """
    Calculates descriptive statistical features from a real dynamic data profile.
    """
    # Power (W) = Voltage * Current. abs() is used in case discharge current is negative.
    dynamic_df['power_W'] = dynamic_df['voltage'] * abs(dynamic_df['current'])
    time_step_s = dynamic_df['time'].diff().mean()
    time_step_h = time_step_s / 3600.0
    
    return {
        'mean_power_W': dynamic_df['power_W'].mean(),
        'peak_power_W': dynamic_df['power_W'].max(),
        'std_dev_power_W': dynamic_df['power_W'].std(),
        'total_energy_Wh': dynamic_df['power_W'].sum() * time_step_h,
    }

def create_fused_dataset(health_indicators_df, load_features):
    fused_df = health_indicators_df.copy()
    for col, value in load_features.items():
        fused_df[col] = value
    return fused_df

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    print("="*60 + f"\nScript is running in this folder: {script_dir}\n" + "="*60)

    # --- Part 1: Load Characterization Data ---
    lifecycle_files = [
        "A002_CCCV_1C.mat", "A002_CCCV_2C.mat", "A002_CCCV_3C.mat", "A002_CCCV_4C.mat",
    ]
    all_cycles_dfs = []
    for file_name in lifecycle_files:
        file_path = os.path.join(script_dir, 'data', file_name)
        if os.path.exists(file_path):
            parsed_dfs = parse_characterization_mat_file(file_path)
            if parsed_dfs: all_cycles_dfs.extend(parsed_dfs)
        else:
            print(f"\nWarning: Characterization file not found: '{file_path}'. Skipping.")

    # --- Part 2: Load Dynamic Profile Data ---
    dynamic_mat_file = "A002_DYN_05_N15.mat" # <-- YOUR DYNAMIC DATA FILE
    dynamic_file_path = os.path.join(script_dir, 'data', dynamic_mat_file)
    dynamic_df = parse_dynamic_mat_file(dynamic_file_path)

    if all_cycles_dfs and dynamic_df is not None:
        print(f"\nTotal characterization cycles parsed: {len(all_cycles_dfs)}")
        
        # --- Part 3: Extract Electrochemical Features ---
        all_health_indicators = []
        for i, cycle_df in enumerate(all_cycles_dfs):
            cycle_df['global_cycle_number'] = i + 1
        for cycle_df in all_cycles_dfs:
            if cycle_df['cycle_type'].iloc[0] == 'charge':
                cycle_num = cycle_df['global_cycle_number'].iloc[0]
                print(f"\nAnalyzing Cycle {cycle_num} for ICA...")
                ica_features = perform_ica(cycle_df, cycle_num, script_dir)
                if ica_features:
                    ica_features['cycle_number'] = cycle_num
                    ica_features['soh_capacity_Ah'] = cycle_df['charge_capacity_Ah'].iloc[0]
                    all_health_indicators.append(ica_features)
        
        if not all_health_indicators:
            print("\nError: No valid ICA features could be extracted.")
        else:
            health_indicators_df = pd.DataFrame(all_health_indicators).set_index('cycle_number')
            health_indicators_df.fillna(0, inplace=True)
            print("\nExtracted Health Indicators (from ICA):"); print(health_indicators_df)

            # --- Part 4: Extract Load Features from Real Dynamic Data ---
            print("\n--- Calculating Load Features from Real Dynamic Profile ---")
            load_features = calculate_load_features_from_dynamic_profile(dynamic_df)
            print("Calculated Load Features:"); print(load_features)

            # --- Part 5: Fuse Datasets ---
            final_fused_dataset = create_fused_dataset(health_indicators_df, load_features)
            print("\n" + "="*60 + "\n  Final Fused Dataset (Ready for ML Training)\n" + "="*60); print(final_fused_dataset)
            
            # --- Part 6: Train Model ---
            print("\n" + "="*60 + "\n  STEP 6: TRAINING A PREDICTIVE SOH MODEL\n" + "="*60)
            final_fused_dataset.dropna(subset=['soh_capacity_Ah'], inplace=True)

            if len(final_fused_dataset) < 2:
                print("Not enough data to train a model.")
            else:
                X = final_fused_dataset.drop('soh_capacity_Ah', axis=1)
                y = final_fused_dataset['soh_capacity_Ah']
                print(f"Training RandomForestRegressor model on all {len(X)} available samples...")
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, y)
                print("Model training complete.")

                model_filename = "soh_prediction_model.joblib"
                joblib.dump(model, model_filename)
                print(f"\nModel saved to '{model_filename}'")

            print("\nScript finished successfully.")
    else:
        print("\nScript finished with errors: Could not parse the necessary data files.")
