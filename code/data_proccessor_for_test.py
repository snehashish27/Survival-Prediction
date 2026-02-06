import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(script_dir, '..', 'data','set-a','set-a')  
TEST_DATA_PATH = os.path.join(script_dir, '..', 'data','set-b','set-b')          
TEST_OUTCOME_PATH = os.path.join(script_dir, '..', 'data', 'Outcomes-b.txt')

TS_FEATURES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium', 
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP', 
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight'
]
STATIC_FEATURES = ['Age', 'Gender', 'Height', 'ICUType']

def process_patient(path):
    df = pd.read_csv(path)
    
    first_val = df['Time'].dropna().iloc[0] if not df['Time'].dropna().empty else 0
    if isinstance(first_val, str) and ':' in first_val:
        df['Time'] = df['Time'].apply(lambda x: int(str(x).split(':')[0])*60 + int(str(x).split(':')[1]))
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df[df['Time'] <= 1440] # First 24h
    

    df['TimeBin'] = np.floor(df['Time'] / 60).astype(int)
    df['TimeBin'] = df['TimeBin'].clip(0, 23)
   
    static_data = df[df['Parameter'].isin(STATIC_FEATURES)].groupby('Parameter')['Value'].max()
    ts_data = df[df['Parameter'].isin(TS_FEATURES)]
    
    df_pivot = ts_data.pivot_table(index='TimeBin', columns='Parameter', values='Value', aggfunc='mean')
    df_pivot = df_pivot.reindex(index=range(24), columns=TS_FEATURES)
    
    df_pivot = df_pivot.ffill()
    
    static_df = pd.DataFrame(np.nan, index=range(24), columns=STATIC_FEATURES)
    for feat in STATIC_FEATURES:
        if feat in static_data:
            static_df[feat] = static_data[feat]
            
    final_df = pd.concat([df_pivot, static_df], axis=1)
    return final_df.values

def main():
    print(f"Reading RAW training data from {TRAIN_DATA_PATH} to calculate stats...")
    train_files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.txt')]
    X_train_raw_list = []
    

    for f in train_files:
        try:
            p_matrix = process_patient(os.path.join(TRAIN_DATA_PATH, f))
            X_train_raw_list.append(p_matrix)
        except:
            continue
            
    X_train_raw = np.array(X_train_raw_list)
    N, T, F = X_train_raw.shape
    X_train_flat = X_train_raw.reshape(N * T, F)
    
 
    train_means = np.nanmean(X_train_flat, axis=0)
    
    inds = np.where(np.isnan(X_train_flat))
    X_train_flat[inds] = np.take(train_means, inds[1])
    
   
    print("Fitting Scaler on Raw Training Data...")
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    
    print(f"Processing Test Data from {TEST_DATA_PATH}...")
    outcomes = pd.read_csv(TEST_OUTCOME_PATH)
    outcomes.set_index('RecordID', inplace=True)
    test_ids = outcomes.index.tolist()
    
    X_test_list = []
    y_test_list = []
    
    for rid in test_ids:
        filepath = os.path.join(TEST_DATA_PATH, f"{rid}.txt")
        if not os.path.exists(filepath):
            continue
        try:
            p_matrix = process_patient(filepath)
            X_test_list.append(p_matrix)
            y_test_list.append(outcomes.loc[rid, 'In-hospital_death'])
        except:
            continue

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    N_test, T_test, F_test = X_test.shape
    X_test_flat = X_test.reshape(N_test * T_test, F_test)
    
    
    inds_test = np.where(np.isnan(X_test_flat))
    X_test_flat[inds_test] = np.take(train_means, inds_test[1])
    

    X_test_flat_scaled = scaler.transform(X_test_flat)
    
    X_test_final = X_test_flat_scaled.reshape(N_test, T_test, F_test)
    

    if not os.path.exists('Processed_Data'):
        os.makedirs('Processed_Data')
        
    np.save('Processed_Data/X_test.npy', X_test_final)
    np.save('Processed_Data/y_test.npy', y_test)
    
    print("Done.")
    print(f"X_test shape: {X_test_final.shape}")
    print(f"y_test shape: {y_test.shape}")

if __name__ == "__main__":
    main()