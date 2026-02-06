import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, "Processed_Data"), exist_ok=True)
DATA_PATH = os.path.join(script_dir, '..', 'data','set-a','set-a')
OUTCOME_PATH = os.path.join(script_dir, '..', 'data', 'Outcomes-a.txt')

print(f"DEBUG: Reading data from: {DATA_PATH}")

TS_FEATURES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium', 
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP', 
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight'
]

STATIC_FEATURES = ['Age', 'Gender', 'Height', 'ICUType']

def get_labels(outcome_file):
    df = pd.read_csv(outcome_file)
    df.set_index('RecordID', inplace=True)
    return df

def process_patient(path, record_id):
    df = pd.read_csv(path)
    
    first_val = df['Time'].dropna().iloc[0] if not df['Time'].dropna().empty else 0
    
    if isinstance(first_val, str) and ':' in first_val:
        df['Time'] = df['Time'].apply(lambda x: int(str(x).split(':')[0])*60 + int(str(x).split(':')[1]))
    
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    df = df[df['Time'] <= 1440]
    
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
    outcomes = get_labels(OUTCOME_PATH)
    ids = outcomes.index.tolist()
    
    X_list = []
    y_list = []
    
    print("Processing files...")
    
    for rid in ids:
        filepath = os.path.join(DATA_PATH, f"{rid}.txt")
        if not os.path.exists(filepath):
            continue
            
        try:
            patient_matrix = process_patient(filepath, rid)
            X_list.append(patient_matrix)
            y_list.append(outcomes.loc[rid, 'In-hospital_death'])
        except Exception:
            continue

    X = np.array(X_list)
    y = np.array(y_list)
    
    N, T, F = X.shape
    
    X_flat = X.reshape(N * T, F)
    
    col_means = np.nanmean(X_flat, axis=0)
    
    inds = np.where(np.isnan(X_flat))
    
    X_flat[inds] = np.take(col_means, inds[1])
    
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    
    X_final = X_flat_scaled.reshape(N, T, F)
    
    np.save('Processed_Data/X_train.npy', X_final)
    np.save('Processed_Data/y_train.npy', y)
    
    print("Done.")
    print(f"X shape: {X_final.shape}")
    print(f"y shape: {y.shape}")

if __name__ == "__main__":
    main()