import numpy as np
import pandas as pd

class TFETDataHandler:
    def __init__(self):
        self.parameters = ['gate_voltage', 'drain_voltage', 'channel_length', 'oxide_thickness']
        self.metrics = ['on_current', 'off_current', 'subthreshold_swing']
    
    def generate_synthetic_dataset(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'gate_voltage': np.random.uniform(0.1, 1.5, n_samples),
            'drain_voltage': np.random.uniform(0.1, 1.5, n_samples),
            'channel_length': np.random.uniform(10e-9, 100e-9, n_samples),
            'oxide_thickness': np.random.uniform(1e-9, 5e-9, n_samples)
        }
        for i in range(n_samples):
            vg, vd, L, tox = data['gate_voltage'][i], data['drain_voltage'][i], data['channel_length'][i], data['oxide_thickness'][i]
            noise = np.random.normal(0, 0.1)
            data.setdefault('on_current', []).append(1e-6 * np.exp(vg/0.1) * (vd/L) * np.exp(-tox/1e-9) * (1 + noise))
            data.setdefault('off_current', []).append(1e-12 * np.exp(-vg/0.2) * np.exp(-L/10e-9) * (1 + noise))
            data.setdefault('subthreshold_swing', []).append(60 + 20 * np.random.random())
        
        return pd.DataFrame(data)
    
    def load_csv_data(self, filepath):
        df = pd.read_csv(filepath)
        required_cols = ['gate_voltage', 'drain_voltage', 'channel_length', 'oxide_thickness']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        return df
    
    def load_excel_data(self, filepath):
        return pd.read_excel(filepath)