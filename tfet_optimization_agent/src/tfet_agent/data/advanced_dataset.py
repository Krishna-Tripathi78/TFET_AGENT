import numpy as np
import pandas as pd

class AdvancedTFETDataset:
    def __init__(self):
        self.materials = {
            'Si': {'bandgap': 1.12, 'effective_mass': 0.26, 'barrier_height': 0.3},
            'Ge': {'bandgap': 0.67, 'effective_mass': 0.12, 'barrier_height': 0.25},
            'InAs': {'bandgap': 0.35, 'effective_mass': 0.023, 'barrier_height': 0.15},
            'GaSb': {'bandgap': 0.72, 'effective_mass': 0.039, 'barrier_height': 0.2},
            'MoS2': {'bandgap': 1.8, 'effective_mass': 0.35, 'barrier_height': 0.4},
            'WSe2': {'bandgap': 1.6, 'effective_mass': 0.27, 'barrier_height': 0.35}
        }
    
    def generate_hybrid_dataset(self, n_samples=1000):
        """Generate advanced dataset with material properties and defects"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Random material selection
            material = np.random.choice(list(self.materials.keys()))
            mat_props = self.materials[material]
            
            # Basic TFET parameters
            gate_voltage = np.random.uniform(0.1, 1.5)
            drain_voltage = np.random.uniform(0.1, 1.5)
            channel_length = np.random.uniform(5e-9, 100e-9)
            oxide_thickness = np.random.uniform(0.5e-9, 5e-9)
            doping = np.random.uniform(1e15, 1e19)
            
            # Advanced parameters
            defect_density = np.random.exponential(1e11)
            barrier_height = mat_props['barrier_height'] * (1 + np.random.normal(0, 0.1))
            effective_mass = mat_props['effective_mass'] * (1 + np.random.normal(0, 0.05))
            
            # Calculate objectives
            natural_length = self._calculate_natural_length(channel_length, oxide_thickness, doping)
            vertical_efield = gate_voltage / oxide_thickness
            ion_ioff_ratio = self._calculate_ion_ioff_ratio(gate_voltage, drain_voltage, 
                                                          channel_length, oxide_thickness, 
                                                          doping, defect_density)
            
            data.append({
                'gate_voltage': gate_voltage,
                'drain_voltage': drain_voltage,
                'channel_length': channel_length,
                'oxide_thickness': oxide_thickness,
                'doping': doping,
                'defect_density': defect_density,
                'barrier_height': barrier_height,
                'effective_mass': effective_mass,
                'material': material,
                'natural_length': natural_length,
                'vertical_efield': vertical_efield,
                'ion_ioff_ratio': ion_ioff_ratio
            })
        
        return pd.DataFrame(data)
    
    def _calculate_natural_length(self, L, tox, doping):
        """Calculate natural length with material effects"""
        epsilon_si = 11.7 * 8.854e-12
        q = 1.602e-19
        return np.sqrt((epsilon_si * tox) / (q * doping)) * 1e9
    
    def _calculate_ion_ioff_ratio(self, vg, vd, L, tox, doping, defect_density):
        """Calculate Ion/Ioff ratio with defect effects"""
        Ion = 1e-6 * np.exp(vg/0.1) * (vd/L) * np.exp(-tox/1e-9) * (doping/1e17)
        Ioff = 1e-12 * np.exp(-vg/0.2) * np.exp(-L/10e-9) / (doping/1e17) * (1 + defect_density/1e12)
        return Ion / Ioff