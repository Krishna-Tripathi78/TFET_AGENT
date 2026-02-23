import numpy as np
from pymoo.core.problem import ElementwiseProblem

class TFETProblem(ElementwiseProblem):
    """Advanced TFET optimization based on ML-MOO research"""
    
    def __init__(self):
        super().__init__(n_var=5, n_obj=3, n_constr=2,
                         xl=np.array([0.1, 0.1, 10e-9, 1e-9, 1e15]),
                         xu=np.array([1.5, 1.5, 100e-9, 5e-9, 1e19]))
    
    def _evaluate(self, x, out, *args, **kwargs):
        vg, vd, L, tox, doping = x
        
        # Research paper objectives: Natural Length & Vertical Electric Field
        natural_length = self._calculate_natural_length(L, tox, doping)
        vertical_efield = self._calculate_vertical_efield(vg, tox)
        ion_ioff_ratio = self._calculate_ion_ioff_ratio(vg, vd, L, tox, doping)
        
        constraint1 = 1e6 - ion_ioff_ratio
        constraint2 = 100e-9 - natural_length
        
        out["F"] = [natural_length, vertical_efield, -ion_ioff_ratio]
        out["G"] = [constraint1, constraint2]
    
    def _calculate_natural_length(self, L, tox, doping):
        """Natural length calculation based on TFET physics"""
        epsilon_si = 11.7 * 8.854e-12
        q = 1.602e-19
        return np.sqrt((epsilon_si * tox) / (q * doping)) * 1e9  # Convert to nm
    
    def _calculate_vertical_efield(self, vg, tox):
        """Vertical electric field in gate oxide"""
        return vg / tox  # V/m
    
    def _calculate_ion_ioff_ratio(self, vg, vd, L, tox, doping):
        """Ion/Ioff ratio calculation"""
        Ion = 1e-6 * np.exp(vg/0.1) * (vd/L) * np.exp(-tox/1e-9) * (doping/1e17)
        Ioff = 1e-12 * np.exp(-vg/0.2) * np.exp(-L/10e-9) / (doping/1e17)
        return Ion / Ioff