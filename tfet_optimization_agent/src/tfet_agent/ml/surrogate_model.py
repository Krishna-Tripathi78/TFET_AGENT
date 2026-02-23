import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

class SurrogateModel:
    def __init__(self, model_type='gp'):
        self.model_type = model_type
        self.models = {}
        self.is_trained = False
        
    def build_models(self, X, y):
        n_objectives = y.shape[1] if len(y.shape) > 1 else 1
        
        for i in range(n_objectives):
            if self.model_type == 'gp':
                kernel = Matern(length_scale=1.0, nu=2.5)
                model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            y_obj = y[:, i] if len(y.shape) > 1 else y
            model.fit(X, y_obj)
            self.models[f'obj_{i}'] = model
            
        self.is_trained = True
        
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Models not trained")
            
        predictions = []
        for i in range(len(self.models)):
            pred = self.models[f'obj_{i}'].predict(X)
            predictions.append(pred)
            
        return np.column_stack(predictions)

class ActiveLearning:
    def __init__(self, surrogate_model):
        self.surrogate = surrogate_model
        
    def select_next_samples(self, candidate_X, n_samples=5):
        return np.random.choice(len(candidate_X), n_samples, replace=False)