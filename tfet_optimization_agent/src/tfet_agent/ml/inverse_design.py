import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class InverseDesigner:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        """Train inverse design model: objectives -> parameters"""
        # Inverse mapping: y (objectives) -> X (parameters)
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Train model to predict parameters from objectives
        self.model.fit(y_scaled, X_scaled)
        self.is_trained = True
    
    def generate_designs(self, target_objectives):
        """Generate device designs for target objectives"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        target_scaled = self.scaler_y.transform(target_objectives)
        predicted_params = self.model.predict(target_scaled)
        designs = self.scaler_X.inverse_transform(predicted_params)
        
        return designs
    
    def optimize_for_target(self, target_performance):
        """Optimize design for specific target performance"""
        if not self.is_trained:
            return None
        
        # Generate multiple candidate designs
        candidates = []
        for _ in range(10):
            noise = np.random.normal(0, 0.1, len(target_performance))
            noisy_target = target_performance + noise
            design = self.generate_designs([noisy_target])
            candidates.append(design[0])
        
        return np.array(candidates)