"""
Accuracy Enhancement Module for TFET Optimization
Adds cross-validation, ensemble methods, hyperparameter tuning, and data augmentation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class AccuracyEnhancer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.ensemble_model = None
        self.best_params = {}
        
    def augment_data(self, X, y, noise_level=0.02):
        """Add synthetic samples with controlled noise"""
        X_aug = []
        y_aug = []
        
        for i in range(len(X)):
            # Original sample
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            # Add 2 noisy variants
            for _ in range(2):
                noise = np.random.normal(0, noise_level, X[i].shape)
                X_noisy = X[i] + noise * X[i]
                X_aug.append(X_noisy)
                y_aug.append(y[i])
        
        return np.array(X_aug), np.array(y_aug)
    
    def create_ensemble_model(self):
        """Create ensemble of multiple models"""
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(), random_state=42)
        
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb),
            ('gp', gp)
        ])
        
        return ensemble
    
    def hyperparameter_tuning(self, X, y):
        """Optimize hyperparameters using GridSearch"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        return grid_search.best_estimator_
    
    def cross_validate_model(self, X, y, model=None, cv=5):
        """Perform k-fold cross-validation"""
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        
        return {
            'mean_r2': float(np.mean(scores)),
            'std_r2': float(np.std(scores)),
            'scores': scores.tolist()
        }
    
    def train_enhanced_model(self, X, y, use_augmentation=True, use_ensemble=True):
        """Train model with all enhancements"""
        # Data augmentation
        if use_augmentation:
            X, y = self.augment_data(X, y)
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Polynomial features
        X_poly = self.poly_features.fit_transform(X_scaled)
        
        # Model selection
        if use_ensemble:
            model = self.create_ensemble_model()
        else:
            model = self.hyperparameter_tuning(X_poly, y)
        
        # Train
        model.fit(X_poly, y)
        self.ensemble_model = model
        
        # Cross-validation
        cv_results = self.cross_validate_model(X_poly, y, model)
        
        # Calculate metrics
        y_pred = model.predict(X_poly)
        metrics = {
            'r2_score': float(r2_score(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'cv_results': cv_results,
            'best_params': self.best_params,
            'augmented_samples': len(X)
        }
        
        return model, metrics
    
    def predict(self, X):
        """Make predictions with trained model"""
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly_features.transform(X_scaled)
        return self.ensemble_model.predict(X_poly)

def enhance_csv_optimization(csv_path):
    """Enhanced optimization with accuracy improvements"""
    df = pd.read_csv(csv_path)
    
    # Prepare features and targets
    X = df[['gate_voltage', 'drain_voltage', 'channel_length', 'oxide_thickness']].values
    
    # Calculate objectives
    natural_length = df['channel_length'].values * 1e9
    vertical_efield = df['gate_voltage'].values / df['oxide_thickness'].values
    ion_ioff_ratio = (df['drain_voltage'].values / df['gate_voltage'].values) * 1e6
    
    enhancer = AccuracyEnhancer()
    
    # Train models for each objective
    results = {}
    for i, (target, name) in enumerate([(natural_length, 'natural_length'),
                                          (vertical_efield, 'vertical_efield'),
                                          (ion_ioff_ratio, 'ion_ioff_ratio')]):
        model, metrics = enhancer.train_enhanced_model(X, target)
        results[name] = metrics
    
    return results
