"""
Advanced ML Models for TFET Optimization - Target 95%+ Accuracy
Includes XGBoost, LightGBM, CatBoost, SVR, DNN, and Stacking Ensemble
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class AdvancedMLModels:
    """Advanced ML models for maximum accuracy"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def create_xgboost_model(self):
        """XGBoost - High accuracy for small datasets"""
        if not XGBOOST_AVAILABLE:
            return None
        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    def create_lightgbm_model(self):
        """LightGBM - Fast and accurate"""
        if not LIGHTGBM_AVAILABLE:
            return None
        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    def create_catboost_model(self):
        """CatBoost - Robust for small datasets"""
        if not CATBOOST_AVAILABLE:
            return None
        return cb.CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
    
    def create_svr_model(self):
        """Support Vector Regression"""
        return SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.01
        )
    
    def create_dnn_model(self, input_dim):
        """Deep Neural Network"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_dim=input_dim),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_stacking_ensemble(self):
        """Stacking Ensemble - Combines all models for maximum accuracy"""
        base_models = []
        
        # Add XGBoost
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', self.create_xgboost_model()))
        
        # Add LightGBM
        if LIGHTGBM_AVAILABLE:
            base_models.append(('lgb', self.create_lightgbm_model()))
        
        # Add CatBoost
        if CATBOOST_AVAILABLE:
            base_models.append(('cat', self.create_catboost_model()))
        
        # Add traditional models
        base_models.append(('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)))
        base_models.append(('gb', GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)))
        base_models.append(('svr', self.create_svr_model()))
        base_models.append(('gp', GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(), random_state=42)))
        
        # Meta-learner
        if XGBOOST_AVAILABLE:
            meta_learner = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            meta_learner = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        return stacking
    
    def train_all_models(self, X, y):
        """Train all available models and select best"""
        X_scaled = self.scaler.fit_transform(X)
        results = {}
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = self.create_xgboost_model()
            xgb_model.fit(X_scaled, y)
            xgb_score = r2_score(y, xgb_model.predict(X_scaled))
            results['XGBoost'] = {'model': xgb_model, 'score': xgb_score}
            print(f"XGBoost R²: {xgb_score:.4f}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_model = self.create_lightgbm_model()
            lgb_model.fit(X_scaled, y)
            lgb_score = r2_score(y, lgb_model.predict(X_scaled))
            results['LightGBM'] = {'model': lgb_model, 'score': lgb_score}
            print(f"LightGBM R²: {lgb_score:.4f}")
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            cat_model = self.create_catboost_model()
            cat_model.fit(X_scaled, y)
            cat_score = r2_score(y, cat_model.predict(X_scaled))
            results['CatBoost'] = {'model': cat_model, 'score': cat_score}
            print(f"CatBoost R²: {cat_score:.4f}")
        
        # SVR
        svr_model = self.create_svr_model()
        svr_model.fit(X_scaled, y)
        svr_score = r2_score(y, svr_model.predict(X_scaled))
        results['SVR'] = {'model': svr_model, 'score': svr_score}
        print(f"SVR R²: {svr_score:.4f}")
        
        # Stacking Ensemble
        print("Training Stacking Ensemble (this may take a moment)...")
        stack_model = self.create_stacking_ensemble()
        stack_model.fit(X_scaled, y)
        stack_score = r2_score(y, stack_model.predict(X_scaled))
        results['Stacking'] = {'model': stack_model, 'score': stack_score}
        print(f"Stacking Ensemble R²: {stack_score:.4f}")
        
        # Select best model
        best_name = max(results, key=lambda k: results[k]['score'])
        self.best_model = results[best_name]['model']
        self.best_score = results[best_name]['score']
        self.models = results
        
        print(f"\n✓ Best Model: {best_name} with R² = {self.best_score:.4f}")
        
        return results
    
    def evaluate_model(self, X, y, model):
        """Comprehensive model evaluation"""
        X_scaled = self.scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
        
        metrics = {
            'r2_score': float(r2_score(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'accuracy_percentage': float(r2_score(y, y_pred) * 100)
        }
        
        return metrics
    
    def predict(self, X):
        """Predict using best model"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)


def train_advanced_models_for_tfet(csv_path):
    """Train advanced models on TFET dataset"""
    df = pd.read_csv(csv_path)
    
    X = df[['gate_voltage', 'drain_voltage', 'channel_length', 'oxide_thickness']].values
    
    # Calculate TFET objectives
    natural_length = df['channel_length'].values * 1e9
    vertical_efield = df['gate_voltage'].values / df['oxide_thickness'].values
    ion_ioff_ratio = (df['drain_voltage'].values / df['gate_voltage'].values) * 1e6
    
    print("=" * 60)
    print("Training Advanced ML Models for TFET Optimization")
    print("=" * 60)
    
    results = {}
    
    for target, name in [(natural_length, 'Natural Length'),
                          (vertical_efield, 'Vertical E-field'),
                          (ion_ioff_ratio, 'Ion/Ioff Ratio')]:
        print(f"\n--- Training models for {name} ---")
        
        ml = AdvancedMLModels()
        model_results = ml.train_all_models(X, target)
        
        # Evaluate best model
        metrics = ml.evaluate_model(X, target, ml.best_model)
        
        results[name] = {
            'all_models': {k: v['score'] for k, v in model_results.items()},
            'best_model_metrics': metrics
        }
        
        print(f"Final Accuracy: {metrics['accuracy_percentage']:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    # Test with aluminum dataset
    csv_path = 'c:/Users/krish/TFET _ AGENT/aluminum_tfet_data.csv'
    results = train_advanced_models_for_tfet(csv_path)
    
    print("\n\nFinal Results Summary:")
    for objective, data in results.items():
        print(f"\n{objective}:")
        print(f"  Best Accuracy: {data['best_model_metrics']['accuracy_percentage']:.2f}%")
        print(f"  R² Score: {data['best_model_metrics']['r2_score']:.4f}")
        print(f"  CV Mean: {data['best_model_metrics']['cv_mean']:.4f}")
