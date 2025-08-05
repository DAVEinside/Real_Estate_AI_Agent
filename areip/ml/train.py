"""ML model training pipeline for AREIP."""

import asyncio
import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

from ..config import settings
from ..utils.database import DatabaseManager
from ..data.ingestion import DataIngestionPipeline
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


class PropertyPricePredictor:
    """Property price prediction model."""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.metrics = {}
        
    def _get_model(self):
        """Get the appropriate model based on type."""
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
        
        return models.get(self.model_type, models['xgboost'])
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Feature engineering
        if 'bedrooms' in data.columns and 'bathrooms' in data.columns:
            data['bed_bath_ratio'] = data['bedrooms'] / (data['bathrooms'] + 0.1)
        
        if 'square_feet' in data.columns and 'lot_size' in data.columns:
            data['sqft_lot_ratio'] = data['square_feet'] / (data['lot_size'] + 1)
        
        if 'year_built' in data.columns:
            current_year = datetime.now().year
            data['property_age'] = current_year - data['year_built']
            data['age_squared'] = data['property_age'] ** 2
        
        if 'days_on_market' in data.columns:
            data['dom_log'] = np.log1p(data['days_on_market'])
        
        # Location-based features (if available)
        if 'city' in data.columns:
            city_stats = data.groupby('city')['price'].agg(['mean', 'median', 'std']).add_prefix('city_price_')
            data = data.merge(city_stats, left_on='city', right_index=True, how='left')
        
        if 'zip_code' in data.columns:
            zip_stats = data.groupby('zip_code')['price'].agg(['mean', 'count']).add_prefix('zip_')
            data = data.merge(zip_stats, left_on='zip_code', right_index=True, how='left')
        
        # Categorical encoding
        categorical_columns = ['property_type', 'city', 'state']
        for col in categorical_columns:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    # Handle unseen categories
                    classes = set(self.label_encoders[col].classes_)
                    data[col] = data[col].astype(str)
                    mask = data[col].isin(classes)
                    data.loc[~mask, col] = 'unknown'
                    
                    if 'unknown' not in classes:
                        # Add unknown category
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                    
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col])
        
        # Select numeric features
        numeric_features = [
            'bedrooms', 'bathrooms', 'square_feet', 'lot_size', 'property_age',
            'days_on_market', 'bed_bath_ratio', 'sqft_lot_ratio', 'age_squared',
            'dom_log', 'city_price_mean', 'city_price_median', 'zip_mean'
        ]
        
        # Add encoded categorical features
        for col in categorical_columns:
            if col in data.columns:
                numeric_features.append(f'{col}_encoded')
        
        # Keep only existing columns
        feature_columns = [col for col in numeric_features if col in data.columns]
        
        if not self.feature_columns:
            self.feature_columns = feature_columns
        
        # Select and fill missing values
        X = data[feature_columns].copy()
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns, index=X.index)
        
        return X
    
    def train(self, df: pd.DataFrame, target_column: str = 'price') -> Dict[str, Any]:
        """Train the price prediction model."""
        
        logger.info(f"Training {self.model_type} model on {len(df)} samples")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df[target_column].copy()
        
        # Remove outliers (prices beyond 3 standard deviations)
        price_mean = y.mean()
        price_std = y.std()
        outlier_mask = (y > price_mean - 3*price_std) & (y < price_mean + 3*price_std)
        
        X = X[outlier_mask]
        y = y[outlier_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.model = self._get_model()
        
        # Use scaled features for linear models, original for tree-based
        if self.model_type in ['linear', 'ridge', 'lasso']:
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Cross-validation
        if self.model_type in ['linear', 'ridge', 'lasso']:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            self.metrics['feature_importance'] = feature_importance
        
        logger.info(f"Model training completed. R² Score: {self.metrics['r2']:.4f}")
        logger.info(f"RMSE: ${self.metrics['rmse']:,.0f}")
        logger.info(f"MAE: ${self.metrics['mae']:,.0f}")
        
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_features(df)
        
        if self.model_type in ['linear', 'ridge', 'lasso']:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {filepath}")


class MarketTrendPredictor:
    """Market trend prediction model using time series analysis."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
    
    def prepare_time_series_features(self, df: pd.DataFrame, target_col: str, lookback: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series features for training."""
        
        # Sort by date
        df_sorted = df.sort_values('date')
        values = df_sorted[target_col].values
        
        X, y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i-lookback:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def train_market_model(self, df: pd.DataFrame, region: str, metric: str = 'median_price') -> Dict[str, Any]:
        """Train market trend prediction model for a specific region."""
        
        # Filter data for the region
        region_data = df[df['RegionName'] == region].copy()
        
        if len(region_data) < 24:  # Need at least 2 years of data
            logger.warning(f"Insufficient data for {region}. Need at least 24 months.")
            return {}
        
        # Prepare features
        X, y = self.prepare_time_series_features(region_data, metric)
        
        if len(X) == 0:
            return {}
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train XGBoost model
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Store model and metrics
        model_key = f"{region}_{metric}"
        self.models[model_key] = model
        self.metrics[model_key] = metrics
        
        logger.info(f"Market model trained for {region} - {metric}. R² Score: {metrics['r2']:.4f}")
        
        return metrics
    
    def predict_market_trend(self, df: pd.DataFrame, region: str, metric: str = 'median_price', periods: int = 6) -> np.ndarray:
        """Predict future market trends."""
        
        model_key = f"{region}_{metric}"
        if model_key not in self.models:
            raise ValueError(f"No trained model found for {region} - {metric}")
        
        model = self.models[model_key]
        
        # Get latest 12 months of data
        region_data = df[df['RegionName'] == region].sort_values('date')
        latest_values = region_data[metric].tail(12).values
        
        predictions = []
        current_window = latest_values.copy()
        
        for _ in range(periods):
            pred = model.predict([current_window])[0]
            predictions.append(pred)
            
            # Update window for next prediction
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred
        
        return np.array(predictions)


class ModelTrainingPipeline:
    """Complete model training pipeline."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.models = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    async def fetch_training_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for training."""
        
        logger.info("Fetching training data from database")
        
        # Fetch property listings
        property_query = """
        SELECT * FROM raw_data.property_listings 
        WHERE price IS NOT NULL 
        AND price > 50000 
        AND price < 50000000
        AND square_feet IS NOT NULL
        AND square_feet > 200
        ORDER BY listing_date DESC
        LIMIT 10000
        """
        
        property_data = await self.db_manager.fetch_query(property_query)
        
        # Fetch market trends
        market_query = """
        SELECT * FROM raw_data.market_trends
        WHERE date >= CURRENT_DATE - INTERVAL '5 years'
        ORDER BY region_code, date
        """
        
        market_data = await self.db_manager.fetch_query(market_query)
        
        return {
            'properties': property_data,
            'market_trends': market_data
        }
    
    async def train_price_prediction_models(self, property_data: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple price prediction models."""
        
        results = {}
        model_types = ['xgboost', 'lightgbm', 'random_forest']
        
        for model_type in model_types:
            logger.info(f"Training {model_type} price prediction model")
            
            try:
                predictor = PropertyPricePredictor(model_type=model_type)
                metrics = predictor.train(property_data)
                
                # Save the model
                model_path = self.model_dir / f"price_predictor_{model_type}.pkl"
                predictor.save_model(str(model_path))
                
                results[model_type] = {
                    'metrics': metrics,
                    'model_path': str(model_path)
                }
                
                # Store in database
                await self._save_model_to_db(
                    model_name=f"price_predictor_{model_type}",
                    model_version="1.0.0",
                    model_type="regression",
                    model_path=str(model_path),
                    metrics=metrics
                )
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    async def train_market_trend_models(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Train market trend prediction models."""
        
        predictor = MarketTrendPredictor()
        results = {}
        
        # Get unique regions
        regions = market_data['region_name'].unique()[:10]  # Limit to top 10 regions
        
        for region in regions:
            try:
                metrics = predictor.train_market_model(market_data, region)
                if metrics:
                    results[region] = metrics
            except Exception as e:
                logger.error(f"Error training market model for {region}: {e}")
                results[region] = {'error': str(e)}
        
        # Save the market predictor
        model_path = self.model_dir / "market_trend_predictor.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)
        
        results['model_path'] = str(model_path)
        
        return results
    
    async def _save_model_to_db(
        self, 
        model_name: str, 
        model_version: str, 
        model_type: str,
        model_path: str,
        metrics: Dict[str, Any]
    ):
        """Save model metadata to database."""
        
        query = """
        INSERT INTO ml_models.model_registry 
        (model_name, model_version, model_type, model_path, training_metrics, is_active, created_by)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (model_name, model_version) 
        DO UPDATE SET 
            training_metrics = EXCLUDED.training_metrics,
            is_active = EXCLUDED.is_active
        """
        
        await self.db_manager.execute_query(
            query,
            model_name, model_version, model_type, model_path,
            json.dumps(metrics), True, "training_pipeline"
        )
    
    async def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete model training pipeline."""
        
        logger.info("Starting full model training pipeline")
        start_time = datetime.now()
        
        try:
            # Fetch training data
            training_data = await self.fetch_training_data()
            
            results = {
                'start_time': start_time.isoformat(),
                'data_summary': {
                    'properties_count': len(training_data['properties']),
                    'market_trends_count': len(training_data['market_trends'])
                }
            }
            
            # Train price prediction models
            if not training_data['properties'].empty:
                price_results = await self.train_price_prediction_models(training_data['properties'])
                results['price_models'] = price_results
            
            # Train market trend models
            if not training_data['market_trends'].empty:
                market_results = await self.train_market_trend_models(training_data['market_trends'])
                results['market_models'] = market_results
            
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration_minutes'] = (end_time - start_time).total_seconds() / 60
            
            logger.info(f"Model training pipeline completed in {results['duration_minutes']:.2f} minutes")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


async def main():
    """Main function for running training pipeline."""
    
    setup_logging()
    logger.info("Starting AREIP model training")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    await db_manager.initialize_database()
    
    # Run training pipeline
    pipeline = ModelTrainingPipeline(db_manager)
    results = await pipeline.run_full_training_pipeline()
    
    print("\n" + "="*50)
    print("AREIP MODEL TRAINING RESULTS")
    print("="*50)
    print(f"Training Duration: {results.get('duration_minutes', 0):.2f} minutes")
    print(f"Properties Processed: {results.get('data_summary', {}).get('properties_count', 0):,}")
    print(f"Market Trends Processed: {results.get('data_summary', {}).get('market_trends_count', 0):,}")
    
    if 'price_models' in results:
        print("\nPrice Prediction Models:")
        for model_name, model_results in results['price_models'].items():
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                print(f"  {model_name}: R² = {metrics.get('r2', 0):.4f}, RMSE = ${metrics.get('rmse', 0):,.0f}")
    
    if 'market_models' in results:
        print(f"\nMarket Trend Models: {len(results['market_models'])} regions trained")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())