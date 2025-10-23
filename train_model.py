"""
CO2 Emission Prediction - Model Training Script
This script trains the ML model and saves it with joblib
Run this file to retrain the model with latest data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("CO2 EMISSION PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset from Our World in Data...")
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    try:
        df = pd.read_csv(url)
        print(f"   ✓ Dataset loaded successfully!")
        print(f"   ✓ Shape: {df.shape}")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    features = ['year', 'population', 'gdp', 'coal_co2', 'oil_co2', 
                'gas_co2', 'cement_co2', 'energy_per_capita']
    target = 'co2'
    
    df_clean = df[['country', 'year'] + features + [target]].dropna()
    print(f"   ✓ Cleaned dataset shape: {df_clean.shape}")
    print(f"   ✓ Countries: {df_clean['country'].nunique()}")
    print(f"   ✓ Years: {df_clean['year'].min()} - {df_clean['year'].max()}")
    
    # Prepare features and target
    X = df_clean[features]
    y = df_clean[target]
    
    # Split data
    print("\n3. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Training samples: {X_train.shape[0]}")
    print(f"   ✓ Testing samples: {X_test.shape[0]}")
    
    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features scaled using StandardScaler")
    
    # Train model
    print("\n5. Training Random Forest model...")
    print("   (This may take 1-2 minutes...)")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train_scaled, y_train)
    print("   ✓ Model trained successfully!")
    
    # Evaluate model
    print("\n6. Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   ✓ R² Score: {r2:.4f}")
    print(f"   ✓ RMSE: {rmse:.2f}")
    print(f"   ✓ MAE: {mae:.2f}")
    
    # Feature importance
    print("\n7. Feature importance:")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"   • {row['feature']:<20}: {row['importance']:.4f}")
    
    # Save model and scaler
    print("\n8. Saving model and scaler...")
    try:
        joblib.dump(model, 'co2_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(df_clean, 'cleaned_data.joblib')
        
        print("   ✓ Model saved as: co2_model.joblib")
        print("   ✓ Scaler saved as: scaler.joblib")
        print("   ✓ Data saved as: cleaned_data.joblib")
    except Exception as e:
        print(f"   ✗ Error saving files: {e}")
        return
    
    # Test loading
    print("\n9. Testing model loading...")
    try:
        loaded_model = joblib.load('co2_model.joblib')
        loaded_scaler = joblib.load('scaler.joblib')
        print("   ✓ Model and scaler loaded successfully!")
        
        # Make test prediction
        test_sample = X_test.iloc[0:1]
        test_scaled = loaded_scaler.transform(test_sample)
        prediction = loaded_model.predict(test_scaled)[0]
        actual = y_test.iloc[0]
        
        print(f"\n   Test Prediction:")
        print(f"   • Predicted: {prediction:.2f} MT")
        print(f"   • Actual: {actual:.2f} MT")
        print(f"   • Difference: {abs(prediction - actual):.2f} MT")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\n✅ All files created successfully!")
    print("\nNext steps:")
    print("1. Test the model: python test_model.py")
    print("2. Run Streamlit app: streamlit run app.py")
    print("3. Push to GitHub and deploy on Streamlit Cloud")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()