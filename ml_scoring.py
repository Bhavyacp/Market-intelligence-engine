import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_market_impact_model(csv_path: str, model_save_path: str):
    print("Training XGBoost Market Impact Prediction Model...")
    df = pd.read_csv(csv_path)
    
    X = df[['sentiment_score', 'mention_frequency', 'competitor_activity_index']]
    y = df['high_impact_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("\n--- ML Model Evaluation ---")
    print(classification_report(y_test, preds))
    
    model.save_model(model_save_path)
    print(f"XGBoost model saved to {model_save_path}")

if __name__ == "__main__":
    train_market_impact_model("data/market_metadata.csv", "data/market_impact_xgb.json")