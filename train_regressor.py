import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

INPUT_FILE = 'processed_data.csv'
MODEL_FILE = 'score_model.pkl'

# Using the SAME vectorizer file name to reuse the vocabulary from the previous step.
# I have tried to keep it simple so we can just create a new specific one for regressor (to basically avoid conflicts)......
VECTORIZER_FILE = 'tfidf_vectorizer_reg.pkl'

def train_regressor():
    print("⏳ Loading data...")
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=['text_feature'], inplace=True)
    X = df['text_feature']
    y = df['problem_score']  # My target is not actually the class but the NUMBER now.
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("🧮 Vectorizing text...") #Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("🌲 Training Random Forest (this might take 30-60 seconds)...") # Now trained the regressor.....
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  #Here n_estimators=100 means using 100 small decision trees to vote.
    model.fit(X_train_vec, y_train)
    print("📝 Testing prediction accuracy...") #evaluating.....
    y_pred = model.predict(X_test_vec)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n📉 Mean Absolute Error: {mae:.2f}")
    print(f"   (On average, our prediction is off by {mae:.2f} points)")
    print("💾 Saving score model...") # Finally I have saved.....
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("✅ Score Model Saved!")

if __name__ == "__main__":
    train_regressor()