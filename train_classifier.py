import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # here I aim to save the trained model for later....
INPUT_FILE = 'processed_data.csv'
MODEL_FILE = 'difficulty_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
def train_model():
    print("⏳ Loading data...") 
    df = pd.read_csv(INPUT_FILE) # here handled any NaN values (just in case)
    df.dropna(subset=['text_feature'], inplace=True)
    X = df['text_feature'] # Input
    y = df['difficulty_level'] # Target

    # here I have splited into training (80%) and testing (20%) sets....
    print("✂️  Splitting data...") #(actually my approach is to hide 20% of the data so we can quiz model later)....
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("🧮 Vectorizing text (TF-IDF)...") # Then here converting Text to Numbers.....
    vectorizer = TfidfVectorizer(max_features=5000) # Looking at the top 5000 most important words so, max_features=5000.
    
    # Here "Fit" would learn the vocabulary from training data ....
    X_train_vec = vectorizer.fit_transform(X_train) 
    X_test_vec = vectorizer.transform(X_test) # Converting the text to number matrices......
    print("🧠 Training the Brain (Logistic Regression)...") # Finally training the model.
    model = LogisticRegression(max_iter=1000) # max_iter is providing more time to find the answer....
    model.fit(X_train_vec, y_train)
    print("📝 Testing the model...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Model Accuracy: {acc*100:.2f}%")
    print("\n--- Detailed Report ---")
    print(classification_report(y_test, y_pred, target_names=['Easy', 'Medium', 'Hard']))
    print("\n🧩 ---- Confusion Matrix ----")
    print(confusion_matrix(y_test, y_pred))

    print("💾 Saving model for the Web App...") # I did save BOTH the model and vectorizer.
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("✅ Done! Model saved.")

if __name__ == "__main__":
    train_model()