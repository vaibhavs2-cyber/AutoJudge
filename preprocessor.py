import pandas as pd
import re
INPUT_FILE = 'problems_data.jsonl'
OUTPUT_FILE = 'processed_data.csv'

def clean_text(text):
    """
    I have made a simple function to clean text:
    1. It'll convert to lowercase
    2. and remove extra spaces
    """
    if not isinstance(text, str):
        return ""
    # I have converted to lowercase (actually to ensure "Graph" and "graph" are same)
    text = text.lower()
    # and here replaced multiple spaces or newlines with a single space.....
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_data():
    print("⏳ Loading data...")
    df = pd.read_json(INPUT_FILE, lines=True)
    print("🔄 Combining text columns...")
    df['text_feature'] = (
        df['title'] + " " + 
        df['description'] + " " + 
        df['input_description'] + " " + 
        df['output_description']
    )
    print("✨ Cleaning text...")
    df['text_feature'] = df['text_feature'].apply(clean_text)
    # Actually I have mapped strings to numbers for AI so as to create a 'target' column for the difficulty
    difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
    # So mapping only if the text matches exactly (I have actually handled the case sensitivity just in case)
    df['difficulty_level'] = df['problem_class'].str.lower().map(difficulty_map)
    # I have only kept the columns needed for training...
    final_df = df[['text_feature', 'difficulty_level', 'problem_score']]
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Processed data saved to '{OUTPUT_FILE}'")
    print(f"   -> Top 5 rows:\n{final_df.head()}")

if __name__ == "__main__":
    preprocess_data()