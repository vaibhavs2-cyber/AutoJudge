import pandas as pd
import os
FILE_NAME = 'problems_data.jsonl' 
def load_and_inspect():
    print(f"📂 Looking for {FILE_NAME}...")
    if not os.path.exists(FILE_NAME):
        print("❌ Error: File not found!")
        print(f"   -> Make sure you downloaded '{FILE_NAME}' and put it in this folder.")
        return
    try: 
        df = pd.read_json(FILE_NAME, lines=True)
        print(f"✅ Successfully loaded {len(df)} problems!")
        print("\n--- 🔍 Raw Columns ---") 
        print(df.columns.tolist())
        print("\n--- ⚠️ Missing Values ---") 
        print(df.isnull().sum())
        print("\n--- 🎯 Difficulty Distribution ---")
        if 'problem_class' in df.columns:
            print(df['problem_class'].value_counts())
        elif 'task_class' in df.columns: 
            print(df['task_class'].value_counts())
        else:
            print("Could not find difficulty column. Check column names above.") 
    except ValueError as e:
        print(f"❌ Error reading JSON: {e}")
if __name__ == "__main__":
    load_and_inspect()