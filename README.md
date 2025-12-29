# 🧩 Vaibhav's Algo-Meter (AutoJudge Project)

## 📌 Here's the overview of my Project.....
**AutoJudge** is an ML project I have built to see if a computer can predict the difficulty of a programming problem just by reading its description. 

As a BSMS Physics student exploring Data Science, I was eager to understand how Natural Language Processing (NLP) applies to real-world tasks. This system reads a problem statement and uses **TF-IDF vectorization** combined with **Logistic Regression** and **Random Forest** to:
1.  Firstly, classify the problem as **Easy, Medium, or Hard**.
2.  And then predicting a numerical **Complexity Score (1-10)**.

## 🛠️ My Tech Stack
I used **Python 3.12** as the core language.
* **For Data Handling:** Pandas & NumPy (processed 4,112 problems).
* **For Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest, TF-IDF).
* **And for Visualization:** Streamlit (built the interactive web interface).

## 📊 How It Performs
I trained the models on a dataset of 4,112 competitive programming problems.
* **Classification Accuracy:** ~51%.
    * *My Analysis:* While 51% sounds low, it is actually a solid baseline for a 3-class text problem (random guessing would be 33%).
* **Regression Error (MAE):** ~1.70.
    * *Meaning:* On a scale of 1-10, the AI's difficulty guess is usually within 1.7 points of the real score.
* **Additional Evaluation:** A confusion matrix and classification report were used to analyze class-wise performance and understand bias toward harder problems.

## 📂 The Project Structure
Here is a breakdown of the code I wrote:

* `app.py`: The main file that runs the **Streamlit** website.
* `train_classifier.py`: The script I used to train the "Easy/Medium/Hard" logic.
* `train_regressor.py`: The script that learns to predict the numerical score.
* `preprocessor.py`: My code for cleaning the text (removing spaces, converting to lowercase, etc....).
* `data_analysis.py`: A utility script I wrote to inspect the raw JSON data.
* `processed_data.csv`: The final cleaned dataset used for training.

## 🚀 How to Run My Code
1.  **Clone this repo:**
    ```bash
    git clone <your-repo-link>
    cd AutoJudge_Project
    ```
2.  **Install the requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```

## ⚠️ Challenges & Future Work
During this project, I discovered a significant **Class Imbalance** in the dataset. There are way more "Hard" problems (approx 1,900) than "Easy" ones (approx 700).
* **The Result:** The model is a bit biased and tends to guess "Hard" when it is unsure.
* **Future Plan:** If I continue this project, I would implement **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the data or try deep learning models like **BERT**.

---
*Built by **Vaibhav Sharma** (BSMS Physics) for **ACM IITR Open Projects**.*