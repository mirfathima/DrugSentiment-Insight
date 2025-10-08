#  DrugSentiment-Insight  
**Analyze patient drug reviews using NLP and machine learning to predict sentiments and recommend the most effective medications for each condition.**

---

## 📘 Overview
This project focuses on analyzing user drug reviews to recommend the most suitable medications for specific conditions based on **sentiment polarity** and **textual analysis**. Using Natural Language Processing (NLP) and machine learning techniques, it extracts meaningful insights from patient feedback and predicts user satisfaction levels.  
The goal is to bridge the gap between **drug efficacy and public sentiment**, helping both patients and healthcare professionals make better-informed decisions.

---

## 🧠 Key Features
- **Sentiment Analysis:** Determines whether a user's drug review is positive, negative, or neutral.  
- **Drug Recommendation System:** Suggests top drugs for a given medical condition based on sentiment scores and historical reviews.  
- **Text Preprocessing:** Includes tokenization, lemmatization, stopword removal, and vectorization (TF-IDF or CountVectorizer).  
- **Machine Learning Models:** Trains and evaluates models such as Logistic Regression, Random Forest, and Naive Bayes for sentiment classification.  
- **Data Visualization:** Visual representations of sentiment distribution, top drugs per condition, and word frequency.  

---

## 🗂️ Dataset
The dataset contains **drug reviews**, **ratings**, and **conditions**.  
Each entry includes:
- `drugName`: Name of the medication  
- `condition`: The disease or health condition  
- `review`: Text review provided by the user  
- `rating`: User rating (numerical)  
- `usefulCount`: Number of people who found the review useful  
- `date`: Review submission date  

*Dataset Source: [Kaggle Drug Review Dataset (Drugs.com Reviews)](https://www.kaggle.com/datasets/)*

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Tools:**
  - `pandas`, `numpy` – data manipulation  
  - `matplotlib`, `seaborn` – data visualization  
  - `scikit-learn` – machine learning  
  - `nltk`, `spacy` – natural language processing  
  - `wordcloud` – word frequency visualization  

---

## 🚀 Workflow
1. **Data Cleaning & Preprocessing**  
   - Handle missing values  
   - Clean text (remove punctuation, stopwords, etc.)  
2. **Exploratory Data Analysis (EDA)**  
   - Visualize review ratings and sentiment trends  
3. **Sentiment Classification**  
   - Train ML models to classify sentiments from text reviews  
4. **Drug Recommendation**  
   - Recommend top-rated drugs per condition based on sentiment and user feedback  
5. **Model Evaluation**  
   - Assess performance using accuracy, precision, recall, and F1-score  

---

## 📊 Results
- Achieved strong accuracy in predicting sentiment polarity.  
- Identified top-performing drugs for several common conditions based on aggregated user sentiment.  
- Demonstrated a clear correlation between rating values and predicted sentiment labels.  

---
