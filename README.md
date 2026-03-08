# 📰 Fake News Detection Using NLP and Machine Learning

## Introduction
The rapid spread of online news has increased the circulation of misinformation, making it difficult for readers and organizations to verify content quickly. Traditional manual fact-checking processes are time-consuming and allow fake news to spread before validation. This project uses Natural Language Processing (NLP) to automatically classify news articles as fake or real, enabling faster verification, improving trust in credible sources, and reducing the impact of false information.

---

## Problem Statement
Develop an automated NLP-based classification system that identifies whether a news article is fake or real within seconds, reducing manual review time from multiple days to near real-time, while achieving an F1-score of at least 85% on a labeled news dataset.

---

## Data Wrangling
The dataset was first created by combining True and Fake news sources, with a new column `status` added to label:

- **True news = 1**
- **Fake news = 0**

Initially, the dataset contained **44,898 rows and 5 columns**.

### Data Preparation
- The **title** and **text** columns were combined into a single **content** column.
- The **date column was cleaned**.
- Text was converted to **lowercase**.
- **Punctuation, special characters, and stopwords were removed**.
- Text was **tokenized** into individual words.
- **WordNet lemmatization** was applied to reduce words to their base forms.

This produced:
- `lemmatized_tokens`
- `lemmatized_content`

### Feature Engineering
A numeric feature **content_length** was created to capture article length.

To analyze whether fake news articles use more aggressive language, toxicity scores were generated using the **Detoxify model**.

The following toxicity features were extracted:

- toxicity
- severe_toxicity
- obscene
- threat
- insult
- identity_attack

After cleaning and feature engineering, the dataset contained:

**43,556 rows and 16 columns**

The dataset was now ready for **exploratory data analysis and machine learning modeling**.

---

## Exploratory Data Analysis (EDA)

### Class Distribution
The dataset is fairly balanced:

- **Fake News ≈ 52%**
- **True News ≈ 48%**

### Observations
- Shorter articles were slightly more likely to be fake, though not strongly predictive.
- Fake news often contained **sensational or politically charged language**.
- True news used **formal reporting terminology**.

### Toxicity Analysis
Using Detoxify revealed that fake news exhibits higher levels of:

- toxicity
- insult
- obscene language
- identity attacks

These insights helped guide **feature engineering and model development**.

---

## Model Preprocessing and Feature Engineering

Additional structural features were created:

- **Average Word Length**
- **Unique Word Ratio**

The **subject category feature was removed** to prevent bias.

### Data Splitting
The dataset was split into **stratified training and testing sets** to maintain class balance.

### Feature Transformation
- Numeric features were **standardized**.
- Text data was converted into **Word2Vec embeddings**.

The Word2Vec model was trained **only on the training dataset** to prevent **data leakage**.

The resulting features formed the final dataset used for modeling.

---

## Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost

### Evaluation Metrics
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Model Performance
Logistic Regression and Linear SVM achieved the strongest results:

- **F1-score ≈ 0.98**
- **ROC-AUC > 0.99**

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV** on the top-performing models.

After tuning, **Logistic Regression** was selected as the final model due to:

- Comparable predictive performance
- Faster training time
- Probabilistic outputs
- Better scalability
- High interpretability

---

## Winning Model

Both Logistic Regression and Linear SVM produced strong results.

Although Linear SVM achieved slightly higher recall, **Logistic Regression** was chosen as the final model because it provides:

- Faster training
- Probability predictions
- Easier interpretability
- Better scalability

### Feature Importance Insights

Features associated with **True News**:

- content_length
- unique_word_ratio
- obscene
- threat
- insult
- identity_attack

Features associated with **Fake News**:

- avg_word_length
- toxicity
- severe_toxicity

---

## Recommendations

The fake news classification system demonstrates strong predictive performance and can support:

- Journalists
- Fact-checkers
- Social media platforms

Platforms could integrate this model into **content moderation systems** to flag suspicious articles for further verification.

Combining linguistic features with additional signals such as:

- source credibility
- information propagation patterns

may further improve detection accuracy.

---

## Limitations

Despite strong performance, the model has several limitations:

- The dataset may not fully represent the diversity of real-world news sources.
- Toxicity analysis was performed only on the **first portion of each article** to reduce computational cost.
- Word2Vec embeddings capture semantic relationships but lack deeper contextual understanding compared to **transformer-based models**.

---

## Conclusion

This project successfully developed a machine learning system for detecting fake news using Natural Language Processing.

By combining:

- text embeddings
- linguistic features
- toxicity analysis

multiple models were evaluated, with **Logistic Regression emerging as the best-performing model**.

The final model achieved:

- **F1-score ≈ 0.98**
- **ROC-AUC > 0.99**

These results demonstrate the effectiveness of combining traditional NLP techniques with engineered features for fake news detection.

---

## Future Work

Future improvements could include:

- Using **transformer-based models such as BERT**.
- Incorporating **publisher credibility and social media features**.
- Expanding datasets to improve generalization.
- Deploying the model as a **real-time API or web application**.

---

## Tech Stack

- Python
- Scikit-learn
- Gensim (Word2Vec)
- Detoxify
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost

---

## Model Performance

| Metric | Score |
|------|------|
| Accuracy | ~0.98 |
| Precision | ~0.98 |
| Recall | ~0.98 |
| F1 Score | ~0.98 |
| ROC-AUC | >0.99 |
