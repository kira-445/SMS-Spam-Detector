
# SMS Spam Detector

A Machine Learning project that classifies SMS messages as Spam or Ham (Not Spam).

## Features

- Classifies SMS messages as Spam or Ham.
- Uses TF-IDF for text feature extraction.
- Uses Multinomial Naive Bayes for classification.
- Displays prediction results through a Streamlit web application.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Streamlit
- Joblib

## Dataset

UCI SMS Spam Collection Dataset.

Dataset link:
https://archive.ics.uci.edu/dataset/228/sms

## Project Structure

SMS-Spam-Detector/
- data/
- train.py
- app.py
- requirements.txt
- README.md
- .gitignore
- spam_model.joblib

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Train the machine learning model:

   python train.py

3. Run the Streamlit application:

   python -m streamlit run app.py

## Machine Learning Algorithm

- TF-IDF Vectorization
- Multinomial Naive Bayes

## Author

sahwaz khan
