# spam_email_detector

📌 Overview

The Spam Email detection system is a machine learning based web application designed to classify emails as Spam or not spam . The system analyzes the content of emails using natural language processing (NLP) techniques and trained machine learning models to provide accurate and real-time predictions.

This project is built using Python, Streamlit, and Scikit-learn, offering an interactive and user-friendly interface for detecting spam emails efficiently.

🚀 Features

🔍 Real-time email classification (spam / not spam)

📊 Confidence score display for predictions

🧠 Multiple ML models (naive bayes, logistic regression, SVM)

🧹 Text preprocessing (cleaning, tokenization, normalization)

📁 Batch prediction using CSV file upload

📈 Model performance visualization (Accuracy, Precision, Recall, F1-score)

📉 Confusion matrix visualization

💾 Download prediction results as CSV

🛠️ Technologies Used

Programming language: Python

Frontend framework: Streamlit

Libraries:

Pandas

NumPy

Scikit learn

Matplotlib

Seaborn

re, string (for text preprocessing)

spam-email-detector

│
├── app.py                # Main Streamlit application

├── requirements.txt      # Required libraries

├── README.md             # Project documentation

└── dataset/              # (Optional) dataset files

🧠 How It Works

User inputs email text or uploads a CSV file

Text is preprocessed (cleaning, removing noise)

TF-IDF vectorization converts text into numerical format

Selected ML model analyzes the input

System predicts whether the email is Spam or Ham

Results are displayed with confidence score and metrics

📊 Models Used

Multinomial naive bayes

Logistic regression

Linear support vector machine (SVM)

⚠️ Limitations

Small dataset may affect accuracy

Cannot detect highly sophisticated or evolving spam patterns

Works primarily on text-based email content

🔮 Future Improvements

Use larger and real-world datasets

Implement deep learning models (LSTM, BERT)

Add email header analysis

Deploy on cloud platforms (AWS, Heroku)

Improve UI/UX design

👨‍💻 Author

Siya Panwar

Branch: CSE AIML

VIT Bhopal
