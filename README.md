Twitter Sentiment Analysis

This repository contains a project that performs sentiment analysis on Twitter data. The goal is to classify tweets into positive, negative, or neutral sentiments using machine learning models. The project is implemented in Python and uses libraries such as scikit-learn, pandas, and matplotlib.

Table of Contents

Overview

Features

Technologies Used

Setup and Installation

Usage

Dataset

Results

Contributing

License

Overview

Sentiment analysis is a common task in natural language processing (NLP) that involves analyzing textual data to determine its sentiment. In this project, we train a machine learning model on a dataset of tweets to classify their sentiments.

The project includes:

Data preprocessing (tokenization, stopword removal, etc.)

Feature extraction using TF-IDF.

Model training and evaluation.

Saving the trained model for future use.

Features

Preprocess raw tweets (removing stopwords, special characters, etc.).

Extract features using TF-IDF Vectorizer.

Train machine learning models (e.g., Logistic Regression).

Evaluate the model on accuracy, precision, recall, and F1 score.

Save and load trained models.

Technologies Used

Python

Google Colab

Libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

Setup and Installation

Prerequisites

Python 3.7 or above

pip (Python package manager)

Steps

Clone the repository:

git clone https://github.com/<your_username>/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

Install the required packages:

pip install -r requirements.txt

Download the dataset:

The project uses the Sentiment140 dataset. Download it from Sentiment140 Dataset.

Place the dataset file (e.g., training.1600000.processed.noemoticon.csv) in the data/ folder.

Run the Jupyter Notebook:

jupyter notebook Twitter_Sentiment_Analysis.ipynb

Usage

Open the notebook Twitter_Sentiment_Analysis.ipynb in Google Colab or Jupyter Notebook.

Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

Use the saved model (trained_model.sav) for sentiment prediction on new tweets.

Dataset

The project uses the Sentiment140 dataset, which contains 1.6 million labeled tweets. Each tweet is labeled as:

0: Negative sentiment

4: Positive sentiment

Dataset Preparation

Download the dataset from Sentiment140 Dataset.

Preprocess the data using the provided functions in the notebook.

Results

The trained model achieves an accuracy of approximately 85% on the test dataset.

Example predictions:

Tweet: "I love this product!" → Sentiment: Positive

Tweet: "This is the worst experience ever." → Sentiment: Negative

Contributing

Contributions are welcome! If you'd like to improve the project or fix bugs, follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/new-feature).

Commit your changes (git commit -m "Add some feature").

Push to the branch (git push origin feature/new-feature).

Open a pull request.

License

This project is licensed under the MIT License - see the LICENSE file for details.
