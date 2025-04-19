# SMS Spam Classifier

This is my first NLP project to classify SMS messages as spam or not spam using Python.

## What It Does

It takes a dataset of SMS messages, cleans the text, and uses a machine learning model to tell if a message is spam or not.

## Dataset

I used the SMS Spam Collection dataset. It has lots of SMS messages labeled as `spam` or `ham`.

- Put the `SMSSpamCollection` file in a folder called `smsspamcollection`.

## Setup

You need these Python libraries:

- pandas
- matplotlib
- seaborn
- nltk
- scikit-learn

Install them with:

```bash
pip install pandas matplotlib seaborn nltk scikit-learn

Also, you need to download NLTK stopwords:

import nltk
nltk.download('stopwords')

## How to Run

1. Clone my project:

git clone https://github.com/m2ry2m/NLP-spam-classifier.git

2. Put the `SMSSpamCollection` file in the `smsspamcollection` folder.

3. Run the code:

python nlp_spam_classifier.py

## What My Code Does

- Loads the SMS data.
- Checks the data and makes some graphs.
- Cleans the text (removes punctuation and extra words).
- Trains a model to classify spam messages.
- Shows how well the model works.

## Results

The model tells you how good it is at finding spam messages with some scores.

## License

This project uses the MIT License.
