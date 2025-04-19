# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# Step 1: Load the SMS dataset
# Loads the SMS Spam Collection dataset into a pandas DataFrame
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
print(f"Total messages: {len(messages)}")

# Step 2: Exploratory Data Analysis (EDA)
# Add a column for message length to explore its distribution
messages['length'] = messages['message'].apply(len)

# Display basic statistics of the dataset
print(messages.describe())
print(messages.groupby('label').describe())

# Visualize the distribution of message lengths
messages['length'].plot(bins=50, kind='hist')
plt.title('Message Length Distribution')
plt.xlabel('Length')
plt.show()

# Visualize message length distribution by label (spam vs ham)
messages.hist(column='length', by='label', bins=50, figsize=(12, 4))
plt.suptitle('Message Length Distribution by Label')
plt.show()

# Step 3: Text Preprocessing
# Define a function to clean text by removing punctuation and stopwords
def text_process(mess):
    """
    Process a text message by:
    1. Removing punctuation
    2. Removing stopwords
    3. Returning a list of cleaned words
    """
    # Remove punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Step 4: Create a Pipeline for Text Processing and Classification
# Split the data into training and testing sets (80% train, 20% test)
msg_train, msg_test, label_train, label_test = train_test_split(
    messages['message'], messages['label'], test_size=0.2, random_state=42
)

# Define the pipeline with three steps:
# 1. Convert text to token counts (Bag of Words)
# 2. Transform token counts to TF-IDF scores
# 3. Train a Naive Bayes classifier
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Step 5: Train the Model and Evaluate
# Fit the pipeline on the training data
pipeline.fit(msg_train, label_train)

# Make predictions on the test set
predictions = pipeline.predict(msg_test)

# Print the classification report to evaluate the model
print("Classification Report:")
print(classification_report(label_test, predictions))