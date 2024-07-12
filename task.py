import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
import nltk
from faker import Faker

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Faker
fake = Faker()

# Generate fake data
fake_data = {
    'text': [fake.text() for _ in range(1000)],
    'label': [fake.random_element(elements=('ham', 'spam')) for _ in range(1000)]
}

# Convert to DataFrame
fake_df = pd.DataFrame(fake_data)

# Load the actual dataset
df = pd.read_csv('spam.csv')

# Combine with fake data
df = pd.concat([df, fake_df], ignore_index=True)

# Preprocessing function
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize words
    words = word_tokenize(text)
    # Convert to lower case
    words = [word.lower() for word in words]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to the text data
df['text'] = df['text'].apply(preprocess)

# Split the data into features and labels
X = df['text']
y = df['label']

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Hyperparameter tuning
parameters = {
    'tfidf__max_df': [0.9, 0.95, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.01, 0.1, 1]
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Predict the labels for the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = grid_search.predict(X_test)

# Evaluate the model
print("Best parameters found: ", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Function to predict if a message is spam or ham
def predict_message(message):
    processed_message = preprocess(message)
    return grid_search.predict([processed_message])[0]

# Test the function
print(predict_message("Win a free ticket now!"))
print(predict_message("Can we meet tomorrow?"))
