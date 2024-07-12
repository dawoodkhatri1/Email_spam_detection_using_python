# Email_spam_detection_using_python

You can run the file in vscode and pycharm

If the library of the code isn't installed, you can easily installed it by:

> pip install pandas
>> pip install seaborn
>>> pip install matplotlib
>>>> pip install scikit-learn
>>>>> pip install nltk
>>>>>> pip install faker
>>>>>>> pip install string

The code is divided into these things:

**Imports:** 

Various libraries are imported including pandas for data manipulation, seaborn and matplotlib for visualization, scikit-learn for machine learning tasks, NLTK for natural language processing, and Faker for generating fake data.

**Fake Data Generation:**

Faker is used to generate fake text data (fake.text()) and labels ('ham' or 'spam'). This simulated data is stored in fake_data.

**Data Loading and Concatenation:**

A real dataset (spam.csv) is loaded using pandas into df.
The generated fake data (fake_df) is concatenated with df to create a larger dataset.

**Text Preprocessing:**

A preprocess function is defined to clean and normalize text data:
Removes punctuation using str.maketrans and translate.
Tokenizes text into words using NLTK's word_tokenize.
Converts words to lowercase.
Removes stopwords (common words like 'the', 'is', etc.).
Stems words using NLTK's SnowballStemmer.
The preprocess function is applied to the 'text' column of df to preprocess all text data.

**Pipeline Creation:**

A scikit-learn Pipeline is created:
TfidfVectorizer is used to convert text data into TF-IDF features.
MultinomialNB is used as the classifier.

**Hyperparameter Tuning:**

GridSearchCV is employed to search for the best combination of hyperparameters ('tfidf__max_df', 'tfidf__ngram_range', 'nb__alpha') using stratified k-fold cross-validation (StratifiedKFold).

**Model Training and Evaluation:**

The best model found by GridSearchCV is trained on the entire dataset (X, y).
The model is evaluated on a held-out test set (X_test, y_test) using accuracy and classification report metrics.
A confusion matrix is generated to visualize the performance of the model.

**Prediction Function:**

A predict_message function is defined to predict whether a given message is 'ham' or 'spam':
It preprocesses the input message using the preprocess function.
Uses the trained grid_search model to predict the label ('ham' or 'spam') of the message.

**Testing the Prediction Function:**

The predict_message function is tested with two example messages to demonstrate its functionality.
