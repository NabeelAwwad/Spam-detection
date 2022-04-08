import pandas as pd
import re
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# spell correction
# from autocorrect import spell


# load the data into dataset
dataset = pd.read_csv('Output2.csv', low_memory=False);
data = []
stemmer = PorterStemmer()
# text processing
for i in range(dataset.shape[0]):
    review = dataset.iloc[i, 2]
    # remove non alphabetic characters
    review = re.sub('[^A-Za-z]', ' ', review)
    # make words lowercase
    review = review.lower()
    # tokenization
    tokenized_review = wt(review)
    # removing stop words
    review_processed = []
    for word in tokenized_review:
        if word not in set(stopwords.words('english')):
            # stemming
            review_processed.append(stemmer.stem(word))
    review_text = " ".join(review_processed)
    data.append(review_text)

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

features = CountVectorizer(max_features=2000, ngram_range=(2, 2))
X = features.fit_transform(data).toarray()
y = dataset.iloc[:, 9]
# Use one of these two feature vectors
# features2 = CountVectorizer(max_features=1000)
features2 = TfidfVectorizer()
W = features2.fit_transform(data).toarray()

# split train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
W_train, W_test, y2_train, y2_test = train_test_split(W, y)

# Classification
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# change one of the classifiers to Decisiontree in case you want to use it
# we noticed that DecisionTrees took a very very long time to compile
classifier = GaussianNB()
classifier2 = GaussianNB()
classifier.fit(X_train, y_train)
classifier2.fit(W_train, y2_train)

# Making predictions
y_pred = classifier.predict(X_test)
y2_pred = classifier2.predict(W_test)

# comparing predictions to results
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

cm2 = confusion_matrix(y_test, y2_pred)
cr2 = classification_report(y_test, y2_pred)
accuracy2 = accuracy_score(y2_test, y2_pred)
print("first results: Bigrams")
print("confusion matrix:")
print(cm)
print("precision and recall:")
print(cr)
print("accuracy:")
print(accuracy)
print("second results: TF-IDF")
print("confusion matrix:")
print(cm2)
print("precision and recall:")
print(cr2)
print("accuracy:")
print(accuracy2)
