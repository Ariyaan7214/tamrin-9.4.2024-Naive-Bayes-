# tamrin-9.4.2024-Naive-Bayes-
(Naive Bayes) tamrin 9.4.2024 github
!pip install pyspark
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('nb').getOrCreate()
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("drive/MyDrive/airline_tweets.csv")
df.head(10)
sns.countplot(data=df,x='airline',hue='airline_sentiment')
sns.countplot(data=df,x='airline_sentiment')
df['airline_sentiment'].value_counts()
data = df[['airline_sentiment','text']]
data.head()
y = df['airline_sentiment']
X = df['text']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english').fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_tfidf,y_train)
from sklearn.metrics import classification_report
preds = nb.predict(X_test_tfidf)
from sklearn import metrics
metrics.accuracy_score(y_test, preds)
 
