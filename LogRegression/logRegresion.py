import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from config import Config
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    dataFeature = csr_matrix(feature_to_add)
    return hstack([X, dataFeature], 'csr')

np.random.seed(7)

print('Read train')
df = pd.read_csv('train.csv',delimiter=',',names=['id', 'article','numP','numA','hyperpartisan'])
df = df.values
print('Fit TfidfVectorizer')
tfidf_vect = TfidfVectorizer(min_df=3).fit(df[:,1])
print('Train to transform and array')
articleTransform =tfidf_vect.transform(df[:,1])

print('Train append')
df[:,2:4] = df[:,2:4].astype('int')
transformer = MaxAbsScaler().fit(df[:,2:4])
X_train = add_feature(articleTransform, transformer.transform(df[:,2:4]))
y_train = df[:,4].astype('int')


print('Read Validate')
df = pd.read_csv('validate.csv',delimiter=',',names=['id', 'article','numP','numA','hyperpartisan'])
df = df.values
print('Validate transform and array')
articleTransform =tfidf_vect.transform(df[:,1].astype('U'))

df[:,2:4] = df[:,2:4].astype('int')
X_test = add_feature(articleTransform, transformer.transform(df[:,2:4]))
y_test= df[:,4].astype('int')


model = LogisticRegression()
print('LogisticRegression fit')
model.fit(X_train,y_train)


print('LogisticRegression predict')
predictions = model.predict(X_test)
print(accuracy_score(y_test,predictions))