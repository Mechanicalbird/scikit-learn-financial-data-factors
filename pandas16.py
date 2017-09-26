import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from statistics import mean
from statistics import mean
from sklearn import svm, preprocessing, cross_validation
style.use('fivethirtyeight')

def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0

def moving_average(values):
    ma = mean(values)
    return ma

housing_data = pd.read_pickle('HPI.pickle')
housing_data = housing_data.pct_change()

housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_data.dropna(inplace=True)

housing_data['US_HPI_future'] = housing_data['Value'].shift(-1)

housing_data.dropna(inplace=True)

housing_data['label'] = list(map(create_labels,housing_data['Value'], housing_data['US_HPI_future']))

housing_data['ma_apply_example'] = pd.rolling_apply(housing_data['M30'], 10, moving_average)

print(housing_data.head())
print(housing_data.tail())

housing_data.dropna(inplace=True)

X = np.array(housing_data.drop(['label','US_HPI_future'], 1))
X = preprocessing.scale(X)

y = np.array(housing_data['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

pred = clf.predict(X_test)
print (pred)


