import time

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from svm import SupportVectorMachine

def prepare_data_to_classify(path_to_csv):
    df = pd.read_csv(path_to_csv, index_col=0)
    # Классифицируем только гоблинов и гулей
    df = df[(df['type']=='Ghoul') | (df['type']=='Goblin')]
    # Кодируем категориальные признаки
    df = pd.get_dummies(df, columns=['color'])
    # Отображаем таргет на множество {-1; 1}
    df['type'] = df['type'].map({'Ghoul':-1, 'Goblin':+1})

    print('Data set for classification:')
    print(df[:10])

    features = list(df.columns)
    features.remove('type')

    return train_test_split(df[features].values, df['type'].values, test_size=0.2)


def check_my_svm(X_train, y_train, X_test, y_test):

    print('|--------------------------------------------------------------|')
    print('                     Testing my SVM model                       ')
    model = SupportVectorMachine(n_passes=200)
    print('Training model...')
    start = time.time()
    model.fit(X_train, y_train, False)
    end = time.time()
    print(f'Model trained. Elapsed time: {end - start}')
    print('Testing model...')
    score = accuracy_score(model.predict(X_test), y_test)
    print(f'Accuracy score on test set: {score}')


def check_sklearn_svm(X_train, y_train, X_test, y_test):
    print('|--------------------------------------------------------------|')
    print('                Testing sklearn SVM model                       ')
    model = SVC(gamma='auto')
    print('Training model...')
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print(f'Model trained. Elapsed time: {end - start}')
    print('Testing model...')
    score = accuracy_score(model.predict(X_test), y_test)
    print(f'Accuracy score on test set: {score}')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data_to_classify('train.csv')
    check_my_svm(X_train, y_train, X_test, y_test)
    check_sklearn_svm(X_train, y_train, X_test, y_test)
