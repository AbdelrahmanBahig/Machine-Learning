import numpy as np
import pandas as pd


def hypothesis(theta, x):
    return 1 / (1 + np.exp(-(np.dot(x, theta))))


def cost(theta, x, y):
    cost = np.sum((y * np.log(hypothesis(theta, x))) + ((1 - y) * np.log(1 - hypothesis(theta, x))))
    return cost


def gradient(theata, x, y, numofiteration):
    alpha = 0.00003
    values = []
    m = x.shape[0]
    for i in range(numofiteration):

        theata = theata - alpha/m * np.dot(x.T,  hypothesis(theata, x)-y)

        values.append(cost(theata, x, y))
    return theata, values


def prediction(theata, x):
    predict = np.zeros(x.shape[0])
    for i in range(predict.size):
        if hypothesis(theata, x[i]) >= 0.5:
            predict[i] = 1
        else:
            predict[i] = 0
    return predict


heart = pd.read_csv('heart.csv', index_col=0)
x = np.array(heart[['trestbps', 'chol', 'thalach', 'oldpeak']])
y = np.array(heart['target'])

m, n = x.shape
x = np.concatenate([np.ones((m, 1)), x], axis=1)

theata = np.zeros(x.shape[1])

theata, values = gradient(theata, x, y, 20000)
print('theta :', theata)

predict = prediction(theata, x)
print(' Accuracy: {:.2f} %'.format(np.mean(predict == y) * 100))
