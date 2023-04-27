import np as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalize(value , mean , standard):
    value = value - mean
    value /= standard
    return value


def hypothesis(theta_transpose, x):
    return np.dot(x, theta_transpose)


def cost_function(theta_transpose, x, y, m):
    cost = np.sum(((hypothesis(theta_transpose, x) - y) ** 2))
    cost /= (m * 2)
    return cost


def mean_square_error(theta_transpose, x, y, n):
    return np.sum(((hypothesis(theta_transpose, x) - y) ** 2)) / n


houses = pd.read_csv('house_data.csv')
x = np.array(houses[['sqft_living']])
y = np.array(houses[['price']])

mean = np.mean(x)
standard = np.std(x)
x = normalize(x , mean , standard)

m = len(x)
alpha = [0.3,0.1,0.03,0.01, 0.003, 0.001]
theta_matrix = [[0], [0]]

x0 = np.ones([x.shape[0], 1])
x_matrix = np.concatenate([x0, x], 1)

houseInput = input("Enter your square feet living for your house : ")
houseInput = int(houseInput)
houseInput = normalize(houseInput , mean , standard)

price_lists_for_alphas = []
mean_square_error_for_alphas = []
for alph in alpha:
    mse = 0.0
    for i in range(10):
        h = hypothesis(theta_matrix, x_matrix)
        temp0 = theta_matrix[0][0] - (alph * np.sum(h - y)) / m
        temp1 = theta_matrix[1][0] - (alph * np.sum((h - y) * x)) / m
        theta_matrix[0][0] = temp0
        theta_matrix[1][0] = temp1
        mse = mean_square_error(theta_matrix, x_matrix, y, m)

        print("For iteration ", i, " : \n" , "Mean Square Error = ", mse)
        print("Cost function = ", cost_function(theta_matrix, x_matrix, y, m), "\nTheta0 =  ", temp0, "  Theta1 =  ", temp1, "\n")

    mean_square_error_for_alphas.append(mse)
    price = np.sum(hypothesis(theta_matrix ,  np.array([1 , houseInput ])))
    price_lists_for_alphas.append(price)

    print('Cost = ', round(price,2))

print('\n')
for i in range(len(price_lists_for_alphas)):
    print('for alpha = ' , alpha[i], ' ' *  (5 - len(str(alpha[i]))) ,
    ', price = ', round(price_lists_for_alphas[i],2) , ', and MSE = ', round(mean_square_error_for_alphas[i],2))

plt.plot(alpha,mean_square_error_for_alphas )
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('Mean Square Error to Alpha Graph')
plt.show()