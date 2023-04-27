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
x = np.array(houses[['grade','bathrooms','lat','sqft_living','view']])
y = np.array(houses[['price']])

mean = np.mean(x)
standard = np.std(x)
x = normalize(x , mean , standard)

m = len(x)
alpha = [0.3,0.1,0.03,0.01, 0.003, 0.001]
theta_matrix = [[0], [1] , [2] , [0] , [0] , [0]]

x0 = np.ones([x.shape[0], 1])
x_matrix = np.concatenate([x0, x], 1)

print("\nPlease enter\n1- grade\n2- bathrooms\n3- lat\n4- sqft_living\n5- view\n ")
input_lst = []
n = 5
input_lst.append([1.0])
for i in range(0, n):
    ele = float(input())#(6 , 1, 47.7379 , 770, 0)[i]
    input_lst.append([ele])
input_lst = normalize(input_lst , mean , standard)

price_lists_for_alphas = []
mean_square_error_for_alphas = []
temp_theta = theta_matrix
for alph in alpha:
    mse = 0.0
    for i in range(10):
        h = hypothesis(theta_matrix, x_matrix)
        print()
        for j in range(len(theta_matrix)):
            temp_theta[j][0] = theta_matrix[j][0] - (alph * np.sum( (h - y) * (x_matrix[:, j:j+1] ) ) ) / m
        theta_matrix = temp_theta
        mse = mean_square_error(theta_matrix, x_matrix, y, m)

        print("For iteration ", i + 1, " :")
        print("Mean Square Error = ", mse)
        print("Cost function = ", cost_function(theta_matrix, x_matrix, y, m), "\nTheta =  ", theta_matrix )

    mean_square_error_for_alphas.append(mse)
    price = np.sum(hypothesis(theta_matrix , np.transpose(input_lst )) )
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