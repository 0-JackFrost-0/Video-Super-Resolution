import pandas as pd
import numpy as np

rate = 0.0001


def normal():
    theta_new = np.matmul(np.matmul(np.linalg.pinv(np.matmul(model.T, model)), model.T), Y)
    print("The loss using the normal equations method: ", cost(theta_new))


def h(row, column):
    return np.matmul(row, column)


def g(column):
    return np.matmul(model, column)


def cost(column):
    sum = 0
    for i in range(model.shape[0]):
        sum += (h(model[i], column) - Y[i])*(h(model[i], column) - Y[i])
    sum /= (2*model.shape[0])
    return sum


def grad_desc(theta_new):
    epsilon = 0.00001
    new_cost = cost(theta_new)
    old_cost = 0
    while abs(new_cost-old_cost) > epsilon:
        # for i in range(len(theta)):
            # term = 0
            # for index in range(model.shape[0]):
            #     term += (h(model[index], theta_old) - Y[index])*model[index][i]
            # term = (term*rate)/(model.shape[0])
            # theta_new[i] -= term
        theta_new = theta_new - (rate*(np.matmul(model.T, g(theta_new).copy() - Y))/model.shape[0])
        old_cost = new_cost
        new_cost = cost(theta_new)
    print("The loss via gradient descent:", new_cost)


data = pd.read_csv("data.csv")
regions = data['Region']
model = pd.DataFrame(data, columns=['Temperature (T)', 'Rainfall (mm)', 'Humidity (%)']).values
model = np.hstack((np.ones((15, 1)), model))
Y = pd.DataFrame(data, columns=['Oranges (ton)']).values
theta = np.zeros((model.shape[1], 1))
# grad_desc(theta)
# normal()
