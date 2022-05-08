import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_image(idx):
    plt.figure()
    exp = test_data.iloc[idx].values.reshape([28, 28])
    plt.imshow(exp, cmap='gray_r', )
    plt.title("actual:"+str(test_label[idx])+"predicted:"+str(prediction(mod_test_data[image])), fontsize=10)


def normalize(stuff):
    mean = np.mean(stuff.to_numpy(), axis=1, keepdims=True)
    std = np.std(stuff.to_numpy(), axis=1, keepdims=True)
    data_normalized = (stuff - mean)/std
    return data_normalized


def sigmoid(val):
    return 1/(1+np.exp(-val))


def h(x, col):
    return sigmoid(np.matmul(col.T, x))


def cost(col, label):
    out = 0
    i = 0
    y = y_new(label).copy()
    for row in model:
        out -= (y[i]*np.log(h(row.T, col))) + (1-y[i])*np.log(1-h(row.T, col))
        i += 1
    out /= model.shape[0]
    return out


def y_new(label):
    y = np.zeros(Y.shape)
    i = 0
    for val in Y:
        y[i] = 1 if val == label else 0
        i += 1
    return y


def grad_desc(theta_new, rate, epsilon, max_iters=10000):
    i = 0
    m = model.shape[0]
    old_cost = 0
    for label in range(10):
        new_cost = cost(theta_new[label], label)
        while abs(new_cost - old_cost) > epsilon and i < max_iters:
            theta_new[label] = theta_new[label] - rate*np.matmul(model.T, h(model.T, theta[label]) - y_new(label))/m
            old_cost = new_cost
            new_cost = cost(theta_new[label], label)
            i += 1
    return theta_new


def prediction(col):
    i = 0
    pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for row in theta:
        pred[i] = h(col, row)
        i += 1
    return pred.index(max(pred))


data = pd.read_csv("mnist_train.csv")
Y = data['label']
data.drop('label', axis=1, inplace=True)
model = normalize(data).to_numpy()

model = np.hstack((np.ones((model.shape[0], 1)), model))
theta = np.ones((10, model.shape[1]))

test_data = pd.read_csv("mnist_test.csv")
test_label = test_data['label']
test_data.drop('label', axis=1, inplace=True)
test_data = normalize(test_data)
mod_test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

theta = grad_desc(theta, 0.01, 0.01, 2000).copy()
correct = 0
for image in range(test_data.shape[0]):
    if prediction(mod_test_data[image]) == test_label[image]:
        correct += 1

# plt.show()

print("The accuracy of the model is around:", correct*100/test_data.shape[0])
