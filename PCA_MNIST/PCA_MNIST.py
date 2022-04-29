import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def eig_max(eig, eig_vec):
    max1, index1, index2 = eig[0], 0, 0
    max2 = min(0, max1)
    for i in range(len(eig)):
        if max2 < eig[i]:
            if max1 < eig[i]:
                max2, index2 = max1, index1
                max1, index1 = eig[i], i
            else:
                max2, index2 = eig[i], i
    return max1, max2, eig_vec[:, index1], eig_vec[:, index2]


data = pd.read_csv("train.csv")

# Safely deleting a column from the database
label = data['label']
data.drop('label', axis=1, inplace=True)

data_standardized = StandardScaler().fit_transform(data)

CovMat = np.matmul(data_standardized.T, data_standardized)  # the cov function was giving memory allocation errors :(
# rip potato lappy

eig, eig_vec = np.linalg.eigh(CovMat)

max_eig1, max_eig2, eig_vec1, eig_vec2 = eig_max(eig, eig_vec)
vector = np.column_stack((eig_vec2, eig_vec1))

projected_data = np.matmul(vector.T, data_standardized.T)

reduced_data = np.vstack((projected_data, label)).T
reduced_data = pd.DataFrame(reduced_data, columns=['pca1', 'pca2', 'label'])
print(reduced_data)
# sns.FacetGrid(reduced_data, hue='label', height=8).map(sns.scatterplot, 'pca1', 'pca2').add_legend()
# plt.scatter(reduced_data.T[0], reduced_data.T[1])
# plt.show()