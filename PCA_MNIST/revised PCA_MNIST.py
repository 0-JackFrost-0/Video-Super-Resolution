import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv("train.csv")

# Safely deleting a column from the database
label = data['label']
data.drop('label', axis=1, inplace=True)

i = int(input("Enter a number between 0 and 41999, and I'll serve you with a number: "))
image = data.iloc[i].values.reshape([28, 28])
plt.imshow(image, cmap='gray_r')
plt.title('Actual image '+str(i)+': Digit '+str(label[i]) + str(len(image)), fontsize=15, pad=15)


pca_100 = PCA(n_components=100)
mnist_pca_100_reduced = pca_100.fit_transform(data)
print(len(mnist_pca_100_reduced[i, :]))
mnist_pca_100_recovered = pca_100.inverse_transform(mnist_pca_100_reduced)
# np.cumsum(pca_100.explained_variance_ratio_ * 100)[-1] shows the percent of variance compared to the original image

image_pca_100 = mnist_pca_100_recovered[i, :].reshape([28, 28])
plt.figure()
plt.imshow(image_pca_100, cmap='gray_r')
plt.title('Compressed image with 100 components', fontsize=15, pad=15)
plt.show()
