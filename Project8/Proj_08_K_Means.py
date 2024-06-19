import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix

X,y = make_blobs(n_samples=2500,centers=4,n_features=2,random_state=10)
y_pred = KMeans(n_clusters=4,random_state=0).fit_predict(X)

cmat = confusion_matrix(y,y_pred)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet',s=10)
plt.suptitle('K-Means Clusters')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='jet',s=10)
plt.suptitle('K-Means Clusters')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()