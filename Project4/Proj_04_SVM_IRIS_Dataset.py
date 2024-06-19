import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

# Load the Iris dataset
irisset = datasets.load_iris()
X = irisset.data[:, :2]  # Use only the first two features
z = irisset.target

# Fit the SVM model
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, z)

# Get the separating hyperplane for each class
w = clf.coef_[0]
a = clf.coef_[1]
b = clf.coef_[2]

# Create a mesh to plot in
xpoints = np.linspace(4, 8, 100)

# Calculate decision boundary for each class
y_w = -w[0] / w[1] * xpoints - clf.intercept_[0] / w[1]
y_a = -a[0] / a[1] * xpoints - clf.intercept_[1] / a[1]
y_b = -b[0] / b[1] * xpoints - clf.intercept_[2] / b[1]

# Plotting
plt.plot(xpoints, y_w, 'r-', label='Class 0 boundary')
plt.plot(xpoints, y_a, 'g-', label='Class 1 boundary')
plt.plot(xpoints, y_b, 'b-', label='Class 2 boundary')
plt.scatter(X[:, 0], X[:, 1], c=z, cmap=plt.cm.Paired)
plt.suptitle('SVM IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(True)
plt.axis('tight')
plt.legend()
plt.show()
