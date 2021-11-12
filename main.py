# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

## populate mnist dataset
mnist = fetch_openml('mnist_784', version=1)

## show dataset
print(mnist.keys())  # show keys


X, y = mnist["data"], mnist["target"]
print(X.shape)  # show data
print(y.shape)  # show target

some_digit = X.iloc[0]
some_digit_image = some_digit.values.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

print(y.iloc[0])
y = y.astype(np.uint8) # cast y to integer

### create a training set
X_train, X_test, y_train, y_test = X.iloc[:60000], X.iloc[:60000], y.iloc[:60000], y.iloc[60000:]