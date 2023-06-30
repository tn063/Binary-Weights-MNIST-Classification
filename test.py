# This test utilizes the optimal weight parameters
import numpy as np
import tensorflow as tf
import resources

# Load the weights from the model
loaded_weights = np.load("model.npy", allow_pickle=True).item()

# load dataset
print('Load MNIST dataset')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the input data
x_test = np.reshape(x_test, (10000, 784))

# convert the data to binary
x_test[x_test <= 127] = 0
x_test[x_test > 127] = 1

# create the one_hot datatype for labels
y_test = np.matrix(np.eye(10)[y_test])

# convert labels to binary
y_test[y_test == 0] = -1

# Retrieve the loaded weights
Wh = loaded_weights.get("Wh")
bh = loaded_weights.get("bh")
Wo = loaded_weights.get("Wo")
bo = loaded_weights.get("bo")

feed_forward = resources.FeedForward(x_test, Wh, bh, Wo, bo)
BiOutN = feed_forward.OutN
acc_test = resources.AccTest(BiOutN, y_test)
BiAccuracy = acc_test.accuracy
print("Binary Accuracy:", BiAccuracy)