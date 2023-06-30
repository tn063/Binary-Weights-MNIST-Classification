# import libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import resources

# load dataset
print('Load MNIST dataset')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the input data
x_train = np.reshape(x_train, (60000, 784))
x_test = np.reshape(x_test, (10000, 784))

# convert the data to binary
x_train[x_train <= 127] = 0
x_train[x_train > 127] = 1
x_test[x_test <= 127] = 0
x_test[x_test > 127] = 1

# create the one_hot datatype for labels
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])

# convert labels to binary
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

# define network parameters
LearningRate = 0.9
BatchSize = 200
Epochs = 100
Momentum = 0.99
NumOfTrainSample = 60000
NumOfTestSample = 10000
NumInput = 784
NumHidden = 512
NumOutput = 10

# hidden layer
Wh = np.matrix(np.random.uniform(-1, 1, (NumHidden, NumInput)))
bh = np.random.uniform(-1, 1, (1, NumHidden))
del_Wh = np.zeros((NumHidden, NumInput))
del_bh = np.zeros((1, NumHidden))

# output layer
Wo = np.random.uniform(-1, 1, (NumOutput, NumHidden))
bo = np.random.uniform(-1, 1, (1, NumOutput))
del_Wo = np.zeros((NumOutput, NumHidden))
del_bo = np.zeros((1, NumOutput))

# train the network with back propagation, SGD
SampleIdx = np.arange(NumOfTrainSample)
t_start = t1 = dt.datetime.now()
BiAcc = np.zeros(Epochs)
MSE = np.zeros(Epochs)
max_accuracy = 0
best_weights = None
IdxCost = 0
Cost = np.zeros(np.int32(np.ceil(NumOfTrainSample / BatchSize)))
for ep in range(Epochs):
    t1 = dt.datetime.now()

    # Shuffle the training samples
    np.random.shuffle(SampleIdx)
    for i in range(0, NumOfTrainSample - BatchSize, BatchSize):
        # Mini-batch Gradient descent algorithm
        Batch_sample = SampleIdx[i:i + BatchSize]

        # print(Batch_sample)
        x = np.matrix(x_train[Batch_sample, :])
        y = np.matrix(y_train[Batch_sample, :])

        # Feedforward propagation
        a = np.sign(np.dot(x, Wh.T) + bh)
        o = np.sign(np.dot(a, Wo.T) + bo)

        # calculate mean square error
        Cost[IdxCost] = np.mean(np.mean(np.power((y - o), 2), axis=1))
        IdxCost += 1

        # calculate loss function
        do = (y - o)
        dWo = np.matrix(np.dot(do.T, a) / BatchSize)
        dbo = np.mean(do, 0)
        WoUpdate = LearningRate * dWo + Momentum * del_Wo
        boUpdate = LearningRate * dbo + Momentum * del_bo
        del_Wo = WoUpdate
        del_bo = boUpdate

        # back propagate error through sign function
        dh = np.dot(do, Wo)
        dWh = np.dot(dh.T, x) / BatchSize
        dbh = np.mean(dh, 0)

        # Update Weight
        WhUpdate = LearningRate * dWh + Momentum * del_Wh
        bhUpdate = LearningRate * dbh + Momentum * del_bh
        del_Wh = WhUpdate
        del_bh = bhUpdate

        Wo = Wo + WoUpdate
        bo += boUpdate
        Wh = Wh + WhUpdate
        bh += bhUpdate

        # binary weights
        Wo = np.sign(Wo)
        bo = np.sign(bo)
        Wh = np.sign(Wh)
        bh = np.sign(bh)
    MSE[ep] = np.mean(Cost)
    IdxCost = 0
    t2 = dt.datetime.now()
    training_time = t2 - t1
    print("Training epoch:", ep)
    print("MSE %f" % MSE[ep])
    print("Training time: %f seconds" % training_time.seconds)

    # test the model
    feed_forward = resources.FeedForward(x_test, Wh, bh, Wo, bo)
    BiOutN = feed_forward.OutN
    acc_test = resources.AccTest(BiOutN, y_test)
    BiAccuracy = acc_test.accuracy
    BiAcc[ep] = BiAccuracy
    print("Binary Accuracy:", BiAccuracy)
    print("------------------------------")

    if BiAccuracy > max_accuracy:
        max_accuracy = BiAccuracy
        best_weights = (Wh.copy(), bh.copy(), Wo.copy(), bo.copy())

# Save best weights to file
weights_params = {"Wh": best_weights[0], "bh": best_weights[1], "Wo": best_weights[2], "bo": best_weights[3]}

# Save the dictionary to a .npy file
np.save("model.npy", weights_params)

t_end = dt.datetime.now() - t_start
print("Total time: ", t_end)
print("Maximum Accuracy of Binary Weights: ")
print(max_accuracy)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axs[0].plot(BiAcc, "b-")
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_title("Accuracy of Binary Weights")

axs[1].plot(MSE, "r-")
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_title("Loss of Binary Weights")

plt.savefig("figure.png")
plt.show()

