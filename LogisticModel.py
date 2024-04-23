import math
import random
import sys
from functools import reduce

import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder

UNCONVERGED_GRADIENTS = [sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max,
                         sys.float_info.max]
FULLY_CONVERGED_GRADIENTS = [sys.float_info.min, sys.float_info.min, sys.float_info.min, sys.float_info.min,
                             sys.float_info.min]


class LogisticModel:
    # TODO hacer la separación de parte numerica y continua automaticamente o con una opción si lo tienes precomputado
    def __init__(self, data, numContinuous, element, subspaceDimension, normalizing_radius, lambda_num):
        self.max_values = None
        self.enc = OneHotEncoder()
        self.n_features = data.shape[1]
        self.n_continuous = numContinuous
        self.lambda_num = lambda_num
        self.normalizing_radius = normalizing_radius

        self.regParameterScaleV = 1 / math.sqrt((numContinuous + 1) * subspaceDimension)
        self.lastGradientLogs = UNCONVERGED_GRADIENTS
        X1 = data[:, :(self.n_features - self.n_continuous)]
        self.enc.fit(X1)
        X2 = self.enc.transform(X1)
        # X3 = pandas.get_dummies(pandas.DataFrame(X1))
        # X1 = self.one_hot_encode_matrix(X1)
        X3 = self.one_hot_encode(X1)
        X4 = self.one_hot_encode2(X1)
        # self.max_values = np.empty(X1.shape[1])
        elementCat2 = self.one_hot_encode_elements2(X1[0])
        elementCat1 = self.one_hot_encode_elements(X1[0])
        elementCat, _ = self._encodeData(X1[0])
        elementCat3 = self.one_hot_encode_elements3(X1[0])
        self.numDiscrete = len(elementCat)
        self.V = np.random.rand(subspaceDimension, self.n_continuous + 1) - 0.5
        self.weights = np.random.rand(subspaceDimension, self.numDiscrete) - 0.5
        # self.V = np.ones((subspaceDimension, self.n_continuous + 1)) - 0.5
        # self.weights = np.ones((subspaceDimension, self.numDiscrete)) - 0.5
        self.regParameterScaleW = 1 / math.sqrt(self.numDiscrete * subspaceDimension)
        # for i in range(X1.shape[1]):
        #     max_value = np.amax(X1[:, i])
        #     self.max_values[i] = max_value
        self.trained = False

    def getProbabilityEstimator(self, element):
        # elementCat, elementCont = self._encodeData(element)
        # elementCat = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        #                         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.,
        #                         0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.]])
        # elementCont = np.array([[0.2059, 0.278, 0.3333, 1., 0.3036, 0.6667, 0.]])

        elementCont = element[-self.n_continuous:]
        elementCat = element[:self.n_features - self.n_continuous]
        elementCat = self.one_hot_encode_elements2(elementCat)
        wArray = np.array(self.weights, dtype="float")
        xCont = np.append(elementCont, [1])

        first = np.matmul(self.V, xCont)

        z = np.where(elementCat == 1, 1, -1)

        w = np.dot(first, wArray)

        p = 1.0 / (1.0 + np.exp(-z * w / self.lambda_num))

        return np.prod(p)

    def one_hot_encode(self, matrix):
        # Get the maximum value for each column in the matrix
        matrix = matrix.astype(int)
        max_values = np.max(matrix, axis=0)
        # Create a list to hold the one-hot encoded matrices
        one_hot_encoded = []
        # Iterate over each column and create a one-hot encoded matrix for it
        for i, max_val in enumerate(max_values):
            # Create a one-hot encoded matrix for the current column
            one_hot_matrix = np.eye(int(max_val) + 1)[matrix[:, i]]
            # Append the one-hot encoded matrix to the list
            one_hot_encoded.append(one_hot_matrix)
        # Concatenate the one-hot encoded matrices horizontally
        one_hot_encoded_matrix = np.hstack(one_hot_encoded)
        self.max_values = [int(max_val)+1 for max_val in max_values]

        return one_hot_encoded_matrix

    def one_hot_encode2(self, matrix):
        # Ensure the matrix is of type int
        matrix = matrix.astype(int)
        # Get the maximum value for each column in the matrix
        max_values = np.max(matrix, axis=0)
        # Calculate the total number of one-hot columns needed
        total_cols = sum(max_val + 1 for max_val in max_values)
        # Initialize the one-hot encoded matrix with zeros
        one_hot_encoded_matrix = np.zeros((matrix.shape[0], total_cols), dtype=int)
        # The starting index for each one-hot encoded column block
        col_start = 0
        # Iterate over each column and create a one-hot encoded matrix for it
        for i, max_val in enumerate(max_values):
            # The indices where ones should be placed
            indices = matrix[:, i] + col_start
            # Place ones in the appropriate positions
            one_hot_encoded_matrix[np.arange(matrix.shape[0]), indices] = 1
            # Update the starting index for the next column block
            col_start += max_val + 1
        # Store the max_values for potential future use
        self.max_values = [int(max_val) + 1 for max_val in max_values]
        return one_hot_encoded_matrix
    def one_hot_encode_elements2(self, elements):
        if len(elements) != len(self.max_values):
            raise ValueError("Length of elements array and max_values array must be the same.")
        total_length = sum(max_val for max_val in self.max_values)
        one_hot_encoded_array = np.zeros((total_length,))
        col_start = 0
        for element, max_val in zip(elements, self.max_values):
            one_hot_encoded_array[col_start + int(element)] = 1
            col_start += max_val
        return one_hot_encoded_array
    def one_hot_encode_elements(self, elements):
        if len(elements) != len(self.max_values):
            raise ValueError("Length of elements array and max_values array must be the same.")
        one_hot_encoded_elements = []
        for element, max_val in zip(elements, self.max_values):
            one_hot_encoded_element = np.eye(max_val)[int(element)]
            one_hot_encoded_elements.append(one_hot_encoded_element)
        one_hot_encoded_array = np.hstack(one_hot_encoded_elements)
        return one_hot_encoded_array



    def one_hot_encode_elements3(self, elements):
        if len(elements) != len(self.max_values):
            raise ValueError("Length of elements array and max_values array must be the same.")

        # Preallocate memory for the one-hot encoded array
        total_length = sum(int(max_val) for max_val in self.max_values)
        one_hot_encoded_array = np.zeros((total_length,))

        # Convert max_values to integers
        # Vectorized one-hot encoding
        col_start = np.cumsum([0] + self.max_values[:-1])
        one_hot_encoded_array[col_start + elements.astype(int)] = 1

        return one_hot_encoded_array

    def update(self, gradientW, gradientV, learningRate, regularizationParameter):
        # TODO falta normalizar, quizá hace falta
        self.weights = self.weights - learningRate * (
                gradientW + 2 * regularizationParameter * self.regParameterScaleW * self.weights)
        self.V = self.V - learningRate * (gradientV + 2 * regularizationParameter * self.regParameterScaleV * self.V)

        self.weights = self._normalizeColumns(self.weights)
        self.V = self._normalizeColumns(self.V)

    def _normalizeColumns(self, m):
        for i in range(m.shape[1]):
            total = 0.0
            for j in range(m.shape[0]):
                total += m[j][i] * m[j][i]
            total = math.sqrt(total)
            if total > self.normalizing_radius:
                for j in range(m.shape[0]):
                    m[j][i] = m[j][i] * self.normalizing_radius / total
        return m

    def _map_func(self, e):
        # Get the random masked representation element
        elemCat, elemCont = self._encodeData(e)
        # TODO hacer oneHot a la parte categorica, pero todo a ceros, únicamente dejando un 1 en un indice random
        # Assign z based on the label
        # El label depende de si en la posición aleatoria (de 0 a len(oneHot)) hay un 1 o no
        index = random.randrange(0, len(elemCat))
        mask = np.zeros(len(elemCat))
        mask[index] = 1

        z = elemCat[index] if elemCat[index] == 1.0 else -1.0
        # Concatenate e.cPart and 1.0
        xCont = np.concatenate((elemCont, np.array([1.0])))
        # Assign yDisc to elem.mPart
        yDisc = mask
        # Calculate w
        first = np.matmul(self.V, xCont)
        second = np.matmul(self.weights, yDisc)
        w = np.dot(first, second)
        # if w > 3:
        #     w
        # print("w: "+str(w))
        s = 1.0 / (1.0 + math.exp(z * w / self.lambda_num))  # TODO a veces w es enorme y crashea
        # Return a tuple of two arrays
        a1 = -s * z * (yDisc.transpose() * first.reshape((first.shape[0], 1)))
        a2 = -s * z * (xCont.transpose() * second.reshape((second.shape[0], 1)))
        return a1, a2

    def _encodeData(self, element):
        elementCont = element[-self.n_continuous:]
        elementCat = element[:self.n_features - self.n_continuous]
        # Transform the element into a one-hot encoded array
        one_hot = self.enc.transform([elementCat]).toarray()

        counter = 0
        space = False  # TODO hay que ver como automatizar esto

        if (space):
            new_data = np.insert(one_hot, counter, 0, axis=1)
            counter += 1
            for value in self.max_values:
                new_data = np.insert(new_data, int(value) + counter, 0, axis=1)
                counter += int(value)
                counter += 1
            new_data = new_data[0][:len(new_data[0]) - 1]
        else:
            new_data = one_hot[0]
        return new_data, elementCont

    def _reduce_func(self, e1, e2):
        # Return a tuple of two arrays
        return e1[0] + e2[0], e1[1] + e2[1]

    def trainWithSGD(self, data, maxIterations, minibatchFraction,
                     regParameter, learningRate0, learningRateSpeed):

        # SGD
        consecutiveNoProgressSteps = 0
        i = 1
        # print(self.V)
        # print(self.weights)
        while ((i < maxIterations) and (consecutiveNoProgressSteps < 10)):
            # Finishing condition: gradients are small for several iterations in a row
            # Use a minibatch instead of the whole dataset
            minibatch = data[np.random.choice(data.shape[0], int(len(data) * minibatchFraction), replace=False), :]
            # minibatch=data
            total = len(minibatch)  # TODO la versión de Scala hace un minibatch de tamaño aproximado, algo random
            while (total <= 0):
                minibatch = data[np.random.choice(data.shape[0], int(len(data) * minibatchFraction), replace=False), :]
                total = len(minibatch)
            # minibatch = data
            learningRate = learningRate0 / (1 + learningRateSpeed * (i - 1))

            # Broadcast samples and parallelize on the minibatch
            # val bSample=sc.broadcast(sample)
            a = list(map(self._map_func, minibatch))

            # (sumW, sumV) = reduce(self._reduce_func, map(self._map_func, minibatch))
            (sumW, sumV) = reduce(self._reduce_func, a)
            #
            # print(len(a))
            # exit()
            sumX = sumW.transpose()
            gradientW = sumW / total
            # gradientX=sumW/total
            # val gradientProgress=Math.abs(sum(gradientWxy))
            gradientProgress = sum(map(math.fabs, gradientW.flatten()))
            # gradientProgress2=sum(map(math.fabs,gradientX.flatten()))
            # sum(gradientW.map({x = > Math.abs(x)}))
            if (gradientProgress < 0.00001):
                consecutiveNoProgressSteps = consecutiveNoProgressSteps + 1
            else:
                consecutiveNoProgressSteps = 0

            # DEBUG
            # if (i % 10 == 0)
            #     println("Gradient size:" + gradientProgress)

            # println("Gradient size:"+gradientProgress)

            if (i >= maxIterations - len(self.lastGradientLogs)):
                self.lastGradientLogs[i - maxIterations + len(self.lastGradientLogs)] = math.log(gradientProgress)

            self.update(gradientW, (sumV / total), learningRate, regParameter)

            i = i + 1
        if (consecutiveNoProgressSteps >= 10):
            self.lastGradientLogs = FULLY_CONVERGED_GRADIENTS
        self.trained = True

    def getProbabilityEstimators(self, elements):
        elementCont = elements[:, -self.n_continuous:]
        elementCat = elements[:, :self.n_features - self.n_continuous]
        elementCat=self.one_hot_encode(elementCat)#TODO comprobar que implementación es más rápida, que mientras se computa Quantus es imposible
        # Convert weights to a NumPy array
        wArray = np.array(self.weights, dtype="float")

        # Append 1 to the continuous features for the bias term
        xCont = np.hstack((elementCont, np.ones((elementCont.shape[0], 1))))

        # Vectorized matrix multiplication
        first = np.dot(xCont, self.V.T)

        # Vectorized element-wise comparison
        z = np.where(elementCat == 1, 1, -1)

        # Vectorized dot product
        w = np.dot(first, wArray)

        # Vectorized sigmoid function
        p = 1.0 / (1.0 + np.exp(-z * w / self.lambda_num))

        # Vectorized product along the rows
        return np.prod(p, axis=1)
        # return list(map(self.getProbabilityEstimator, elements))



