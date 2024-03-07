import math
import random
import sys
from functools import reduce

import numpy as np
from sklearn.preprocessing import OneHotEncoder

UNCONVERGED_GRADIENTS = [sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max,
                         sys.float_info.max]
FULLY_CONVERGED_GRADIENTS = [sys.float_info.min, sys.float_info.min, sys.float_info.min, sys.float_info.min,
                             sys.float_info.min]


class LogisticModel:
    # TODO hacer la separación de parte numerica y continua automaticamente o con una opción si lo tienes precomputado
    def __init__(self, data, numContinuous, element, subspaceDimension, normalizing_radius, lambda_num):
        self.enc = OneHotEncoder()
        self.n_features = data.shape[1]
        self.n_continuous = numContinuous
        self.lambda_num = lambda_num
        self.normalizing_radius = normalizing_radius

        self.regParameterScaleV = 1 / math.sqrt((numContinuous + 1) * subspaceDimension)
        self.lastGradientLogs = UNCONVERGED_GRADIENTS
        X1 = data[:, :(self.n_features - self.n_continuous)]
        self.enc.fit(X1)
        self.max_values = np.empty(X1.shape[1])
        elementCat, _ = self._encodeData(element)
        self.numDiscrete = len(elementCat)
        self.V = np.random.rand(subspaceDimension, self.n_continuous + 1) - 0.5
        self.weights = np.random.rand(subspaceDimension, self.numDiscrete) - 0.5
        # self.V = np.ones((subspaceDimension, self.n_continuous + 1)) - 0.5
        # self.weights = np.ones((subspaceDimension, self.numDiscrete)) - 0.5
        self.regParameterScaleW = 1 / math.sqrt(self.numDiscrete * subspaceDimension)
        for i in range(X1.shape[1]):
            max_value = np.amax(X1[:, i])
            self.max_values[i] = max_value

    def getProbabilityEstimator(self, element):
        elementCat, elementCont = self._encodeData(element)
        x = elementCat.size
        wArray = np.array(self.weights, dtype="float")
        array = np.zeros(x)
        for i in range(x):
            yDisc = [0] * x
            yDisc[i] = 1

            xCont = np.append(elementCont, [1])

            if elementCat[i] == 1:
                z = 1
            else:
                z = -1
            first = np.matmul(self.V, xCont)
            second = np.matmul(wArray, yDisc)
            w = np.dot(first, second)
            p = 1.0 / (1.0 + math.exp(-z * w / self.lambda_num))
            array[i] = p
        return np.multiply.reduce(array)

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
        index =random.randrange(0, len(elemCat))
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
        print("w: "+str(w))
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

    def getProbabilityEstimators(self, elements):
        return list(map(self.getProbabilityEstimator, elements))
