import math
import sys

import numpy as np


UNCONVERGED_GRADIENTS = [sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max,
                         sys.float_info.max]
FULLY_CONVERGED_GRADIENTS = [sys.float_info.min, sys.float_info.min, sys.float_info.min, sys.float_info.min,
                             sys.float_info.min]


class LogisticModel:
    # TODO hacer la separación de parte numerica y continua automaticamente o con una opción si lo tienes precomputado
    def __init__(self, data, numContinuous, element, subspaceDimension, normalizing_radius, lambda_num, data2):
        self.max_values = None
        self.n_features = data.shape[1]
        self.n_continuous = numContinuous
        self.lambda_num = lambda_num
        self.normalizing_radius = normalizing_radius
        self.trained = False

        self.regParameterScaleV = 1 / math.sqrt((numContinuous + 1) * subspaceDimension)
        self.lastGradientLogs = UNCONVERGED_GRADIENTS
        X1 = data2[:, :(self.n_features - self.n_continuous)]

        X3 = self.one_hot_encode(X1)
        self.numDiscrete = len(X3[0])
        self.V = np.random.rand(subspaceDimension, self.n_continuous + 1) - 0.5
        self.weights = np.random.rand(subspaceDimension, self.numDiscrete) - 0.5

        self.regParameterScaleW = 1 / math.sqrt(self.numDiscrete * subspaceDimension)


    # def getProbabilityEstimator(self, element):
    #     # elementCat, elementCont = self._encodeData(element)
    #     # elementCat = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    #     #                         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.,
    #     #                         0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.]])
    #     # elementCont = np.array([[0.2059, 0.278, 0.3333, 1., 0.3036, 0.6667, 0.]])
    #     element = np.reshape(element,(self.n_features,))
    #     # if element.shape
    #     # a = element[0]
    #     elementCont = element[-self.n_continuous:]
    #     elementCat = element[:self.n_features - self.n_continuous]
    #     elementCat = self.one_hot_encode_elements2(elementCat)
    #     wArray = np.array(self.weights, dtype="float")
    #     xCont = np.append(elementCont, [1])
    #
    #     first = np.matmul(self.V, xCont)
    #
    #     z = np.where(elementCat == 1, 1, -1)
    #
    #     w = np.dot(first, wArray)
    #
    #     p = 1.0 / (1.0 + np.exp(-z * w / self.lambda_num))
    #
    #     return np.prod(p)

    def one_hot_encode(self, matrix):
        matrix = matrix.astype(int)
        if self.max_values is None:
            max_values = np.max(matrix, axis=0)
            self.max_values = [int(max_val)+1 for max_val in max_values]
        one_hot_encoded = []
        for i, max_val in enumerate(self.max_values):
            one_hot_matrix = np.eye(max_val)[matrix[:, i]]#TODO when test matrix has bigger max_values, it crashes
            one_hot_encoded.append(one_hot_matrix)
        one_hot_encoded_matrix = np.hstack(one_hot_encoded)


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

    # def _map_func(self, e):
    #     elemCat, elemCont = self._encodeData(e)
    #     # TODO hacer oneHot a la parte categorica, pero todo a ceros, únicamente dejando un 1 en un indice random
    #     # El label depende de si en la posición aleatoria (de 0 a len(oneHot)) hay un 1 o no
    #     index = random.randrange(0, len(elemCat))
    #     # index = 1
    #     mask = np.zeros(len(elemCat))
    #     mask[index] = 1
    #
    #     z = elemCat[index] if elemCat[index] == 1.0 else -1.0
    #     xCont = np.concatenate((elemCont, np.array([1.0])))
    #     yDisc = mask
    #     first = np.matmul(self.V, xCont)
    #     second = np.matmul(self.weights, yDisc)
    #     w = np.dot(first, second)
    #     # if w > 3:
    #     #     w
    #     # print("w: "+str(w))
    #     s = 1.0 / (1.0 + math.exp(z * w / self.lambda_num))  # TODO a veces w es enorme y crashea
    #     a1 = -s * z * (yDisc.transpose() * first.reshape((first.shape[0], 1)))
    #     a2 = -s * z * (xCont.transpose() * second.reshape((second.shape[0], 1)))
    #     return a1, a2
    def _map_func_batch(self, e_batch):
        elemConts = e_batch[:, -self.n_continuous:]
        elemCats = e_batch[:, :self.n_features - self.n_continuous]
        elemCats = self.one_hot_encode(elemCats)

        # indices = np.ones(elemCats.shape[0],dtype=int)
        indices = np.random.randint(0, elemCats.shape[1], size=elemCats.shape[0])
        masks = np.zeros_like(elemCats)
        masks[np.arange(masks.shape[0]), indices] = 1

        zs = np.where(elemCats[np.arange(elemCats.shape[0]), indices] == 1.0, 1.0, -1.0)
        xConts = np.concatenate((elemConts, np.ones((elemConts.shape[0], 1))), axis=1)
        yDiscs = masks

        first = np.matmul(xConts, self.V.T)
        second = np.matmul(yDiscs, self.weights.T)
        ws = np.einsum('ij,ij->i', first, second)

        s_values = 1.0 / (1.0 + np.exp(zs * ws / self.lambda_num))

        a1s = -s_values[:, np.newaxis, np.newaxis] * zs[:, np.newaxis, np.newaxis] * (yDiscs[:, :, np.newaxis] * first[:, np.newaxis, :])
        a2s = -s_values[:, np.newaxis, np.newaxis] * zs[:, np.newaxis, np.newaxis] * (xConts[:, :, np.newaxis] * second[:, np.newaxis, :])

        return a1s, a2s


    def _reduce_func2(self, e1, e2):
        # Sum the arrays in the tuples element-wise
        a1_sum = np.sum(e1, axis=0)
        a2_sum = np.sum(e2, axis=0)
        return a1_sum, a2_sum
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
            total = len(minibatch)  # TODO la versión de Scala hace un minibatch de tamaño aproximado, algo random
            while (total <= 0):
                minibatch = data[np.random.choice(data.shape[0], int(len(data) * minibatchFraction), replace=False), :]
                total = len(minibatch)
            learningRate = learningRate0 / (1 + learningRateSpeed * (i - 1))

            a1 = self._map_func_batch(minibatch)
            # a2 = list(map(self._map_func, minibatch))
            # b = self._map_func2(minibatch)

            # (sumW, sumV) = reduce(self._reduce_func, map(self._map_func, minibatch))
            # (sumW, sumV) = reduce(self._reduce_func, a2)
            #
            (sumW, sumV) = self._reduce_func2(a1[0],a1[1])


            sumW=sumW.transpose()
            sumV=sumV.transpose()

            gradientW = sumW / total

            gradientProgress = sum(map(math.fabs, gradientW.flatten()))

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


    def getProbabilityEstimators(self, elements,y=None):

            elementCont = elements[:, -self.n_continuous:]
            elementCat = elements[:, :self.n_features - self.n_continuous]
            elementCat=self.one_hot_encode(elementCat)
            wArray = np.array(self.weights, dtype="float")

            xCont = np.hstack((elementCont, np.ones((elementCont.shape[0], 1))))

            first = np.dot(xCont, self.V.T)

            z = np.where(elementCat == 1, 1, -1)

            w = np.dot(first, wArray)

            p = 1.0 / (1.0 + np.exp(-z * w / self.lambda_num))


            #Prueba que estaba haciendo para números muy grandes en los que se pierde la precisión
            log_p = np.log(p)
            # log_p_sum = np.sum(log_p, axis=1)
            # return abs(log_p_sum)


            return np.prod(p, axis=1)



