import sys

from ADMNC_LogisticModel import ADMNC_LogisticModel


class ADMNC:
    # TODO poner constraints de valores en los parámetros
    def __init__(self, subspace_dimension=ADMNC_LogisticModel.DEFAULT_SUBSPACE_DIMENSION,
                 regularization_parameter=ADMNC_LogisticModel.DEFAULT_REGULARIZATION_PARAMETER,
                 learning_rate_start=ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_START,
                 learning_rate_speed=ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_SPEED,
                 gaussian_num=ADMNC_LogisticModel.DEFAULT_GAUSSIAN_COMPONENTS,
                 normalizing_radius=ADMNC_LogisticModel.DEFAULT_NORMALIZING_R,
                 max_iterations=ADMNC_LogisticModel.DEFAULT_MAX_ITERATIONS,
                 normalizing_factor=ADMNC_LogisticModel.DEFAULT_LOGISTIC_LAMBDA):
        self.normalizing_factor = normalizing_factor
        self.max_iterations = max_iterations
        self.normalizing_radius = normalizing_radius
        self.gaussian_num = gaussian_num
        self.learning_rate_speed = learning_rate_speed
        self.learning_rate_start = learning_rate_start
        self.regularization_parameter = regularization_parameter
        self.subspace_dimension = subspace_dimension
        # self.options = self.parseParams(params)

    def showUsageAndExit(self):  # TODO quizá hacer que se pueda con más tipos de datasets
        print("""Usage: ADMNC dataset [options]
        Dataset must be a libsvm file
    Options:
        -k    Intermediate parameter subspace dimension (default: """ + str(
            ADMNC_LogisticModel.DEFAULT_SUBSPACE_DIMENSION) +
              """)
          -r    Regularization parameter (default: """ + str(ADMNC_LogisticModel.DEFAULT_REGULARIZATION_PARAMETER) +
              """)
          -l0   Learning rate start (default: """ + str(ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_START) +
              """)
          -ls   Learning rate speed (default: """ + str(ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_SPEED) +
              """)
          -g    Number of gaussians in the GMM (default: """ + str(ADMNC_LogisticModel.DEFAULT_GAUSSIAN_COMPONENTS) +
              """)
          -nr   Normalizing radius (default: """ + str(ADMNC_LogisticModel.DEFAULT_NORMALIZING_R) +
              """)
          -n    Maximum number of SGD iterations (default: """ + str(ADMNC_LogisticModel.DEFAULT_MAX_ITERATIONS) +
              """)
          -ll   Normalizing factor of the logistic function (default: """ + str(
            ADMNC_LogisticModel.DEFAULT_LOGISTIC_LAMBDA) +
              """)
          -t    Test file (default: input dataset)""")
        sys.exit(-1)

    # Probablemente innecesario
    def parseParams(self, p):
        m = {"subspace_dimension": float(ADMNC_LogisticModel.DEFAULT_SUBSPACE_DIMENSION),
             "regularization_parameter": ADMNC_LogisticModel.DEFAULT_REGULARIZATION_PARAMETER,
             "learning_rate_start": ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_START,
             "learning_rate_speed": ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_SPEED,
             "first_continuous": float(ADMNC_LogisticModel.DEFAULT_FIRST_CONTINUOUS),
             "minibatch": float(ADMNC_LogisticModel.DEFAULT_MINIBATCH_SIZE),
             "gaussian_components": float(ADMNC_LogisticModel.DEFAULT_GAUSSIAN_COMPONENTS),
             "max_iterations": float(ADMNC_LogisticModel.DEFAULT_MAX_ITERATIONS),
             "normalizing_radius": ADMNC_LogisticModel.DEFAULT_NORMALIZING_R,
             "logistic_lambda": ADMNC_LogisticModel.DEFAULT_LOGISTIC_LAMBDA,
             "test_file": None}
        if len(p) <= 0:
            self.showUsageAndExit()
        p = p.split(" ")
        m["dataset"] = p[0]

        i = 1
        while i < len(p):
            if (i >= len(p) - 1) or (p[i][0] != '-'):
                print("Unknown option: " + p[i])
                self.showUsageAndExit()
            readOptionName = p[i][1:]
            option = {
                "k": "subspace_dimension",
                "l0": "learning_rate_start",
                "ls": "learning_rate_speed",
                "r": "regularization_parameter",
                "fc": "first_continuous",
                "g": "gaussian_components",
                "n": "max_iterations",
                "nr": "normalizing_radius",
                "ll": "logistic_lambda",
                "t": "test_file"
            }.get(readOptionName, readOptionName)
            if option not in m:
                print("Unknown option:" + readOptionName)
                self.showUsageAndExit()
            if option == "test_file":
                m[option] = p[i + 1]
            else:
                m[option] = float(p[i + 1])
            i = i + 2
        return m

    def fit(self, data):
        admnc = ADMNC_LogisticModel()
        # admnc.subspaceDimension = int(self.options["subspace_dimension"])
        # admnc.maxIterations = int(self.options["max_iterations"])
        # admnc.minibatchSize = int(self.options["minibatch"])
        # admnc.regParameter = float(self.options["regularization_parameter"])
        # admnc.learningRate0 = float(self.options["learning_rate_start"])
        # admnc.learningRateSpeed = float(self.options["learning_rate_speed"])
        # admnc.gaussianK = int(self.options["gaussian_components"])
        # admnc.normalizingR = float(self.options["normalizing_radius"])


if __name__ == '__main__':
    admnc = ADMNC()
    admnc.fit()
