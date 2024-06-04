import inspect
import math
import sys
import time

import numpy as np
import shap as shap
import sklearn.metrics
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from LogisticModel import LogisticModel
import pandas as pd
import GMM
import GMM2
from skopt import gp_minimize


class ADMNC_LogisticModel:
    DEFAULT_SUBSPACE_DIMENSION = 10
    DEFAULT_REGULARIZATION_PARAMETER = 1.0
    DEFAULT_LEARNING_RATE_START = 1.0
    DEFAULT_LEARNING_RATE_SPEED = 0.1
    DEFAULT_FIRST_CONTINUOUS = 2
    DEFAULT_MINIBATCH_SIZE = 100
    DEFAULT_MAX_ITERATIONS = 50
    DEFAULT_GAUSSIAN_COMPONENTS = 4
    DEFAULT_NORMALIZING_R = 10.0
    DEFAULT_LOGISTIC_LAMBDA = 1.0

    def __init__(self, subspace_dimension=DEFAULT_SUBSPACE_DIMENSION,
                 regularization_parameter=DEFAULT_REGULARIZATION_PARAMETER,
                 learning_rate_start=DEFAULT_LEARNING_RATE_START,
                 learning_rate_speed=DEFAULT_LEARNING_RATE_SPEED,
                 gaussian_num=DEFAULT_GAUSSIAN_COMPONENTS,
                 normalizing_radius=DEFAULT_NORMALIZING_R,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 logistic_lambda=DEFAULT_LOGISTIC_LAMBDA,
                 minibatch_size=DEFAULT_MINIBATCH_SIZE,
                 first_continuous=0, logistic=None, gmm=None, threshold=-1.0, anomaly_ratio=0.1, max=0, min=0,
                 _estimator_type="classifier", classes_=None):
        # TODO los atributos logistic y gmm solo son para que funcione BayesianSearch, hay que buscar otras alternativas
        # min y max lo mismo, y estimator, y classes
        self.classes_ = classes_
        self._estimator_type = _estimator_type
        self.max = max
        self.min = min
        self.first_continuous = first_continuous
        self.logistic_lambda = logistic_lambda
        self.max_iterations = max_iterations
        self.normalizing_radius = normalizing_radius
        self.gaussian_num = gaussian_num
        self.learning_rate_speed = learning_rate_speed
        self.learning_rate_start = learning_rate_start
        self.regularization_parameter = regularization_parameter
        self.subspace_dimension = subspace_dimension
        self.minibatch_size = minibatch_size
        self.logistic = logistic
        self.gmm = gmm
        self.threshold = threshold
        self.anomaly_ratio = anomaly_ratio


    def fit(self, data, y=None):

        sampleElement = data[0]
        numElems = data.shape[0]
        minibatchFraction = self.minibatch_size / numElems
        if minibatchFraction > 1:
            minibatchFraction = 1

        self.logistic = LogisticModel(data, data.shape[1] - self.first_continuous,
                                      sampleElement,
                                      self.subspace_dimension, self.normalizing_radius, self.logistic_lambda)
        tries = 0
        while self.logistic.trained is False:
            try:
                self.logistic.trainWithSGD(data, self.max_iterations, minibatchFraction, self.regularization_parameter,
                                           self.learning_rate_start,
                                           self.learning_rate_speed)
            except Exception as e:
                tries += 1
                print("Logistic training failed, {tries} times, with Exception {e}".format(tries=tries, e=e))
                if tries > 20:
                    exit(0)
        #Para cambiar entre GMM custom y GMM de sklearn
        self.gmm = GMM2.GaussianMixtureModel(n_components=self.gaussian_num)
        # self.gmm = GaussianMixture(n_components=self.gaussian_num)
        data_cont = data[:, self.first_continuous:]
        self.gmm.fit(data_cont)

        estimators = self.getProbabilityEstimators(data)

        targetSize = int(numElems * self.anomaly_ratio)
        if targetSize <= 0: targetSize = 1

        estimators.sort()
        self.threshold = estimators[numElems - targetSize]
        self.findMinMax(data)
        self.classes_ = np.array([-1, 1])
    #Esta función actualmente no se usa ya que la versión para varios elementos funciona con uno
    def getProbabilityEstimator(self, element):
        gmmEstimator = self.gmm.predict_proba([element[self.first_continuous:]])
        # gmmEstimator = 1
        # logisticEstimator = 1
        logisticEstimator = self.logistic.getProbabilityEstimator(element)
        # TODO cuando logistic da 0.0, el fit falla (no hay log de 0), en Scala no pasa porque se suma el gmm antes del log
        # TODO mirar de calcular los centroides del gmm por adelantado con KNN como dicen en el paper original
        # DEBUG
        # print("gmm: " + str(gmmEstimator) + "   logistic: " + str(logisticEstimator))
        # TODO Sometimes logisticEstimator is so small it becomes zero, which crashes as log(0) is not defined
        if logisticEstimator == 0:
            math.log(sys.float_info.min)
        return math.log(logisticEstimator) * gmmEstimator  # TODO el score ya hace log, no hace falta log otra vez?

    def getProbabilityEstimators(self, elements):


        logisticEstimators = self.logistic.getProbabilityEstimators(elements)
        logisticEstimators = np.log(logisticEstimators)
        # logisticEstimators=np.ones(elements.shape[0])*-1
        # gmmEstimators = np.ones(elements.shape[0])
        # gmmEstimators = list(map(lambda e:self.gmm.score([e[self.first_continuous:]]),elements))
        # gmmEstimators = self.gmm.score_samples(elements)
        gmmEstimators = self.gmm.predict_proba(elements)
        # result = np.log(logisticEstimators)*gmmEstimators
        # print(logisticEstimators)

        return logisticEstimators * gmmEstimators

    def getContCat(self, dataset):  # Reordenar dataset internamente para que esté en orden categórico continuo?
        # dataset = np.array(dataset)
        df = pd.DataFrame(dataset)
        a = df._get_numeric_data()
        # b = dataset.select_dtypes(exclude=[np.number])

    def isAnomaly(self, elements):
        return self.getProbabilityEstimators(elements) > self.threshold

    # set_params: a function that sets the parameters of the model
    def set_params(self, **params):
        # loop through the parameters and assign them to the model attributes
        for key, value in params.items():
            setattr(self, key, value)
        # return the model object
        return self

    # get_params: a function that returns the parameters of the model
    def get_params(self, deep=True):
        # initialize an empty dictionary to store the parameters
        params = {}
        # loop through the model attributes and add them to the dictionary
        for key in self.__dict__:
            # check if the attribute is a model object and deep is True
            if isinstance(self.__dict__[key], ADMNC_LogisticModel) and deep:
                # recursively get the parameters of the model object
                params[key] = self.__dict__[key].get_params(deep)
            else:
                # add the attribute value to the dictionary
                params[key] = self.__dict__[key]
        # return the dictionary of parameters
        return params

    # get_param_names: a function that returns the names of the parameters of the model
    def get_param_names(self):
        # initialize an empty list to store the names
        names = []
        # loop through the model attributes and add them to the list
        for key in self.__dict__:
            # check if the attribute is a model object
            if isinstance(self.__dict__[key], ADMNC_LogisticModel):
                # recursively get the names of the model object
                names.extend(self.__dict__[key].get_param_names())
            else:
                # add the attribute name to the list
                names.append(key)
        # return the list of names
        return names

    def predict_proba(self, elements):
        return self.getProbabilityEstimators(elements)
        results = np.zeros((elements.shape[0], 2))  # TODO
        for i in range(len(elements)):
            result = self.getProbabilityEstimator(elements[i])
            interpolation = self.interpolate(result)
            results[i] = np.array([interpolation, 1 - interpolation])
        return results

    def findMinMax(self, data):
        results = self.getProbabilityEstimators(data)
        self.max = max(results)
        self.min = min(results)

    def interpolate(self, value):
        t = (value - self.min) / (self.max - self.min)
        return t

    def predict(self, elements):
        return self.getProbabilityEstimators(elements) > self.threshold

        # for i in range(len(elements)):
        #     # a = self.getProbabilityEstimator(elements[i])
        #     if self.getProbabilityEstimators(elements[i]) < self.threshold:
        #         result[i] = 1
        #     else:
        #         result[i] = 0
        # return result

    def decision_function(self, elements):
        return self.getProbabilityEstimators(elements)

    def feature_selection_before_encoding(self, elements, target, n_features, n_continuous):
        # Assuming `elements` includes both continuous and categorical features
        elementCont = elements[:, -n_continuous:]
        elementCat = elements[:, :n_features - n_continuous]

        # Variance thresholding
        self.variance_threshold.fit(elementCont)
        elementCont = self.variance_threshold.transform(elementCont)

        # Model-based feature selection for continuous features
        self.model_based_selector.fit(elementCont, target)
        cont_importances = self.model_based_selector.feature_importances_
        cont_selected_indices = np.argsort(cont_importances)[::-1][:n_continuous]
        elementCont = elementCont[:, cont_selected_indices]

        # Use mutual information for categorical features
        mutual_info = mutual_info_classif(elementCat, target, discrete_features=True)
        cat_selected_indices = np.argsort(mutual_info)[::-1][:n_features - n_continuous]
        elementCat = elementCat[:, cat_selected_indices]

        return np.hstack((elementCat, elementCont))


def kfold_csr(data, y, k, randomize=False, remove_ones=False):
    # data: una base de datos en formato csr_matrix
    # k: el número de pliegues para la validación cruzada
    # randomize: un booleano que indica si queremos mezclar los datos antes de dividirlos
    # remove_ones: un booleano que indica si queremos eliminar las filas con etiqueta 1 del conjunto de entrenamiento
    # Devuelve: una lista de tuplas, cada una con dos arrays de índices para el conjunto de entrenamiento y el de prueba

    # data = data.toarray()

    labels = y

    kf = KFold(n_splits=k, shuffle=randomize)

    results = []

    for train_index, test_index in kf.split(data):
        if remove_ones:
            mask = labels[train_index] == 0
            train_index = train_index[mask]
        results.append((train_index, test_index))

    return results


if __name__ == '__main__':
    X, y = load_svmlight_file("reduced_data_movie_0.03_4_0.4_0.3_random.libsvm")

    X = X.toarray()

    folds = kfold_csr(X, y, 5, True, False)
    results = []
    # np.seterr(all='raise')

    # X[:, 13:] += 0.1
    start_time = time.time()
    for i in range(5):
        admnc = ADMNC_LogisticModel(first_continuous=19, subspace_dimension=2, logistic_lambda=0.1,
                                    regularization_parameter=0.001, learning_rate_start=1,
                                    learning_rate_speed=0.2, gaussian_num=2, normalizing_radius=10, anomaly_ratio=0.2)
        # X=admnc.feature_selection_before_encoding(X,y,X.shape[0],14)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        X_train = X_train[y_train[:] != 1]
        y_train = y_train[y_train[:] != 1]
        # print(admnc.get_params())
        admnc.fit(X_train)
        resultsBools = admnc.isAnomaly(X_test)
        resultsProb = admnc.getProbabilityEstimators(X_test)
        result = roc_auc_score(y_test, resultsProb)
        results.append(result)
        print("AUC: " + str(roc_auc_score(y_test, resultsBools)))
        print("\nAUCProb: " + str(result))
        # precision = precision_score(y_test, resultsBools)
        # recall = recall_score(y_test, resultsBools)
        # f1 = f1_score(y_test, resultsBools)
        # # roc_auc = roc_auc_score(y_test, y_proba)
        # conf_matrix = confusion_matrix(y_test, resultsBools)
        #
        # # Printing the results
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1-Score:", f1)
        # # print("ROC AUC:", roc_auc)
        # print("Confusion Matrix:\n", conf_matrix)
        # print("Classification Report:\n", classification_report(y_test, resultsBools))
    end_time = time.time()
    feature_names = ["No_gender", "Action", "Adventure",
                     "Animation", "Children's", "Comedy",
                     "Crime",
                     "Documentary", "Drama", "Fantasy",
                     "Film-Noir", "Horror", "IMAX",
                     "Musical", "Mystery",
                     "Romance", "Sci-fi",
                     "Thriller", "War",
                     "Western",
                     "User", "Timestamp", "Rating"]
    explainer2 = shap.SamplingExplainer(admnc.getProbabilityEstimators, X_test,
                                        feature_names=feature_names, )
    # explainer2.explain(X_test[0])
    shap_values = explainer2(X_test[0])
    shap_values.feature_names = ["No_gender", "Action", "Adventure",
                                 "Animation", "Children's", "Comedy",
                                 "Crime",
                                 "Documentary", "Drama", "Fantasy",
                                 "Film-Noir", "Horror", "IMAX",
                                 "Musical", "Mystery",
                                 "Romance", "Sci-fi",
                                 "Thriller", "War",
                                 "Western",
                                 "User", "Timestamp", "Rating"]
    print(X_test[0])
    waterfall = shap.plots.waterfall(shap_values, max_display=14, show=True)
    # waterfall.
    duration = end_time - start_time
    print(f"TIME: {duration}")
    arrayResults = np.array(results)
    print(arrayResults)
    print("MEAN AND STD: " + str(arrayResults.mean()) + " | " + str(arrayResults.std()))
    space = [
        Integer(1, 10, prior="uniform", name="subspace_dimension"),
        Integer(1, 10, prior="uniform", name="learning_rate_start"),
        Real(10 ** -2, 0.5, prior="uniform", name="logistic_lambda"),
        Real(10 ** -3, 0.5, prior="uniform", name="regularization_parameter"),
        Integer(1, 10, prior="uniform", name="gaussian_num"),
        Real(2, 30, prior="uniform", name="normalizing_radius")
    ]


    @use_named_args(space)
    def objective(**params):
        # admnc2 = ADMNC_LogisticModel()
        # admnc2.set_params(**admnc.get_params())
        admnc.set_params(**params)

        a = np.mean(cross_val_score(admnc, X, y, cv=folds, n_jobs=-1, scoring="roc_auc"))
        # print(admnc.get_params())
        # print(admnc.logistic.V)
        # print(admnc.logistic.weights)
        print("AUC: " + str(a))
        return 1 - a


    # search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=200, cv=folds, verbose=0,
    #                             n_jobs=1, return_train_score=True)
    # search=GridSearchCV(xgb_model, params, scoring="roc_auc", cv=folds, n_jobs=-1)

    # search.fit(X, y)
    # print(admnc.predict(X.toarray()))
    # print(admnc.predict_proba(X.toarray()))
    res_gp = gp_minimize(objective, space, n_calls=50)
    #
    print(str(res_gp.fun))
    #
    print(res_gp)
    # csv_writer.update_csv(dataset, "XGBoost", metric, 1 - res_gp.fun)
