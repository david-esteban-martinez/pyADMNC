import math

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


class GaussianMixtureModel:
    def __init__(self, n_components, tol=1e-6, max_iter=100, reg_covar=1e-6):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        # Use KMeans to initialize the means more robustly
        kmeans = KMeans(n_clusters=self.n_components, n_init=10)
        kmeans.fit(X)
        self.means_ = kmeans.cluster_centers_
        self.covariances_ = np.array([np.cov(X, rowvar=False)] * self.n_components)

    def _e_step(self, X):
        n_samples, _ = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * multivariate_normal(mean=self.means_[k], cov=self.covariances_[k], allow_singular=True)\
                .pdf(X)

        sum_responsibilities = np.sum(responsibilities, axis=1)[:, np.newaxis]
        sum_responsibilities += 1e-10
        responsibilities /= sum_responsibilities
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        for k in range(self.n_components):
            responsibility = responsibilities[:, k]
            total_responsibility = responsibility.sum()
            if total_responsibility == 0:  # Reinitialize the component if it has no responsibility
                self.means_[k] = X[np.random.choice(n_samples)]
                self.covariances_[k] = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
                self.weights_[k] = 1 / self.n_components
            else:
                self.weights_[k] = total_responsibility / n_samples
                self.means_[k] = np.dot(responsibility, X) / total_responsibility
                centered_X = X - self.means_[k]
                cov_matrix = np.dot(responsibility * centered_X.T, centered_X) / total_responsibility
                self.covariances_[k] = cov_matrix + self.reg_covar * np.eye(n_features)

    def fit(self, X):
        self._initialize_parameters(X)
        log_likelihood = -np.inf

        for i in range(self.max_iter):
            prev_log_likelihood = log_likelihood
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            log_likelihood = np.sum(np.log(np.sum([
                self.weights_[k] * multivariate_normal(mean=self.means_[k], cov=self.covariances_[k], allow_singular=True)\
                .pdf(X) + 1e-10
                for k in range(self.n_components)
            ], axis=0)))

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

    def predict_proba(self, X):
        return np.sum([
            self.weights_[k] * multivariate_normal(mean=self.means_[k], cov=self.covariances_[k], allow_singular=True)\
                .pdf(X)
            for k in range(self.n_components)
        ], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def pdf(self, X):
        return self.predict_proba(X)