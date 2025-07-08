import numpy as np
import numba
from copy import deepcopy
from joblib import Parallel, delayed


def _get_rng(seed):
    """Return a new NumPy random generator."""
    return np.random.default_rng(seed)


def mean(X, n_replications, seed=None):
    """Simulate the posterior distribution of the mean."""
    rng = _get_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications)
    # use matrix multiplication for efficiency
    return X @ weights.T


def var(X, n_replications, seed=None):
    """Simulate the posterior distribution of the variance."""
    rng = _get_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications)
    # Vectorized calculation of variance: E[X^2] - (E[X])^2
    X = np.asarray(X)
    X_squared = np.square(X)
    weighted_X_squared = X_squared @ weights.T
    weighted_X = X @ weights.T
    return weighted_X_squared - np.square(weighted_X)


def covar(X, Y, n_replications, seed=None):
    """Simulate the posterior distribution of the covariance."""
    rng = _get_rng(seed)
    # Transpose weights so columns correspond to replications
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T
    return _weighted_covariance(np.asarray(X), np.asarray(Y), weights)


def pearsonr(X, Y, n_replications, seed=None):
    """
    Pearson correlation coefficient and p-value for testing non-correlation.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    """
    rng = _get_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T
    return _weighted_pearsonr(np.asarray(X), np.asarray(Y), weights)


def _weighted_covariance(X, Y, weights):
    """Return weighted covariance for multiple weight sets."""
    # weights shape: (n_samples, n_replications)
    weighted_X = X @ weights
    weighted_Y = Y @ weights
    weighted_XY = (X * Y) @ weights
    return weighted_XY - (weighted_X * weighted_Y)


def _weighted_pearsonr(X, Y, weights):
    """Weighted Pearson correlation for multiple weight sets."""
    cov_XY = _weighted_covariance(X, Y, weights)
    var_X = _weighted_covariance(X, X, weights)
    var_Y = _weighted_covariance(Y, Y, weights)
    return cov_XY / np.sqrt(var_X * var_Y)


def _weighted_ls(X, w, y):
    x_rows, x_cols = X.shape
    w_matrix = np.array(w) * np.eye(x_rows)
    coef = np.dot(
        np.dot(np.dot(np.linalg.inv(np.dot(np.dot(X.T, w_matrix), X)), X.T), w_matrix),
        y,
    )
    return coef


def linear_regression(X, y, n_replications, seed=None):
    coef_samples = []
    weights = np.random.default_rng(seed).dirichlet([1] * len(X), n_replications)
    for w in weights:
        coef_samples.append(_weighted_ls(X, w, y))
    return np.vstack(coef_samples)


def bayesian_bootstrap(X, statistic, n_replications, resample_size, seed=None):
    """Simulate the posterior distribution of the given statistic."""
    if isinstance(X, list):
        X = np.array(X)

    samples = []
    rng = _get_rng(seed)
    # generate all weights up front for efficiency
    weights = rng.dirichlet(np.ones(len(X)), n_replications)

    for i in range(n_replications):
        choice_rng = _get_rng(seed + i if seed is not None else None)
        sample_index = choice_rng.choice(len(X), p=weights[i], size=resample_size)
        resample_X = X[sample_index]
        s = statistic(resample_X)
        samples.append(s)
    return samples


def bayesian_bootstrap_regression(X, y, statistic, n_replications, resample_size, low_mem=False, seed=None):
    """Simulate the posterior distribution of a statistic that uses dependent and independent variables.

    Parameter X: The observed data, independent variables (matrix like)

    Parameter y: The observed data, dependent variable (array like)

    Parameter statistic: A function of the data to use in simulation (Function mapping array-like to number)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter resample_size: The size of the dataset in each replication

    Parameter low_mem(bool): Use looping instead of generating all the dirichlet, use if program use too much memory

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    samples = []
    X_arr = np.array(X)
    y_arr = np.array(y)
    rng = np.random.default_rng(seed)
    if low_mem:
        weights = (rng.dirichlet([1] * len(X)) for _ in range(n_replications))
    else:
        weights = rng.dirichlet([1] * len(X), n_replications)
    for w in weights:
        if resample_size is None:
            s = statistic(X, y, w)
        else:
            resample_i = rng.choice(range(len(X_arr)), p=w, size=resample_size)
            resample_X = X_arr[resample_i]
            resample_y = y_arr[resample_i]
            s = statistic(resample_X, resample_y)
        samples.append(s)

    return samples


class BayesianBootstrapBagging:
    """A bootstrap aggregating model using the Bayesian bootstrap with parallel processing."""

    def __init__(self, base_learner, n_replications, resample_size=None, n_jobs=-1, seed=None):
        self.base_learner = base_learner
        self.n_replications = n_replications
        self.resample_size = resample_size
        self.n_jobs = n_jobs
        self.seed = seed

    def _fit_single_model(self, X, y, seed):
        rng = _get_rng(seed)
        weights = rng.dirichlet(np.ones(len(X)))
        if self.resample_size is None:
            model = deepcopy(self.base_learner)
            return model.fit(X, y, sample_weight=weights)
        resample_i = rng.choice(len(X), p=weights, size=self.resample_size)
        model = deepcopy(self.base_learner)
        return model.fit(X[resample_i], y[resample_i])

    def fit(self, X, y):
        """Fit the base learners of the ensemble on a dataset."""
        self.base_models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_model)(
                X, y, self.seed + i if self.seed is not None else None
            )
            for i in range(self.n_replications)
        )
        return self

    def predict(self, X):
        """Make average predictions for a collection of observations."""
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.mean(y_posterior_samples, axis=1)

    def predict_posterior_samples(self, X):
        """Simulate posterior samples for a collection of observations."""
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(m.predict)(X) for m in self.base_models_
        )
        return np.array(predictions).T

    def predict_central_interval(self, X, alpha=0.05):
        """The equal-tailed interval prediction containing a (1-alpha) fraction of the posterior samples.

        Parameter X: The observed data, independent variables (matrix like)

        Parameter alpha: The total size of the tails (Float between 0 and 1)

        Returns: Left and right interval bounds for each input (matrix like)
        """
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.array([central_credible_interval(r, alpha=alpha) for r in y_posterior_samples])

    def predict_highest_density_interval(self, X, alpha=0.05):
        """Return highest density intervals for the predictions."""
        y_posterior_samples = self.predict_posterior_samples(X)
        intervals = Parallel(n_jobs=self.n_jobs)(
            delayed(highest_density_interval)(y_posterior_samples[i], alpha=alpha)
            for i in range(len(X))
        )
        return np.array(intervals)


def central_credible_interval(samples, alpha=0.05):
    """The equal-tailed interval containing a (1-alpha) fraction of the posterior samples.

    Parameter samples: The posterior samples (array like)

    Parameter alpha: The total size of the tails (Float between 0 and 1)

    Returns: Left and right interval bounds (tuple)
    """
    return np.quantile(samples, alpha / 2), np.quantile(samples, 1 - alpha / 2)


@numba.njit
def _highest_density_interval_array(samples_arr, alpha=0.05):
    """JIT-compiled helper operating on numpy arrays."""
    samples_sorted = np.sort(samples_arr)
    window_size = int(len(samples_sorted) - round(len(samples_sorted) * alpha))

    if window_size <= 0:
        return samples_sorted[0], samples_sorted[-1]

    min_window_len = np.inf
    best_start = 0
    for i in range(len(samples_sorted) - window_size + 1):
        window_len = samples_sorted[i + window_size - 1] - samples_sorted[i]
        if window_len < min_window_len:
            min_window_len = window_len
            best_start = i
    return samples_sorted[best_start], samples_sorted[best_start + window_size - 1]


def highest_density_interval(samples, alpha=0.05) -> tuple:
    """Return the highest-density interval containing a (1-alpha) fraction of the samples."""
    samples_arr = np.asarray(samples, dtype=np.float64)
    return _highest_density_interval_array(samples_arr, alpha)


def _bootstrap_replicate(X, seed=None):
    random_points = sorted(_get_rng(seed).uniform(0, 1, len(X) - 1))
    random_points.append(1)
    random_points.insert(0, 0)
    gaps = [right - left for left, right in zip(random_points[:-1], random_points[1:])]
    return np.array(gaps)
