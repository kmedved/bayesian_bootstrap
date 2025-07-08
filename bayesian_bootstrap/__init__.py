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


def var(X, n_replications, seed=None) -> np.ndarray:
    """Simulate the posterior distribution of the variance. (Vectorized for speed)"""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T  # Shape (n_samples, n_replications)
    X = np.asarray(X)
    X_squared = np.square(X)
    weighted_X_squared = X_squared @ weights
    weighted_X = X @ weights
    return weighted_X_squared - np.square(weighted_X)


def _weighted_covariance(X, Y, weights: np.ndarray) -> np.ndarray:
    """Helper for vectorized covariance."""
    weighted_X = X @ weights
    weighted_Y = Y @ weights
    weighted_XY = (X * Y) @ weights
    return weighted_XY - (weighted_X * weighted_Y)


def covar(X, Y, n_replications, seed=None) -> np.ndarray:
    """Simulate the posterior distribution of the covariance. (Vectorized for speed)"""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T
    X = np.asarray(X)
    Y = np.asarray(Y)
    return _weighted_covariance(X, Y, weights)


def pearsonr(X, Y, n_replications, seed=None) -> np.ndarray:
    """Simulate the posterior of the Pearson correlation. (Vectorized for speed)."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T
    X = np.asarray(X)
    Y = np.asarray(Y)
    cov_XY = _weighted_covariance(X, Y, weights)
    var_X = _weighted_covariance(X, X, weights)
    var_Y = _weighted_covariance(Y, Y, weights)
    std_devs = np.sqrt(var_X * var_Y)
    return np.divide(cov_XY, std_devs, out=np.full_like(cov_XY, np.nan), where=std_devs!=0)


def _weighted_covariance(X, Y, weights):
    """Return weighted covariance for multiple weight sets."""
    # weights shape: (n_samples, n_replications)
    weighted_X = X @ weights
    weighted_Y = Y @ weights
    weighted_XY = (X * Y) @ weights
    return weighted_XY - (weighted_X * weighted_Y)

def _weighted_ls(X, y, w):
    """
    Solves weighted least squares problem using a more stable method.
    w is a single weight vector for one replication.
    """
    sqrt_w = np.sqrt(w)
    X_w = X * sqrt_w[:, np.newaxis]
    y_w = y * sqrt_w
    coeffs, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
    return coeffs


def linear_regression(X, y, n_replications, seed=None, n_jobs=-1):
    """Simulate the posterior of linear regression coefficients in parallel."""
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications)

    coef_samples = Parallel(n_jobs=n_jobs)(
        delayed(_weighted_ls)(X, y, w) for w in weights
    )
    return np.vstack(coef_samples)


def bayesian_bootstrap_resample(X, statistic, n_replications, resample_size, low_mem=False, seed=None):
    import warnings
    warnings.warn(
        "`bayesian_bootstrap_resample` is deprecated as it adds unnecessary variance. "
        "Use the direct-weighting `bayesian_bootstrap` function instead where possible.",
        DeprecationWarning
    )
    if isinstance(X, list):
        X = np.array(X)

    samples = []
    rng = _get_rng(seed)
    weights_iter = rng.dirichlet(np.ones(len(X)), n_replications)
    if low_mem:
        weights_iter = (rng.dirichlet(np.ones(len(X))) for _ in range(n_replications))

    for i, w in enumerate(weights_iter):
        choice_rng = _get_rng(seed + i if seed is not None else None)
        sample_index = choice_rng.choice(len(X), p=w, size=resample_size)
        resample_X = X[sample_index]
        s = statistic(resample_X)
        samples.append(s)
    return samples


def bayesian_bootstrap(X, statistic, n_replications, alpha=1.0, seed=None, statistic_args=None):
    """
    Simulate the posterior distribution of a statistic using direct weighting.

    Parameter X: The observed data (array-like).
    Parameter statistic: A function that takes X and weights (and optional args)
                       and returns the statistic. e.g., `np.average`.
    Parameter n_replications: The number of bootstrap replications.
    Parameter alpha: The concentration parameter for the Dirichlet distribution.
                     Default is 1.0 (non-informative). Can be a scalar or a vector.
    Parameter seed: Seed for the random number generator.
    Parameter statistic_args: A dictionary of additional keyword arguments for the statistic function.

    Returns: Samples from the posterior.
    """
    rng = np.random.default_rng(seed)
    if isinstance(alpha, (int, float)):
        alpha_prior = np.repeat(alpha, len(X))
    else:
        alpha_prior = np.asarray(alpha)

    weights = rng.dirichlet(alpha_prior, n_replications)

    if statistic_args is None:
        statistic_args = {}

    return statistic(X, weights=weights.T, **statistic_args)


def bayesian_bootstrap_regression(X, y, statistic, n_replications, resample_size, low_mem=False, seed=None):
    import warnings
    warnings.warn(
        "`bayesian_bootstrap_regression` is deprecated. Use the `BayesianBootstrapBagging` class instead.",
        DeprecationWarning
    )
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
        """Helper function to fit one bootstrapped model."""
        rng = np.random.default_rng(seed)
        weights = rng.dirichlet(np.ones(len(X)))
        model = deepcopy(self.base_learner)

        import inspect
        fit_params = inspect.signature(model.fit).parameters

        if self.resample_size is None and 'sample_weight' in fit_params:
            return model.fit(X, y, sample_weight=weights)
        else:
            resample_i = rng.choice(len(X), p=weights, size=self.resample_size or len(X))
            return model.fit(X[resample_i], y[resample_i])

    def fit(self, X, y):
        """Fit the ensemble in parallel."""
        from joblib import Parallel, delayed
        seeds = (self.seed + i if self.seed is not None else None for i in range(self.n_replications))
        self.base_models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_model)(X, y, seed=s) for s in seeds
        )
        return self

    def predict(self, X):
        """Make average predictions for a collection of observations."""
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.mean(y_posterior_samples, axis=1)

    def predict_posterior_samples(self, X):
        """Simulate posterior samples in parallel."""
        from joblib import Parallel, delayed
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
