import numpy as np
from copy import deepcopy


def mean(X, n_replications, seed=None):
    """Simulate the posterior distribution of the mean.

    Parameter X: The observed data (array like)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    weights = np.random.default_rng(seed).dirichlet(np.ones(len(X)), n_replications)
    return np.dot(X, weights.T)


def var(X, n_replications, seed=None) -> np.ndarray:
    """Simulate the posterior distribution of the variance. (Vectorized for speed)"""
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T
    X_squared = np.square(X)
    weighted_X_squared = X_squared @ weights
    weighted_X = X @ weights
    return weighted_X_squared - np.square(weighted_X)


def _weighted_covariance(X, Y, weights: np.ndarray) -> np.ndarray:
    """Helper for vectorized covariance."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    weighted_X = X @ weights
    weighted_Y = Y @ weights
    weighted_XY = (X * Y) @ weights
    return weighted_XY - (weighted_X * weighted_Y)


def covar(X, Y, n_replications, seed=None) -> np.ndarray:
    """Simulate the posterior distribution of the covariance. (Vectorized for speed)"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T
    return _weighted_covariance(X, Y, weights)


def pearsonr(X, Y, n_replications, seed=None) -> np.ndarray:
    """Simulate the posterior of the Pearson correlation. (Vectorized for speed)."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(X)), n_replications).T
    cov_XY = _weighted_covariance(X, Y, weights)
    var_X = _weighted_covariance(X, X, weights)
    var_Y = _weighted_covariance(Y, Y, weights)
    std_devs = np.sqrt(var_X * var_Y)
    return np.divide(cov_XY, std_devs, out=np.full_like(cov_XY, np.nan), where=std_devs!=0)


def _weighted_ls(X, y, w):
    """Solves weighted least squares problem using a more stable method."""
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
    """Simulate the posterior distribution of the given statistic.

    Deprecated in favour of :func:`bayesian_bootstrap` which uses direct weighting.

    Parameter X: The observed data (array like)
    Parameter statistic: A function of the data to use in simulation (Function mapping array-like to number)
    Parameter n_replications: The number of bootstrap replications to perform (positive integer)
    Parameter resample_size: The size of the dataset in each replication
    Parameter low_mem(bool): Generate the weights for each iteration lazily instead of in a single batch. Will use
        less memory, but will run slower as a result.
    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    import warnings
    warnings.warn(
        "`bayesian_bootstrap_resample` is deprecated as it adds unnecessary variance. "
        "Use the direct-weighting `bayesian_bootstrap` function instead where possible.",
        DeprecationWarning
    )

    if isinstance(X, list):
        X = np.array(X)
    samples = []
    rng = np.random.default_rng(seed)
    if low_mem:
        weights = (rng.dirichlet([1] * len(X)) for _ in range(n_replications))
    else:
        weights = rng.dirichlet([1] * len(X), n_replications)
    for w in weights:
        sample_index = rng.choice(range(len(X)), p=w, size=resample_size)
        resample_X = X[sample_index]
        s = statistic(resample_X)
        samples.append(s)
    return samples


def bayesian_bootstrap(X, statistic, n_replications, alpha=1.0, seed=None, statistic_args=None):
    """
    Simulate the posterior distribution of a statistic using direct weighting.

    Parameter X: The observed data (array-like).
    Parameter statistic: A function that takes X and weights (and optional args)
                       and returns the statistic.
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
    """Simulate the posterior distribution of a statistic that uses dependent and independent variables.

    .. deprecated:: 2.0
       Use :class:`BayesianBootstrapBagging` instead.
    
    Parameter X: The observed data, independent variables (matrix like)

    Parameter y: The observed data, dependent variable (array like)

    Parameter statistic: A function of the data to use in simulation (Function mapping array-like to number)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter resample_size: The size of the dataset in each replication

    Parameter low_mem(bool): Use looping instead of generating all the dirichlet, use if program use too much memory

    Parameter seed: Seed for PRNG (default None)

    Returns: Samples from the posterior
    """
    import warnings
    warnings.warn(
        "`bayesian_bootstrap_regression` is deprecated. Use the `BayesianBootstrapBagging` class instead.",
        DeprecationWarning
    )
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
    """A bootstrap aggregating model using the Bayesian bootstrap."""

    def __init__(self, base_learner, n_replications, resample_size=None, n_jobs=-1, seed=None):
        """Initialize the ensemble.

        Parameter base_learner: A scikit-learn like estimator implementing ``fit`` and ``predict``.
        Parameter n_replications: The number of bootstrap replications.
        Parameter resample_size: Optional size of each bootstrap sample. If ``None`` use direct weighting.
        Parameter n_jobs: Number of jobs for parallelism with ``joblib``. ``-1`` uses all cores.
        Parameter seed: Optional base random seed.
        """
        self.base_learner = base_learner
        self.n_replications = n_replications
        self.resample_size = resample_size
        self.n_jobs = n_jobs
        self.seed = seed

    def _fit_single_model(self, X, y, seed):
        """Helper to fit one bootstrapped model."""
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
        """Make average predictions for a collection of observations.

        Parameter X: The observed data, independent variables (matrix like)

        Returns: The predicted dependent variable values (array like)
        """
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.array([np.mean(r) for r in y_posterior_samples])

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
        """The highest density interval prediction containing a (1-alpha) fraction of the posterior samples.

        Parameter X: The observed data, independent variables (matrix like)

        Parameter alpha: The total size of the tails (Float between 0 and 1)

        Returns: Left and right interval bounds for each input (matrix like):
        """
        y_posterior_samples = self.predict_posterior_samples(X)
        return np.array([highest_density_interval(r, alpha=alpha) for r in y_posterior_samples])


def central_credible_interval(samples, alpha=0.05):
    """The equal-tailed interval containing a (1-alpha) fraction of the posterior samples.

    Parameter samples: The posterior samples (array like)

    Parameter alpha: The total size of the tails (Float between 0 and 1)

    Returns: Left and right interval bounds (tuple)
    """
    return np.quantile(samples, alpha / 2), np.quantile(samples, 1 - alpha / 2)


def highest_density_interval(samples, alpha=0.05) -> tuple:
    """The highest-density interval containing a (1-alpha) fraction of the posterior samples.

    Parameter samples: The posterior samples (array like)

    Parameter alpha: The total size of the tails (Float between 0 and 1)

    Returns: Left and right interval bounds (tuple)
    """
    samples_sorted = sorted(samples)
    window_size = int(len(samples) - round(len(samples) * alpha))
    smallest_window = (None, None)
    smallest_window_length = float("inf")
    for i in range(len(samples_sorted) - window_size + 1):
        window = samples_sorted[i + window_size - 1], samples_sorted[i]
        window_length = samples_sorted[i + window_size - 1] - samples_sorted[i]
        if window_length < smallest_window_length:
            smallest_window_length = window_length
            smallest_window = window
    return smallest_window[1], smallest_window[0]


def _bootstrap_replicate(X, seed=None):
    random_points = sorted(np.random.default_rng(seed).uniform(0, 1, len(X) - 1))
    random_points.append(1)
    random_points.insert(0, 0)
    gaps = [right - left for left, right in zip(random_points[:-1], random_points[1:])]
    return np.array(gaps)
