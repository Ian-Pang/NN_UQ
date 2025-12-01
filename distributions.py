"""
Signal and background distributions for the toy model.
"""
import numpy as np
from scipy.stats import multivariate_normal


class GaussianDistributions:
    """
    10D Gaussian distributions for signal and background.
    Background: mean = 0, std = 1
    Signal: mean = signal_mean (user specified), std = 1
    """

    def __init__(self, dim=10, signal_mean=None, random_state=None):
        """
        Args:
            dim: Dimensionality of the problem (default: 10)
            signal_mean: Mean vector for signal distribution. If None, defaults to [1, 0, 0, ...]
            random_state: Random seed for reproducibility
        """
        self.dim = dim
        self.rng = np.random.RandomState(random_state)

        # Background: centered at origin
        self.background_mean = np.zeros(dim)
        self.background_cov = np.eye(dim)

        # Signal: shifted from background
        if signal_mean is None:
            signal_mean = np.zeros(dim)
            signal_mean[0] = 1.0  # Shift only in first dimension
        self.signal_mean = np.array(signal_mean)
        self.signal_cov = np.eye(dim)

        # Create scipy distributions for PDF computation
        self.background_dist = multivariate_normal(
            mean=self.background_mean,
            cov=self.background_cov
        )
        self.signal_dist = multivariate_normal(
            mean=self.signal_mean,
            cov=self.signal_cov
        )

    def sample_background(self, n_samples):
        """Sample from background distribution."""
        return self.rng.multivariate_normal(
            self.background_mean,
            self.background_cov,
            size=n_samples
        )

    def sample_signal(self, n_samples):
        """Sample from signal distribution."""
        return self.rng.multivariate_normal(
            self.signal_mean,
            self.signal_cov,
            size=n_samples
        )

    def sample_mixed(self, n_samples, signal_fraction=0.5):
        """
        Sample mixed signal and background.

        Args:
            n_samples: Total number of samples
            signal_fraction: Fraction of signal samples

        Returns:
            X: Features (n_samples, dim)
            y: Labels (n_samples,) - 1 for signal, 0 for background
        """
        n_signal = int(n_samples * signal_fraction)
        n_background = n_samples - n_signal

        X_signal = self.sample_signal(n_signal)
        X_background = self.sample_background(n_background)

        X = np.vstack([X_signal, X_background])
        y = np.concatenate([np.ones(n_signal), np.zeros(n_background)])

        # Shuffle
        indices = self.rng.permutation(n_samples)
        return X[indices], y[indices]

    def background_pdf(self, X):
        """Compute background PDF for given points."""
        return self.background_dist.pdf(X)

    def signal_pdf(self, X):
        """Compute signal PDF for given points."""
        return self.signal_dist.pdf(X)

    def likelihood_ratio(self, X):
        """
        Compute likelihood ratio: p(x|signal) / p(x|background)

        This is the Neyman-Pearson optimal test statistic.
        """
        return self.signal_pdf(X) / self.background_pdf(X)
