"""
Bootstrap resampling utilities for training data.
"""
import numpy as np


class BootstrapSampler:
    """
    Bootstrap resampling for capturing finite data uncertainty.

    Each bootstrap sample is drawn with replacement from the original dataset,
    allowing us to train multiple models that capture both training uncertainty
    and finite data uncertainty.
    """

    def __init__(self, random_state=None):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_state)

    def resample(self, X, y):
        """
        Create a bootstrap sample by sampling with replacement.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            X_boot: Bootstrap resampled features
            y_boot: Bootstrap resampled labels
        """
        n_samples = len(X)
        indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def create_bootstrap_samples(self, X, y, n_bootstrap):
        """
        Create multiple bootstrap samples.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            n_bootstrap: Number of bootstrap samples to create

        Returns:
            List of (X_boot, y_boot) tuples
        """
        samples = []
        for _ in range(n_bootstrap):
            X_boot, y_boot = self.resample(X, y)
            samples.append((X_boot, y_boot))
        return samples

    def stratified_resample(self, X, y):
        """
        Create a stratified bootstrap sample that preserves class balance.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            X_boot: Bootstrap resampled features
            y_boot: Bootstrap resampled labels
        """
        unique_classes = np.unique(y)
        X_boot_list = []
        y_boot_list = []

        for cls in unique_classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]

            n_samples_cls = len(X_cls)
            indices = self.rng.choice(n_samples_cls, size=n_samples_cls, replace=True)

            X_boot_list.append(X_cls[indices])
            y_boot_list.append(y_cls[indices])

        X_boot = np.vstack(X_boot_list)
        y_boot = np.concatenate(y_boot_list)

        # Shuffle
        indices = self.rng.permutation(len(X_boot))
        return X_boot[indices], y_boot[indices]

    def create_stratified_bootstrap_samples(self, X, y, n_bootstrap):
        """
        Create multiple stratified bootstrap samples.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            n_bootstrap: Number of bootstrap samples to create

        Returns:
            List of (X_boot, y_boot) tuples
        """
        samples = []
        for _ in range(n_bootstrap):
            X_boot, y_boot = self.stratified_resample(X, y)
            samples.append((X_boot, y_boot))
        return samples
