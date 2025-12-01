"""
Neyman-Pearson optimal classifier using analytical likelihood ratio.
"""
import numpy as np


class NeymanPearsonClassifier:
    """
    Optimal classifier based on likelihood ratio test.

    For binary classification with known distributions, the Neyman-Pearson
    lemma states that the optimal test statistic is the likelihood ratio:
    LR(x) = p(x|signal) / p(x|background)

    Decision rule: classify as signal if LR(x) > threshold
    """

    def __init__(self, distributions):
        """
        Args:
            distributions: GaussianDistributions object
        """
        self.distributions = distributions

    def predict_proba(self, X, return_lr=False):
        """
        Predict probability that samples are signal.

        Args:
            X: Input features (n_samples, dim)
            return_lr: If True, also return likelihood ratios

        Returns:
            proba: Probability of being signal (n_samples,)
            lr: Likelihood ratios (if return_lr=True)
        """
        lr = self.distributions.likelihood_ratio(X)

        # Convert likelihood ratio to probability using Bayes rule
        # Assuming equal priors: p(signal) = p(background) = 0.5
        # p(signal|x) = LR / (1 + LR)
        proba = lr / (1 + lr)

        if return_lr:
            return proba, lr
        return proba

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Args:
            X: Input features (n_samples, dim)
            threshold: Decision threshold on probability

        Returns:
            y_pred: Predicted labels (n_samples,) - 1 for signal, 0 for background
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

    def get_threshold_at_fpr(self, X_background, target_fpr):
        """
        Find threshold that achieves target false positive rate.

        Args:
            X_background: Background samples for calibration
            target_fpr: Target false positive rate

        Returns:
            threshold: Probability threshold
        """
        proba = self.predict_proba(X_background)
        threshold = np.percentile(proba, 100 * (1 - target_fpr))
        return threshold

    def roc_curve(self, X, y, n_thresholds=100):
        """
        Compute ROC curve.

        Args:
            X: Input features
            y: True labels
            n_thresholds: Number of thresholds to evaluate

        Returns:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Probability thresholds
        """
        proba = self.predict_proba(X)
        thresholds = np.linspace(0, 1, n_thresholds)

        fpr = np.zeros(n_thresholds)
        tpr = np.zeros(n_thresholds)

        for i, thresh in enumerate(thresholds):
            y_pred = (proba > thresh).astype(int)

            # True positives / False positives
            tp = np.sum((y_pred == 1) & (y == 1))
            fp = np.sum((y_pred == 1) & (y == 0))
            tn = np.sum((y_pred == 0) & (y == 0))
            fn = np.sum((y_pred == 0) & (y == 1))

            tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

        return fpr, tpr, thresholds
