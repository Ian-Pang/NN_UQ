"""
Coverage computation relative to Neyman-Pearson optimal classifier.
"""
import numpy as np


class CoverageEvaluator:
    """
    Evaluate coverage of ensemble predictions relative to Neyman-Pearson optimal.

    Coverage measures how often the true optimal prediction falls within
    the uncertainty bands predicted by the ensemble.
    """

    def __init__(self, neyman_pearson_classifier):
        """
        Args:
            neyman_pearson_classifier: NeymanPearsonClassifier instance
        """
        self.np_classifier = neyman_pearson_classifier

    def compute_coverage(self, X, ensemble, confidence_level=0.95):
        """
        Compute coverage: fraction of points where NP prediction falls within ensemble CI.

        Args:
            X: Test data (n_samples, n_features)
            ensemble: MLPEnsemble instance
            confidence_level: Confidence level for coverage (default: 0.95)

        Returns:
            coverage: Fraction of points with NP prediction in ensemble CI
            in_interval: Boolean array indicating coverage for each point
        """
        # Get Neyman-Pearson optimal predictions
        np_proba = self.np_classifier.predict_proba(X)

        # Get ensemble predictions and quantiles
        alpha = 1 - confidence_level
        quantiles = [alpha / 2, 1 - alpha / 2]
        ensemble_quantiles = ensemble.predict_proba_quantiles(X, quantiles)

        lower_bound = ensemble_quantiles[0]
        upper_bound = ensemble_quantiles[1]

        # Check if NP prediction falls within ensemble interval
        in_interval = (np_proba >= lower_bound) & (np_proba <= upper_bound)
        coverage = np.mean(in_interval)

        return coverage, in_interval

    def compute_calibration(self, X, ensemble, n_bins=10):
        """
        Compute calibration: does the predicted uncertainty match actual error?

        Args:
            X: Test data (n_samples, n_features)
            ensemble: MLPEnsemble instance
            n_bins: Number of bins for calibration curve

        Returns:
            predicted_std: Predicted uncertainty (ensemble std) per bin
            actual_error: Actual error (MSE from NP optimal) per bin
        """
        # Get predictions
        np_proba = self.np_classifier.predict_proba(X)
        ensemble_mean = ensemble.predict_proba(X)
        ensemble_std = ensemble.predict_proba_std(X)

        # Compute error
        error = np.abs(ensemble_mean - np_proba)

        # Bin by predicted uncertainty
        std_bins = np.percentile(ensemble_std, np.linspace(0, 100, n_bins + 1))
        predicted_std = []
        actual_error = []

        for i in range(n_bins):
            mask = (ensemble_std >= std_bins[i]) & (ensemble_std < std_bins[i + 1])
            if np.sum(mask) > 0:
                predicted_std.append(np.mean(ensemble_std[mask]))
                actual_error.append(np.mean(error[mask]))

        return np.array(predicted_std), np.array(actual_error)

    def compute_multiple_coverage_levels(self, X, ensemble, confidence_levels=None):
        """
        Compute coverage at multiple confidence levels.

        Args:
            X: Test data
            ensemble: MLPEnsemble instance
            confidence_levels: List of confidence levels to test

        Returns:
            results: Dict mapping confidence_level -> coverage
        """
        if confidence_levels is None:
            confidence_levels = [0.50, 0.68, 0.90, 0.95, 0.99]

        results = {}
        for level in confidence_levels:
            coverage, _ = self.compute_coverage(X, ensemble, level)
            results[level] = coverage

        return results

    def compute_interval_width(self, X, ensemble, confidence_level=0.95):
        """
        Compute average width of prediction intervals.

        Args:
            X: Test data
            ensemble: MLPEnsemble instance
            confidence_level: Confidence level

        Returns:
            mean_width: Average interval width
            widths: Interval widths for each point
        """
        alpha = 1 - confidence_level
        quantiles = [alpha / 2, 1 - alpha / 2]
        ensemble_quantiles = ensemble.predict_proba_quantiles(X, quantiles)

        widths = ensemble_quantiles[1] - ensemble_quantiles[0]
        mean_width = np.mean(widths)

        return mean_width, widths

    def evaluate_ensemble(self, X, ensemble, confidence_level=0.95):
        """
        Comprehensive evaluation of ensemble.

        Args:
            X: Test data
            ensemble: MLPEnsemble instance
            confidence_level: Confidence level for coverage

        Returns:
            results: Dictionary with evaluation metrics
        """
        # Coverage
        coverage, in_interval = self.compute_coverage(X, ensemble, confidence_level)

        # Interval width
        mean_width, widths = self.compute_interval_width(X, ensemble, confidence_level)

        # Prediction error
        np_proba = self.np_classifier.predict_proba(X)
        ensemble_mean = ensemble.predict_proba(X)
        signed_error = ensemble_mean - np_proba  # positive = overestimate
        mae = np.mean(np.abs(signed_error))
        rmse = np.sqrt(np.mean(signed_error ** 2))
        mean_bias = np.mean(signed_error)  # signed mean error (bias)
        std_error = np.std(signed_error)   # spread of errors

        # Ensemble spread
        ensemble_std = ensemble.predict_proba_std(X)
        mean_std = np.mean(ensemble_std)

        results = {
            'coverage': coverage,
            'mean_interval_width': mean_width,
            'mae': mae,
            'rmse': rmse,
            'mean_bias': mean_bias,          # NEW: signed bias
            'std_error': std_error,          # NEW: error spread
            'mean_ensemble_std': mean_std,
            'confidence_level': confidence_level
        }

        return results

    def compare_ensembles(self, X, ensembles, ensemble_names=None, confidence_level=0.95):
        """
        Compare multiple ensembles.

        Args:
            X: Test data
            ensembles: List of MLPEnsemble instances
            ensemble_names: Optional names for ensembles
            confidence_level: Confidence level

        Returns:
            comparison: Dictionary mapping ensemble_name -> metrics
        """
        if ensemble_names is None:
            ensemble_names = [f"Ensemble_{i}" for i in range(len(ensembles))]

        comparison = {}
        for name, ensemble in zip(ensemble_names, ensembles):
            results = self.evaluate_ensemble(X, ensemble, confidence_level)
            comparison[name] = results

        return comparison

    def compute_coverage_across_ensembles(self, X, ensembles, confidence_level=0.95):
        """
        Compute coverage across multiple ensembles.

        For each data point, check what fraction of ensembles have
        confidence intervals that contain the NP optimal prediction.

        Args:
            X: Test data (n_samples, n_features)
            ensembles: List of MLPEnsemble instances
            confidence_level: Confidence level

        Returns:
            coverage: Overall coverage (mean across all points)
            point_coverages: Coverage for each point (n_samples,)
        """
        n_samples = len(X)
        n_ensembles = len(ensembles)

        # Get NP optimal predictions
        np_proba = self.np_classifier.predict_proba(X)

        # For each point, count how many ensembles cover it
        point_coverages = np.zeros(n_samples)

        alpha = 1 - confidence_level
        quantiles = [alpha / 2, 1 - alpha / 2]

        for ensemble in ensembles:
            # Get quantiles for this ensemble
            ensemble_quantiles = ensemble.predict_proba_quantiles(X, quantiles)
            lower_bound = ensemble_quantiles[0]
            upper_bound = ensemble_quantiles[1]

            # Check which points are covered by this ensemble
            in_interval = (np_proba >= lower_bound) & (np_proba <= upper_bound)
            point_coverages += in_interval.astype(float)

        # Normalize by number of ensembles
        point_coverages /= n_ensembles

        # Overall coverage is the mean
        coverage = np.mean(point_coverages)

        return coverage, point_coverages
