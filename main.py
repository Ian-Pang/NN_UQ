"""
Main script for testing MLP ensemble coverage on 10D Gaussian toy model.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from distributions import GaussianDistributions
from neyman_pearson import NeymanPearsonClassifier
from bootstrap import BootstrapSampler
from mlp_classifier import MLPEnsemble
from coverage import CoverageEvaluator


def main():
    """Run the full pipeline."""
    print("=" * 80)
    print("MLP Ensemble Coverage Testing on 10D Gaussian Toy Model")
    print("=" * 80)

    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/global/homes/i/ipang001/NN_UQ/results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # ============================================================================
    # 1. Setup distributions
    # ============================================================================
    print("\n[1] Setting up distributions...")
    dim = 10
    signal_mean = np.zeros(dim)
    signal_mean[0] = 2.0  # Shift signal in first dimension for good separation

    distributions = GaussianDistributions(
        dim=dim,
        signal_mean=signal_mean,
        random_state=seed
    )
    print(f"   - Dimensionality: {dim}")
    print(f"   - Background mean: {distributions.background_mean[:3]}...")
    print(f"   - Signal mean: {distributions.signal_mean[:3]}...")

    # ============================================================================
    # 2. Create Neyman-Pearson optimal classifier
    # ============================================================================
    print("\n[2] Creating Neyman-Pearson optimal classifier...")
    np_classifier = NeymanPearsonClassifier(distributions)
    print("   - Using analytical likelihood ratio")

    # ============================================================================
    # 3. Generate training and test data
    # ============================================================================
    print("\n[3] Generating training and test data...")
    n_train = 10000
    n_test = 5000

    X_train, y_train = distributions.sample_mixed(n_train, signal_fraction=0.5)
    X_test, y_test = distributions.sample_mixed(n_test, signal_fraction=0.5)

    print(f"   - Training samples: {n_train}")
    print(f"   - Test samples: {n_test}")
    print(f"   - Signal fraction: 0.5")

    # ============================================================================
    # 4. Create bootstrap samples for ensemble
    # ============================================================================
    print("\n[4] Creating bootstrap samples...")
    n_estimators = 50  # Number of models in each ensemble (increased from 20)
    n_ensembles = 20   # Number of different ensembles to test

    bootstrap_sampler = BootstrapSampler(random_state=seed)

    ensembles_data = []
    for i in range(n_ensembles):
        print(f"   - Generating bootstrap samples for ensemble {i + 1}/{n_ensembles}")
        bootstrap_samples = bootstrap_sampler.create_stratified_bootstrap_samples(
            X_train, y_train, n_estimators
        )
        ensembles_data.append(bootstrap_samples)

    # ============================================================================
    # 5. Train ensembles
    # ============================================================================
    print("\n[5] Training MLP ensembles...")

    mlp_kwargs = {
        'input_dim': dim,
        'hidden_dims': [64, 32],
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'batch_size': 512,  # Increased for GPU efficiency
        'n_epochs': 30,     # Reduced from 50
        'random_state': seed
    }

    # Check for available GPUs
    import torch
    n_gpus = torch.cuda.device_count()
    n_jobs = min(n_gpus, 4) if n_gpus > 0 else 1  # Use up to 4 GPUs
    print(f"   - Available GPUs: {n_gpus}, using {n_jobs} for parallel training")

    ensembles = []
    for i, bootstrap_samples in enumerate(ensembles_data):
        print(f"\n   Training Ensemble {i + 1}/{n_ensembles}")
        print(f"   - {n_estimators} models with bootstrap resampling")

        ensemble = MLPEnsemble(n_estimators=n_estimators, **mlp_kwargs)
        ensemble.fit(bootstrap_samples, verbose=True, n_jobs=n_jobs)
        ensembles.append(ensemble)

        print(f"   - Ensemble {i + 1} training complete")

    # ============================================================================
    # 6. Evaluate coverage
    # ============================================================================
    print("\n[6] Evaluating coverage relative to Neyman-Pearson optimal...")

    evaluator = CoverageEvaluator(np_classifier)

    ensemble_names = [f"Ensemble_{i + 1}" for i in range(n_ensembles)]
    comparison = evaluator.compare_ensembles(
        X_test,
        ensembles,
        ensemble_names,
        confidence_level=0.95
    )

    print("\n" + "=" * 80)
    print("RESULTS: Ensemble Comparison (95% Confidence Level)")
    print("=" * 80)

    for name, metrics in comparison.items():
        print(f"\n{name}:")
        print(f"  Coverage:              {metrics['coverage']:.4f}")
        print(f"  Mean Interval Width:   {metrics['mean_interval_width']:.4f}")
        print(f"  MAE (vs NP optimal):   {metrics['mae']:.4f}")
        print(f"  RMSE (vs NP optimal):  {metrics['rmse']:.4f}")
        print(f"  Mean Bias (signed):    {metrics['mean_bias']:+.4f}")
        print(f"  Std Error:             {metrics['std_error']:.4f}")
        print(f"  Mean Ensemble Std:     {metrics['mean_ensemble_std']:.4f}")

    # ============================================================================
    # 7. Coverage at multiple confidence levels (mean ± std across ensembles)
    # ============================================================================
    print("\n" + "=" * 80)
    print("Coverage at Multiple Confidence Levels")
    print("=" * 80)

    # Use more confidence levels for smoother plot
    confidence_levels = np.linspace(0.1, 0.99, 50)

    # Compute coverage for EACH ensemble independently at EACH confidence level
    all_coverages = np.zeros((n_ensembles, len(confidence_levels)))

    print(f"\nEvaluating {len(confidence_levels)} confidence levels for {n_ensembles} ensembles...")
    for j, ensemble in enumerate(ensembles):
        for i, level in enumerate(confidence_levels):
            coverage, _ = evaluator.compute_coverage(X_test, ensemble, level)
            all_coverages[j, i] = coverage

    # Compute mean and std across ensembles (for each confidence level)
    mean_coverage = np.mean(all_coverages, axis=0)  # Shape: (50,)
    std_coverage = np.std(all_coverages, axis=0)    # Shape: (50,)

    # Print summary at key confidence levels
    key_levels = [0.50, 0.68, 0.90, 0.95, 0.99]
    print("\nCoverage statistics (mean ± std across ensembles):")
    key_level_coverage = {}
    for level in key_levels:
        idx = np.argmin(np.abs(confidence_levels - level))
        print(f"  {level:.0%} CI: {mean_coverage[idx]:.4f} ± {std_coverage[idx]:.4f}")
        key_level_coverage[f"{level:.0%}"] = {
            "mean": float(mean_coverage[idx]),
            "std": float(std_coverage[idx])
        }

    # ============================================================================
    # Save results for reproducibility
    # ============================================================================
    print("\n[7b] Saving results for reproducibility...")

    # Save experiment configuration
    config = {
        "timestamp": timestamp,
        "seed": seed,
        "dim": dim,
        "signal_mean_shift": float(signal_mean[0]),
        "n_train": n_train,
        "n_test": n_test,
        "n_estimators": n_estimators,
        "n_ensembles": n_ensembles,
        "mlp_kwargs": {k: v for k, v in mlp_kwargs.items() if k != 'random_state'}
    }

    # Compute summary statistics from comparison
    coverage_values = [comparison[name]['coverage'] for name in ensemble_names]
    interval_width_values = [comparison[name]['mean_interval_width'] for name in ensemble_names]
    mae_values = [comparison[name]['mae'] for name in ensemble_names]
    rmse_values = [comparison[name]['rmse'] for name in ensemble_names]
    bias_values = [comparison[name]['mean_bias'] for name in ensemble_names]
    std_error_values = [comparison[name]['std_error'] for name in ensemble_names]
    std_values = [comparison[name]['mean_ensemble_std'] for name in ensemble_names]

    results = {
        "config": config,
        "summary_95ci": {
            "coverage": {"mean": float(np.mean(coverage_values)), "std": float(np.std(coverage_values))},
            "interval_width": {"mean": float(np.mean(interval_width_values)), "std": float(np.std(interval_width_values))},
            "mae": {"mean": float(np.mean(mae_values)), "std": float(np.std(mae_values))},
            "rmse": {"mean": float(np.mean(rmse_values)), "std": float(np.std(rmse_values))},
            "mean_bias": {"mean": float(np.mean(bias_values)), "std": float(np.std(bias_values))},
            "std_error": {"mean": float(np.mean(std_error_values)), "std": float(np.std(std_error_values))},
            "ensemble_std": {"mean": float(np.mean(std_values)), "std": float(np.std(std_values))}
        },
        "coverage_at_key_levels": key_level_coverage,
        "individual_ensembles": {name: {k: float(v) for k, v in metrics.items()}
                                  for name, metrics in comparison.items()}
    }

    # Save JSON results
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   - Saved: {results_file}")

    # Save raw coverage data as numpy arrays for plotting
    np.savez(os.path.join(results_dir, "coverage_data.npz"),
             confidence_levels=confidence_levels,
             all_coverages=all_coverages,
             mean_coverage=mean_coverage,
             std_coverage=std_coverage)
    print(f"   - Saved: {results_dir}/coverage_data.npz")

    # Copy writeup.tex to results directory for reproducibility
    import shutil
    writeup_src = "/global/homes/i/ipang001/NN_UQ/writeup.tex"
    if os.path.exists(writeup_src):
        shutil.copy(writeup_src, os.path.join(results_dir, "writeup.tex"))
        print(f"   - Saved: {results_dir}/writeup.tex")

    # ============================================================================
    # 8. Visualize results
    # ============================================================================
    print("\n[8] Generating visualizations...")

    # Plot 1: Coverage vs Confidence Level with uncertainty band
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(confidence_levels, mean_coverage, 'b-', linewidth=2,
                 label=f'Mean coverage ({n_ensembles} ensembles)')
    axes[0].fill_between(confidence_levels,
                         mean_coverage - std_coverage,
                         mean_coverage + std_coverage,
                         alpha=0.3, color='b', label='±1 std')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    axes[0].set_xlabel('Confidence Level')
    axes[0].set_ylabel('Empirical Coverage')
    axes[0].set_title('Single Ensemble Coverage vs Confidence Level')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # Plot 2: Prediction comparison (NP vs Ensemble) for first ensemble
    np_proba = np_classifier.predict_proba(X_test)
    ensemble_mean = ensembles[0].predict_proba(X_test)

    axes[1].scatter(np_proba, ensemble_mean, alpha=0.3, s=10)
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect agreement')
    axes[1].set_xlabel('Neyman-Pearson Prediction')
    axes[1].set_ylabel('Ensemble Mean Prediction')
    axes[1].set_title(f'{ensemble_names[0]} vs NP Optimal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/global/homes/i/ipang001/NN_UQ/coverage_analysis.png', dpi=150)
    plt.savefig(os.path.join(results_dir, 'coverage_analysis.png'), dpi=150)
    print("   - Saved: coverage_analysis.png")
    print(f"   - Saved: {results_dir}/coverage_analysis.png")

    # Plot 3: Uncertainty vs Error
    fig, ax = plt.subplots(figsize=(8, 6))

    ensemble_std = ensembles[0].predict_proba_std(X_test)
    error = np.abs(ensemble_mean - np_proba)

    ax.scatter(ensemble_std, error, alpha=0.3, s=10)
    ax.set_xlabel('Ensemble Uncertainty (Std)')
    ax.set_ylabel('Prediction Error (|Ensemble - NP|)')
    ax.set_title(f'{ensemble_names[0]}: Uncertainty vs Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/global/homes/i/ipang001/NN_UQ/uncertainty_vs_error.png', dpi=150)
    plt.savefig(os.path.join(results_dir, 'uncertainty_vs_error.png'), dpi=150)
    print("   - Saved: uncertainty_vs_error.png")
    print(f"   - Saved: {results_dir}/uncertainty_vs_error.png")

    # Create a symlink to latest results for easy access
    latest_link = "/global/homes/i/ipang001/NN_UQ/results/latest"
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(results_dir, latest_link)
    print(f"\n   - Created symlink: results/latest -> {timestamp}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
