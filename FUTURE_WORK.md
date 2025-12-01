# Future Work and Experiments

## Priority 1: Improve Coverage

### Experiment 1: Increase Ensemble Size
- **Goal**: Test if larger ensembles improve coverage
- **Action**: Run with 100, 200, or 500 models per ensemble
- **Edit**: `main.py` line 65: `n_estimators = 100`
- **Expected**: Better coverage, but still likely undercovered
- **Runtime**: ~2-3x longer (45-60 minutes with 4 GPUs)

### Experiment 2: Deep Ensembles (Different Initializations)
- **Goal**: Compare bootstrap vs random initialization diversity
- **Action**: Modify `bootstrap.py` to use same data with different seeds
- **Implementation**:
  - Keep full training set, vary only model initialization
  - Set different `random_state` for each model
- **Expected**: Potentially better coverage due to more diverse models

### Experiment 3: Post-hoc Calibration
- **Goal**: Apply temperature scaling to improve calibration
- **Action**: Add calibration step in `mlp_classifier.py`
- **Method**:
  1. Hold out validation set from training
  2. Learn temperature parameter T to minimize NLL
  3. Scale logits by T before softmax
- **Expected**: Improved coverage toward nominal level

### Experiment 4: Conformal Prediction
- **Goal**: Obtain guaranteed coverage
- **Action**: Implement conformal prediction wrapper
- **Method**:
  1. Use ensemble predictions as base scores
  2. Calibrate on held-out set
  3. Compute conformal quantiles
- **Expected**: Nominal coverage by construction

## Priority 2: Different Architectures

### Experiment 5: MC Dropout
- **Goal**: Test alternative uncertainty quantification method
- **Action**:
  - Keep dropout active during inference
  - Sample predictions 50 times per model
- **Edit**: `mlp_classifier.py` - add `mc_dropout` mode in `predict_proba`
- **Expected**: May improve coverage, faster than full ensemble

### Experiment 6: Vary Network Capacity
- **Goal**: Test if model size affects calibration
- **Variations**:
  - Smaller: [32, 16]
  - Current: [64, 32]
  - Larger: [128, 64, 32]
  - Very large: [256, 128, 64]
- **Expected**: Larger models may be more overconfident

### Experiment 7: Bayesian Neural Networks
- **Goal**: Compare to proper Bayesian uncertainty
- **Action**: Implement variational inference or Laplace approximation
- **Libraries**: Use PyTorch BNN libraries (e.g., Blitz)
- **Expected**: Better calibrated posteriors

## Priority 3: Dataset Variations

### Experiment 8: Increase Training Data
- **Goal**: Test if finite data is the issue
- **Variations**:
  - Current: 10,000
  - Test: 50,000, 100,000, 500,000
- **Edit**: `main.py` line 51: `n_train = 50000`
- **Expected**: Improved coverage with more data

### Experiment 9: Vary Signal Separation
- **Goal**: Test across difficulty levels
- **Variations**:
  - Easy: shift = 3.0 (current: 2.0)
  - Hard: shift = 1.0
  - Very hard: shift = 0.5
- **Edit**: `main.py` line 29: `signal_mean[0] = 1.0`
- **Expected**: Coverage may depend on task difficulty

### Experiment 10: Higher Dimensional Space
- **Goal**: Test scalability and curse of dimensionality
- **Variations**:
  - Current: 10D
  - Test: 20D, 50D, 100D
- **Edit**: `main.py` line 27: `dim = 20`
- **Note**: May need to adjust architecture and training data size

## Priority 4: Analysis and Diagnostics

### Analysis 1: Investigate Undercoverage Pattern
- **Goal**: Understand where/why coverage fails
- **Actions**:
  - Plot coverage vs true probability p*(x)
  - Analyze coverage in signal vs background regions
  - Check if coverage worse at decision boundary
- **New file**: `diagnosis.py`

### Analysis 2: Ensemble Diversity Metrics
- **Goal**: Quantify model disagreement
- **Metrics**:
  - Pairwise correlation between models
  - Effective ensemble size
  - Diversity-accuracy decomposition
- **Expected insight**: Low diversity → poor uncertainty

### Analysis 3: Compare to Frequentist Bootstrap Confidence Intervals
- **Goal**: Separate NN issues from bootstrap issues
- **Action**: Apply bootstrap to simple logistic regression
- **Expected**: If logistic regression calibrates well, issue is with NNs

### Analysis 4: Prediction Interval Diagnostics
- **Actions**:
  - Plot interval width distribution
  - Check for adaptive vs constant width
  - Examine intervals that miss vs contain
- **Goal**: Understand if intervals too narrow everywhere or selectively

## Quick Wins

### Quick 1: Save Trained Models
```python
# Add to main.py after training
import pickle
with open('trained_ensembles.pkl', 'wb') as f:
    pickle.dump(ensembles, f)
```
- **Benefit**: Avoid retraining for analysis experiments

### Quick 2: Add Command Line Arguments
```python
# Add to main.py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_ensembles', type=int, default=20)
parser.add_argument('--n_estimators', type=int, default=50)
parser.add_argument('--n_train', type=int, default=10000)
args = parser.parse_args()
```
- **Benefit**: Easy parameter sweeps

### Quick 3: Add Learning Curves
- Plot coverage vs number of models in ensemble (1, 2, 5, 10, 20, 50)
- **Action**: Modify `coverage.py` to take subset of models
- **Benefit**: Understand convergence

### Quick 4: Log Results to CSV
```python
# Save results for later analysis
import pandas as pd
results_df = pd.DataFrame(comparison).T
results_df.to_csv('ensemble_results.csv')
```
- **Benefit**: Track results across experiments

## Code Quality Improvements

### Refactor 1: Configuration File
- Move all hyperparameters to `config.yaml`
- Use `yaml` or `hydra` for configuration management
- **Benefit**: Cleaner code, easier experiments

### Refactor 2: Add Unit Tests
- Test bootstrap sampling preserves class balance
- Test Neyman-Pearson classifier matches analytical solution
- Test coverage computation
- **Files**: Create `tests/` directory

### Refactor 3: Add Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```
- Replace `print()` with `logger.info()`
- **Benefit**: Better control over output

### Refactor 4: Modularize Plotting
- Move plotting code from `main.py` to `visualization.py`
- **Benefit**: Cleaner main script, reusable plots

## Documentation

### Doc 1: Add Docstrings
- Ensure all functions have NumPy-style docstrings
- Run `pdoc` to generate API documentation

### Doc 2: Jupyter Notebook Tutorial
- Create `tutorial.ipynb` showing:
  - Basic usage
  - How to interpret results
  - How to modify experiments
- **Benefit**: Easier for others to use

### Doc 3: Update writeup.tex
- Add section on parallelization
- Include new results with 50 models
- Add lessons learned section

## Long-term Research Directions

### Research 1: Real-world Dataset
- Move beyond toy problem to actual physics/ML dataset
- Test if findings generalize

### Research 2: Compare to Other UQ Methods
- Evidential deep learning
- Gaussian processes
- Quantile regression

### Research 3: Theoretical Analysis
- Derive analytical prediction for bootstrap ensemble coverage
- Understand when/why bootstrap fails for neural networks

### Research 4: Publication
- Write up results for ML or physics conference
- Focus on: "When do bootstrap ensembles fail for UQ?"
