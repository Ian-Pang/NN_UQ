# Experiment Log

This file tracks all experiments run and their results.

## Current Baseline Configuration

### Data
- **Dimensions**: 10D
- **Training samples**: 10,000 (5,000 signal, 5,000 background)
- **Test samples**: 5,000 (2,500 signal, 2,500 background)
- **Signal mean**: (2, 0, 0, ..., 0)
- **Background mean**: (0, 0, 0, ..., 0)
- **Covariance**: Identity matrix

### Model Architecture
- **Type**: Feedforward MLP
- **Hidden layers**: [64, 32]
- **Activation**: ReLU
- **Dropout**: 0.1
- **Output**: Sigmoid (binary classification)
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam
- **Learning rate**: 0.001
- **Batch size**: 512
- **Epochs**: 30

### Ensemble Configuration
- **Number of ensembles**: 20
- **Models per ensemble**: 50
- **Total models**: 1,000
- **Sampling method**: Stratified bootstrap (with replacement)
- **Random seed**: 42

### Computational Setup
- **Platform**: NERSC Perlmutter
- **GPUs**: 4x NVIDIA A100
- **Parallelization**: Multi-GPU with round-robin assignment
- **Training time**: ~23 minutes
- **Framework**: PyTorch 2.8.0

---

## Experiment History

### Experiment 1: Initial Baseline (Original)
**Date**: Before parallelization (referenced in writeup.tex)

**Configuration**:
- Models per ensemble: 20
- All other parameters: Same as current baseline

**Results**:
- Coverage at 95% CI: ~69.5%
- Undercoverage gap: ~25.5%

**Notes**:
- Sequential training only
- No progress tracking

---

### Experiment 2: Increased Ensemble Size + Parallelization
**Date**: 2025-11-20

**Changes from Baseline**:
- Models per ensemble: 20 → 50
- Added multi-GPU parallelization (4 GPUs)
- Added verbose progress tracking

**Results**:
- Coverage at 95% CI: ~74.3% (range: 73.1% - 75.3%)
- Undercoverage gap: ~20.7%
- Mean Interval Width: 0.094-0.097
- MAE vs NP optimal: 0.025-0.026
- RMSE vs NP optimal: 0.037-0.039
- Mean Ensemble Std: 0.026-0.027

**Detailed Results by Ensemble**:
```
Ensemble  | Coverage | Int Width | MAE    | RMSE   | Std
----------|----------|-----------|--------|--------|--------
1         | 0.7378   | 0.0958    | 0.0257 | 0.0381 | 0.0263
2         | 0.7394   | 0.0951    | 0.0259 | 0.0386 | 0.0261
3         | 0.7526   | 0.0970    | 0.0252 | 0.0374 | 0.0266
4         | 0.7462   | 0.0967    | 0.0257 | 0.0382 | 0.0265
5         | 0.7468   | 0.0966    | 0.0256 | 0.0378 | 0.0265
6         | 0.7406   | 0.0963    | 0.0259 | 0.0386 | 0.0264
7         | 0.7490   | 0.0961    | 0.0256 | 0.0379 | 0.0263
8         | 0.7310   | 0.0944    | 0.0259 | 0.0385 | 0.0258
9         | 0.7452   | 0.0955    | 0.0256 | 0.0381 | 0.0262
10        | 0.7360   | 0.0962    | 0.0260 | 0.0384 | 0.0263
11        | 0.7430   | 0.0957    | 0.0255 | 0.0378 | 0.0262
12        | 0.7438   | 0.0939    | 0.0252 | 0.0376 | 0.0258
13        | 0.7452   | 0.0970    | 0.0258 | 0.0382 | 0.0265
14        | 0.7414   | 0.0957    | 0.0253 | 0.0373 | 0.0262
15        | 0.7454   | 0.0969    | 0.0259 | 0.0384 | 0.0265
16        | 0.7390   | 0.0962    | 0.0257 | 0.0383 | 0.0263
17        | 0.7508   | 0.0959    | 0.0256 | 0.0382 | 0.0262
18        | 0.7422   | 0.0965    | 0.0261 | 0.0389 | 0.0265
19        | 0.7442   | 0.0962    | 0.0258 | 0.0382 | 0.0264
20        | 0.7438   | 0.0968    | 0.0256 | 0.0379 | 0.0265

Mean      | 0.7432   | 0.0960    | 0.0257 | 0.0381 | 0.0263
Std       | 0.0053   | 0.0009    | 0.0002 | 0.0004 | 0.0002
```

**Observations**:
1. Improvement over 20-model ensembles: 69.5% → 74.3% (+4.8 pp)
2. Still substantial undercoverage: ~21% gap from target **for each individual ensemble**
3. Very consistent across ensembles (std = 0.53%) - indicates **systematic** problem, not random variation
4. Good prediction accuracy (MAE ≈ 0.026) - models predict well but are overconfident
5. Narrow intervals suggest overconfidence - each ensemble's intervals are too narrow

**Outputs Generated**:
- `coverage_analysis.png` (257 KB) - Coverage vs confidence level
- `uncertainty_vs_error.png` (198 KB) - Uncertainty-error relationship

**Performance**:
- Total runtime: ~23 minutes
- Per ensemble: ~1.1 minutes
- Speedup: ~3.5-4x vs sequential

**Conclusion**:
Increasing ensemble size helps but doesn't solve fundamental undercoverage issue. **Each individual** bootstrap ensemble systematically underestimates uncertainty in this setting. The fact that all 20 independent ensembles show nearly identical undercoverage (std=0.5%) proves this is a methodological problem, not due to particular bootstrap samples.

---

### Methodology Note (2025-11-20)
**Coverage Computation**: Updated to properly evaluate **single ensemble** coverage:
- OLD: Averaged over both test points AND ensembles simultaneously
- NEW: Compute coverage for EACH ensemble separately, then report mean ± std
- **Interpretation**: "A single ensemble achieves X% coverage, with ±Y% variation across different bootstrap samples"
- **Why this matters**: In practice, you deploy ONE ensemble, not an average over many ensembles
- The small std (0.5%) shows the undercoverage is systematic to the method, not random

**Code Changes**:
- `main.py`: Updated Section 7 to compute per-ensemble coverage, report mean ± std, plot with error bands
- `writeup.tex`: Updated abstract, methodology, results, and conclusion to reflect single ensemble interpretation
- **Note**: `writeup.pdf` needs to be regenerated manually with `pdflatex writeup.tex` (not available on Perlmutter)

---

## Future Experiments (Planned)

### Next: Experiment 3 - Even Larger Ensembles
**Goal**: Test if 100 models per ensemble improves coverage further

**Changes**:
- `main.py` line 65: `n_estimators = 100`

**Expected runtime**: ~45 minutes with 4 GPUs

**Hypothesis**: Coverage will improve toward 80-85%, but still undercovered

---

### Next: Experiment 4 - Deep Ensembles
**Goal**: Compare bootstrap sampling vs different initializations

**Changes**:
- Use full training set (no bootstrap)
- Vary `random_state` for each model
- `mlp_kwargs['random_state'] = seed + model_idx`

**Hypothesis**: May achieve better coverage due to more diverse models

---

## Quick Reference: Parameter Locations

Key parameters and where to change them:

| Parameter | File | Line | Default Value |
|-----------|------|------|---------------|
| Dimensions | `main.py` | 27 | 10 |
| Signal shift | `main.py` | 29 | 2.0 |
| Training samples | `main.py` | 51 | 10000 |
| Test samples | `main.py` | 52 | 5000 |
| Number of ensembles | `main.py` | 66 | 20 |
| Models per ensemble | `main.py` | 65 | 50 |
| Hidden layers | `main.py` | 85 | [64, 32] |
| Dropout rate | `main.py` | 86 | 0.1 |
| Learning rate | `main.py` | 87 | 0.001 |
| Batch size | `main.py` | 88 | 512 |
| Epochs | `main.py` | 89 | 30 |
| Random seed | `main.py` | 21 | 42 |
| Confidence level | `main.py` | 116 | 0.95 |
| Number of GPUs | `main.py` | 96 | min(n_gpus, 4) |

---

## Notes

- Always set `verbose=True` in `ensemble.fit()` for progress tracking
- Results are reproducible with fixed random seed (42)
- Coverage evaluation uses quantile-based confidence intervals
- Ground truth is Neyman-Pearson analytical optimal classifier
- All metrics computed on held-out test set

---

## Code Verification and Updates (2025-11-20)

**Status**: ✅ All code and documentation updated and verified

### Changes Implemented:
1. **main.py Section 7** (lines 137-165): Modified to compute per-ensemble coverage
   - Coverage computed for each of 20 ensembles independently
   - Report mean ± std across ensembles
   - Plots now show error bands representing variability

2. **Documentation Updates**:
   - README.md: Updated interpretation section
   - EXPERIMENT_LOG.md: Added methodology note
   - PARALLELIZATION_NOTES.md: Updated results format
   - writeup.tex: Comprehensive updates throughout

3. **Code Execution**:
   - Successfully ran on 4 GPUs (NERSC Perlmutter)
   - Runtime: ~23 minutes
   - Generated updated plots with error bands

### Important Notes:
- **PDF Regeneration**: `writeup.pdf` needs manual regeneration with `pdflatex writeup.tex` on a machine with LaTeX installed (not available on Perlmutter)
- **Interpretation**: Results now correctly answer "How well does a SINGLE ensemble provide coverage?"
- **Small std (0.5%)**: Proves undercoverage is systematic methodological problem, not random variation
