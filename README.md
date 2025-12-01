# Neural Network Uncertainty Quantification

Testing coverage of ensembles of MLP-based classifiers for a 10D Gaussian toy model against the Neyman-Pearson optimal classifier.

## Project Overview

This project evaluates whether bootstrap-based neural network ensembles produce well-calibrated uncertainty estimates. Using a 10D Gaussian toy problem where the optimal classifier (Neyman-Pearson) is analytically known, we test if ensemble prediction intervals achieve their nominal coverage levels.

### Key Question
Do 95% confidence intervals from a single bootstrap ensemble actually contain the true optimal prediction 95% of the time?

### Key Finding
**No** - a single bootstrap ensemble shows systematic undercoverage. With 50 models per ensemble:
- Target: 95% coverage
- Actual: ~74% coverage (±0.5% across 20 independent ensembles)
- Gap: ~21 percentage points of undercoverage

This indicates that **a single bootstrap ensemble underestimates prediction uncertainty** in this setting. The small variation (±0.5%) across different bootstrap samples shows this is a systematic problem with the method, not random noise.

## Project Structure

```
NN_UQ/
├── main.py                      # Main execution script
├── distributions.py             # 10D Gaussian signal/background generators
├── neyman_pearson.py           # Analytical optimal classifier (ground truth)
├── bootstrap.py                # Stratified bootstrap resampling
├── mlp_classifier.py           # MLP classifier and ensemble implementation
├── coverage.py                 # Coverage evaluation metrics
├── run_job.sh                  # SLURM batch script for Perlmutter
├── requirements.txt            # Python dependencies
├── writeup.tex/writeup.pdf     # Detailed analysis and results
├── PARALLELIZATION_NOTES.md    # Multi-GPU implementation notes
└── *.png                       # Generated visualizations
```

## Components

### 1. Problem Setup
- **Signal distribution**: 10D Gaussian with mean μ_s = (2, 0, ..., 0), covariance I
- **Background distribution**: 10D Gaussian with mean μ_b = (0, 0, ..., 0), covariance I
- **Training data**: 10,000 samples (balanced)
- **Test data**: 5,000 samples (balanced)

### 2. Neyman-Pearson Optimal Classifier
Ground truth classifier using analytical likelihood ratio:
```
p*(y=1|x) = Λ(x) / (1 + Λ(x))
where Λ(x) = p(x|signal) / p(x|background)
```

### 3. Bootstrap Ensemble Strategy
- **Ensembles**: 20 independent ensembles
- **Models per ensemble**: 50 MLPs
- **Sampling**: Stratified bootstrap (with replacement, preserving class balance)
- **Total models trained**: 1,000

### 4. MLP Architecture
```
Input: 10 features
Hidden layers: [64, 32] with ReLU activation
Dropout: 0.1 after each hidden layer
Output: 1 unit with sigmoid (binary classification)
Optimizer: Adam (lr=0.001)
Training: 30 epochs, batch size 512
Loss: Binary cross-entropy
```

### 5. Coverage Evaluation
For a single ensemble and each test point x:
- Ensemble predictions from K=50 models: {p_1(x), ..., p_K(x)}
- Confidence interval: [q_α/2(x), q_1-α/2(x)]
- Check if Neyman-Pearson optimal prediction p*(x) falls within interval
- Aggregate across all test points to get coverage for this ensemble
- Repeat for 20 independent ensembles to estimate variability

## Setup

### On Perlmutter (NERSC)

```bash
# Load modules
module load pytorch/2.8.0

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy scipy scikit-learn matplotlib
```

### Local Setup

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Run on GPU (Recommended)
```bash
srun -A <your_account> -C gpu -q interactive -t 30:00 -N 1 --gpus-per-node=4 \
  bash -c "module load pytorch/2.8.0 && source venv/bin/activate && python main.py"
```

### Batch Job Submission
```bash
sbatch run_job.sh
```

### Local Run
```bash
python main.py
```

## Performance

### Multi-GPU Parallelization
The code automatically detects and uses available GPUs:
- **4 GPUs**: ~23 minutes (4x speedup)
- **1 GPU**: ~80 minutes
- **CPU**: Several hours

Training progress is printed in real-time showing which models are training on which GPUs.

## Results

### Coverage Statistics (95% Confidence Level)
Across 20 ensembles with 50 models each:

| Metric | Value |
|--------|-------|
| Coverage | 73-75% (target: 95%) |
| Mean Interval Width | 0.094-0.097 |
| MAE vs NP optimal | 0.025-0.026 |
| RMSE vs NP optimal | 0.037-0.039 |
| Ensemble Std | 0.026-0.027 |

### Key Observations
1. **Systematic undercoverage**: Each individual ensemble shows ~21% gap from nominal 95% level
2. **Good prediction accuracy**: Low MAE/RMSE vs optimal
3. **Consistent across ensembles**: All 20 independent ensembles show similar results (std=0.5%)
4. **Improvement with ensemble size**: 50 models (74%) vs 20 models (69.5%)
5. **Small variability**: The ±0.5% variation shows the problem is with the method, not specific bootstrap samples

### Generated Outputs
- `coverage_analysis.png` - Coverage vs confidence level plot
- `uncertainty_vs_error.png` - Ensemble uncertainty vs prediction error
- `writeup.pdf` - Full analysis with methodology and discussion

## Interpretation

### What Does This Mean?
If you train a **single** bootstrap ensemble (50 models on 10,000 samples):
- It will achieve approximately **74% coverage** at the 95% confidence level
- Different bootstrap samples give nearly identical results (±0.5%)
- Your confidence intervals will be **too narrow** to contain the true value 95% of the time
- This is a **systematic bias**, not random variation

### Why Undercoverage?
1. **Bootstrap limitations**: Resampled data too similar, models correlate
2. **Finite ensemble size**: 50 models may be insufficient for stable quantiles
3. **Neural network overconfidence**: Models overfit despite dropout/regularization
4. **No explicit calibration**: Raw predictions not post-processed

### Potential Improvements
- Increase ensemble size (100+ models)
- Deep ensembles (different random initializations, not just bootstrap)
- MC Dropout for additional uncertainty
- Temperature scaling / Platt scaling for calibration
- Conformal prediction for coverage guarantees
- More training data

## References

Key papers:
- Neyman & Pearson (1933) - Optimal hypothesis testing
- Breiman (1996) - Bagging predictors
- Lakshminarayanan et al. (2017) - Deep ensembles
- Guo et al. (2017) - Neural network calibration

## Notes

See `PARALLELIZATION_NOTES.md` for details on multi-GPU implementation.

For questions about methodology, see `writeup.pdf`.

---

## Project Status (2025-11-20)

**Status**: ✅ Code verified and updated

All code and documentation have been updated to correctly compute **single ensemble coverage** (mean ± std across 20 independent ensembles). The updated methodology properly answers: "How well does a SINGLE ensemble (trained on limited data) provide coverage?"

**Key Changes**:
- [main.py](main.py) lines 137-165: Per-ensemble coverage computation
- All documentation updated to reflect single ensemble interpretation
- Plots regenerated with error bands showing variability across bootstrap samples

**Note**: `writeup.pdf` requires manual regeneration with `pdflatex writeup.tex` on a machine with LaTeX (not available on Perlmutter).
