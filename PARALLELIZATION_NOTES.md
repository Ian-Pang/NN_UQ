# NN_UQ Multi-GPU Parallelization

## Summary
Added multi-GPU parallel training support to speed up ensemble training.

## Changes Made

### 1. mlp_classifier.py
- Added `n_jobs` parameter to `MLPEnsemble.fit()` method
- Implemented parallel training across multiple GPUs using `torch.multiprocessing`
- Models are distributed round-robin across available GPUs
- Each GPU trains models independently in parallel

### 2. main.py
- Auto-detects available GPUs using `torch.cuda.device_count()`
- Passes `n_jobs=n_gpus` to ensemble training
- Maintains verbose output for progress tracking

### 3. run_job.sh
- Updated to request 4 GPUs per node: `--gpus-per-node=4`
- Uses PyTorch 2.8.0 module
- Account: m4539, interactive queue

## Usage

### Running on Interactive Queue
```bash
cd /global/homes/i/ipang001/NN_UQ
srun -A m4539 -C gpu -q interactive -t 30:00 -N 1 --gpus-per-node=4 \
  bash -c "module load pytorch/2.8.0 && source venv/bin/activate && python main.py"
```

### Or submit as batch job
```bash
sbatch run_job.sh
```

## Performance
- **Sequential (1 GPU)**: ~4 minutes per ensemble × 20 = ~80 minutes total
- **Parallel (4 GPUs)**: ~1 minute per ensemble × 20 = ~23 minutes total
- **Speedup**: ~3.5-4x

## Results (with 50 models per ensemble)
- Coverage at 95% CI: 74.3% ± 0.5% (improved from 69.5% with 20 models)
- Each individual ensemble shows systematic undercoverage (~21% gap from target)
- Small std (0.5%) shows this is a methodological problem, not random variation
- MAE vs Neyman-Pearson optimal: ~0.026
- All 20 independent ensembles show highly consistent results

## Key Code Locations
- Parallel training implementation: `mlp_classifier.py:177-256`
- GPU detection and configuration: `main.py:93-97`
- Training loop with n_jobs parameter: `main.py:105`

## Notes
- Multiprocessing uses 'spawn' context for PyTorch compatibility
- Individual epoch progress is suppressed in parallel mode (verbose=False in workers)
- Models are serialized after training and returned to main process
