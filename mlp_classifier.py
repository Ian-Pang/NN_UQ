"""
MLP classifier with ensemble support.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for binary classification."""

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.1):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPClassifier:
    """Wrapper for MLP with sklearn-like interface."""

    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=128,
        n_epochs=100,
        device=None,
        random_state=None
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: torch device (cpu/cuda)
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)

        self.model = None

    def fit(self, X, y, verbose=False):
        """
        Train the classifier.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            verbose: Print training progress
        """
        self.model = MLP(
            self.input_dim,
            self.hidden_dims,
            self.dropout_rate
        ).to(self.device)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        return self

    def predict_proba(self, X):
        """
        Predict probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            proba: Probability of being signal (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            proba = self.model(X_tensor).cpu().numpy().flatten()
        return proba

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold

        Returns:
            y_pred: Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)


class MLPEnsemble:
    """Ensemble of MLP classifiers trained on bootstrap samples."""

    def __init__(self, n_estimators, **mlp_kwargs):
        """
        Args:
            n_estimators: Number of models in the ensemble
            **mlp_kwargs: Arguments passed to MLPClassifier
        """
        self.n_estimators = n_estimators
        self.mlp_kwargs = mlp_kwargs
        self.models = []

    def fit(self, bootstrap_samples, verbose=False, n_jobs=1):
        """
        Train ensemble on bootstrap samples.

        Args:
            bootstrap_samples: List of (X, y) tuples from bootstrap resampling
            verbose: Print training progress
            n_jobs: Number of parallel jobs (GPUs) to use. If > 1, trains multiple models in parallel.
        """
        assert len(bootstrap_samples) == self.n_estimators, \
            f"Expected {self.n_estimators} bootstrap samples, got {len(bootstrap_samples)}"

        if n_jobs == 1:
            # Sequential training (original behavior)
            self.models = []
            for i, (X_boot, y_boot) in enumerate(bootstrap_samples):
                if verbose:
                    print(f"\nTraining model {i + 1}/{self.n_estimators}")

                model = MLPClassifier(**self.mlp_kwargs)
                model.fit(X_boot, y_boot, verbose=verbose)
                self.models.append(model)
        else:
            # Parallel training across multiple GPUs
            import torch.multiprocessing as mp
            from functools import partial

            # Determine number of available GPUs
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                print("WARNING: No GPUs available, falling back to CPU sequential training")
                return self.fit(bootstrap_samples, verbose=verbose, n_jobs=1)

            n_workers = min(n_jobs, n_gpus)
            if verbose:
                print(f"Using {n_workers} GPUs for parallel training")

            # Create a pool of workers
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=n_workers) as pool:
                # Assign each model to a GPU in round-robin fashion
                train_args = []
                for i, (X_boot, y_boot) in enumerate(bootstrap_samples):
                    gpu_id = i % n_workers
                    train_args.append((i, X_boot, y_boot, gpu_id, self.mlp_kwargs, verbose))

                # Train models in parallel
                results = pool.map(self._train_single_model, train_args)

            # Sort by index and extract models
            results.sort(key=lambda x: x[0])
            self.models = [model for _, model in results]

        return self

    @staticmethod
    def _train_single_model(args):
        """
        Train a single model on a specific GPU.

        Args:
            args: Tuple of (index, X, y, gpu_id, mlp_kwargs, verbose)

        Returns:
            Tuple of (index, trained_model)
        """
        idx, X, y, gpu_id, mlp_kwargs, verbose = args

        # Set device for this worker
        device = torch.device(f'cuda:{gpu_id}')
        mlp_kwargs_copy = mlp_kwargs.copy()
        mlp_kwargs_copy['device'] = device

        if verbose:
            print(f"Training model {idx + 1} on GPU {gpu_id}")

        model = MLPClassifier(**mlp_kwargs_copy)
        model.fit(X, y, verbose=False)  # Don't print epoch-level progress in parallel mode

        return (idx, model)

    def predict_proba_ensemble(self, X):
        """
        Get predictions from all models in the ensemble.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            proba_ensemble: Array of shape (n_estimators, n_samples)
        """
        proba_list = []
        for model in self.models:
            proba = model.predict_proba(X)
            proba_list.append(proba)
        return np.array(proba_list)

    def predict_proba(self, X):
        """
        Predict probabilities using ensemble mean.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            proba: Mean probability across ensemble (n_samples,)
        """
        proba_ensemble = self.predict_proba_ensemble(X)
        return np.mean(proba_ensemble, axis=0)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels using ensemble mean.

        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold

        Returns:
            y_pred: Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

    def predict_proba_std(self, X):
        """
        Compute standard deviation of predictions across ensemble.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            std: Standard deviation of probabilities (n_samples,)
        """
        proba_ensemble = self.predict_proba_ensemble(X)
        return np.std(proba_ensemble, axis=0)

    def predict_proba_quantiles(self, X, quantiles=[0.025, 0.975]):
        """
        Compute quantiles of predictions across ensemble.

        Args:
            X: Features (n_samples, n_features)
            quantiles: List of quantiles to compute

        Returns:
            quantile_values: Array of shape (len(quantiles), n_samples)
        """
        proba_ensemble = self.predict_proba_ensemble(X)
        return np.quantile(proba_ensemble, quantiles, axis=0)
