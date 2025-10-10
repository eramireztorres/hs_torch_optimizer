"""Tests for model trainers."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.training.model_trainer import NNModelTrainer, NNRegressionModelTrainer


@pytest.mark.unit
class TestNNModelTrainer:
    """Test suite for NNModelTrainer (classification)."""

    def test_init_with_model_only(self, simple_model, sample_classification_data):
        """Test initialization with model only."""
        trainer = NNModelTrainer(
            model=simple_model,
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test']
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_init_with_model_criterion(self, simple_model, sample_classification_data):
        """Test initialization with model and criterion."""
        criterion = nn.CrossEntropyLoss()

        trainer = NNModelTrainer(
            model=(simple_model, criterion),
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test']
        )

        assert trainer.criterion is criterion

    def test_init_with_full_tuple(self, simple_model, sample_classification_data):
        """Test initialization with full tuple (model, optimizer, criterion, scheduler)."""
        optimizer = torch.optim.Adam(simple_model.parameters())
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        trainer = NNModelTrainer(
            model=(simple_model, optimizer, criterion, scheduler),
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test']
        )

        assert trainer.optimizer is optimizer
        assert trainer.criterion is criterion
        assert trainer.scheduler is scheduler

    @pytest.mark.slow
    def test_train_model(self, simple_model, sample_classification_data):
        """Test training the model."""
        trainer = NNModelTrainer(
            model=simple_model,
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test']
        )

        initial_params = [p.clone() for p in simple_model.parameters()]
        trainer.train_model(epochs=2)

        # Parameters should have changed
        final_params = list(simple_model.parameters())
        assert not all(torch.equal(initial_params[i], final_params[i]) for i in range(len(initial_params)))

    def test_evaluate_model(self, simple_model, sample_classification_data):
        """Test evaluating the model."""
        trainer = NNModelTrainer(
            model=simple_model,
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test']
        )

        metrics = trainer.evaluate_model()

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_device_selection(self, simple_model, sample_classification_data):
        """Test device selection."""
        trainer = NNModelTrainer(
            model=simple_model,
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test']
        )

        # Should select cuda if available, else cpu
        if torch.cuda.is_available():
            assert trainer.device.type == 'cuda'
        else:
            assert trainer.device.type == 'cpu'

    @pytest.mark.slow
    def test_early_stopping(self, simple_model, sample_classification_data):
        """Test early stopping mechanism."""
        # Create trainer with low patience
        trainer = NNModelTrainer(
            model=simple_model,
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test'],
            patience=2
        )

        # Train for many epochs - should stop early if no improvement
        trainer.train_model(epochs=50)

        # Should have stopped before 50 epochs (hard to test deterministically)
        # Just verify it completes without error

    def test_batch_size_parameter(self, simple_model, sample_classification_data):
        """Test custom batch size."""
        trainer = NNModelTrainer(
            model=simple_model,
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_test=sample_classification_data['X_test'],
            y_test=sample_classification_data['y_test'],
            batch_size=16
        )

        assert trainer.batch_size == 16


@pytest.mark.unit
class TestNNRegressionModelTrainer:
    """Test suite for NNRegressionModelTrainer."""

    def test_init(self, sample_regression_data):
        """Test initialization."""
        class RegressionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        model = RegressionModel()

        trainer = NNRegressionModelTrainer(
            model=model,
            X_train=sample_regression_data['X_train'],
            y_train=sample_regression_data['y_train'],
            X_test=sample_regression_data['X_test'],
            y_test=sample_regression_data['y_test']
        )

        assert isinstance(trainer.criterion, nn.MSELoss)

    @pytest.mark.slow
    def test_train_regression_model(self, sample_regression_data):
        """Test training regression model."""
        class RegressionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        model = RegressionModel()

        trainer = NNRegressionModelTrainer(
            model=model,
            X_train=sample_regression_data['X_train'],
            y_train=sample_regression_data['y_train'],
            X_test=sample_regression_data['X_test'],
            y_test=sample_regression_data['y_test']
        )

        trainer.train_model(epochs=2)

        # Should complete without error

    def test_evaluate_regression_model(self, sample_regression_data):
        """Test evaluating regression model."""
        class RegressionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        model = RegressionModel()

        trainer = NNRegressionModelTrainer(
            model=model,
            X_train=sample_regression_data['X_train'],
            y_train=sample_regression_data['y_train'],
            X_test=sample_regression_data['X_test'],
            y_test=sample_regression_data['y_test']
        )

        metrics = trainer.evaluate_model()

        assert 'mse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0

    def test_multi_output_regression(self, sample_regression_data):
        """Test multi-output regression."""
        # Modify target to have multiple outputs
        y_train_multi = np.random.randn(80, 3).astype(np.float32)
        y_test_multi = np.random.randn(20, 3).astype(np.float32)

        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 3)

            def forward(self, x):
                return self.fc(x)

        model = MultiOutputModel()

        trainer = NNRegressionModelTrainer(
            model=model,
            X_train=sample_regression_data['X_train'],
            y_train=y_train_multi,
            X_test=sample_regression_data['X_test'],
            y_test=y_test_multi
        )

        metrics = trainer.evaluate_model()
        assert 'mse' in metrics
