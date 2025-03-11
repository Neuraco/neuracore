import pytest
import torch
import torch.nn as nn

from neuracore import (
    BatchedInferenceOutputs,
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    DatasetDescription,
)
from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP

BS = 2
CAMS = 1
STATE_DIM = 32
ACTION_DIM = 7
PRED_HORIZON = 10


@pytest.fixture
def dataset_description() -> DatasetDescription:
    return DatasetDescription(
        max_num_cameras=CAMS,
        max_state_size=STATE_DIM,
        max_action_size=ACTION_DIM,
        action_mean=torch.ones(ACTION_DIM, dtype=torch.float32),
        action_std=torch.ones(ACTION_DIM, dtype=torch.float32),
        state_mean=torch.ones(STATE_DIM, dtype=torch.float32),
        state_std=torch.ones(STATE_DIM, dtype=torch.float32),
        action_prediction_horizon=PRED_HORIZON,
    )


@pytest.fixture
def model_config() -> dict:
    return {}


@pytest.fixture
def sample_batch() -> BatchedTrainingSamples:
    return BatchedTrainingSamples(
        states=torch.randn(BS, STATE_DIM, dtype=torch.float32),
        states_mask=torch.ones(BS, STATE_DIM, dtype=torch.float32),
        camera_images=torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
        camera_images_mask=torch.ones(BS, CAMS, dtype=torch.float32),
        actions=torch.randn(BS, PRED_HORIZON, ACTION_DIM, dtype=torch.float32),
        actions_mask=torch.ones(BS, ACTION_DIM, dtype=torch.float32),
        actions_sequence_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
    )


@pytest.fixture
def sample_inference_batch() -> BatchedTrainingSamples:
    return BatchedInferenceSamples(
        states=torch.randn(BS, STATE_DIM, dtype=torch.float32),
        states_mask=torch.ones(BS, STATE_DIM, dtype=torch.float32),
        camera_images=torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
        camera_images_mask=torch.ones(BS, CAMS, dtype=torch.float32),
    )


@pytest.fixture
def mock_dataloader(sample_batch):
    """Create a mock dataloader."""

    def generate_batch():
        return sample_batch

    class MockDataLoader:
        def __iter__(self):
            for _ in range(2):  # 2 batches per epoch
                yield generate_batch()

        def __len__(self):
            return 2

    return MockDataLoader()


def test_model_construction(
    dataset_description: DatasetDescription, model_config: dict
):
    model = CNNMLP(dataset_description, **model_config)
    assert isinstance(model, nn.Module)


def test_model_forward(
    dataset_description: DatasetDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceSamples,
):
    model = CNNMLP(dataset_description, **model_config)
    output = model(sample_inference_batch)
    assert isinstance(output, BatchedInferenceOutputs)
    assert output.action_predicitons.shape == (BS, PRED_HORIZON, ACTION_DIM)


def test_model_backward(
    dataset_description: DatasetDescription,
    model_config: dict,
    sample_batch: BatchedTrainingSamples,
):
    model = CNNMLP(dataset_description, **model_config)
    output: BatchedTrainingOutputs = model.training_step(sample_batch)

    # Compute loss
    loss = output.losses["mse_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
