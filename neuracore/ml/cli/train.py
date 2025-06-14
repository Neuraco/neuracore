import argparse
import gc
import json
import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, random_split

import neuracore as nc
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.nc_types import DataType, ModelInitDescription
from neuracore.ml import NeuracoreModel
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.trainers.batch_autotuner import find_optimal_batch_size
from neuracore.ml.trainers.distributed_trainer import (
    DistributedTrainer,
    cleanup_distributed,
    setup_distributed,
)
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader
from neuracore.ml.utils.metrics import GCPMetricsLogger, LocalMetricsLogger
from neuracore.ml.utils.storage_handler import (
    GCPStorageHandler,
    LocalStorageHandler,
    StorageHandler,
)

os.environ["PJRT_DEVICE"] = "GPU"


def create_parser():
    """Create argument parser with subparsers for local and GCP training."""
    parser = argparse.ArgumentParser(description="Robot Training Script")

    # Common arguments for both local and GCP training
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parent_parser.add_argument(
        "--resume", type=str, help="Path to checkpoint to resume from"
    )
    # parent_parser.add_argument(
    #     "--bucket",
    #     type=str,
    #     required=True,
    #     help="GCS bucket name for loading data (and storing outputs in GCP mode)",
    # )
    # parent_parser.add_argument(
    #     "--org_id", type=str, required=True, help="Organization ID"
    # )
    # parent_parser.add_argument(
    #     "--synced_data_id", type=str, required=True, help="Synced Data ID"
    # )
    # parent_parser.add_argument(
    #     "--database_name",
    #     type=str,
    #     required=True,
    #     help="Firestore database name",
    # )
    parent_parser.add_argument(
        "--batch_size",
        type=str,
        required=True,
        help="Batch size for training",
    )
    parent_parser.add_argument(
        "--epochs", type=int, required=True, help="Number of epochs to train for"
    )
    parent_parser.add_argument(
        "--output_prediction_horizon",
        type=int,
        default=10,
        help="Number of actions to predict in the future",
    )
    parent_parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation",
    )
    parent_parser.add_argument(
        "--output_data_types",
        nargs="+",
        type=str,
        default=[
            DataType.JOINT_TARGET_POSITIONS.value,
        ],
        help="Type of action data to use (joint_positions, joint_torques, etc.)",
    )
    parent_parser.add_argument(
        "--input_data_types",
        nargs="+",
        type=str,
        default=[
            DataType.JOINT_POSITIONS.value,
            DataType.RGB_IMAGE.value,
        ],
        help="List of input data types to use (joint_positions, joint_torques, etc.)",
    )

    # Create subparsers for local and GCP training
    # subparsers = parser.add_subparsers(dest="mode", help="Training mode")
    # subparsers.required = True

    # # Local training subparser
    # local_parser = subparsers.add_parser(
    #     "local", parents=[parent_parser], help="Run training locally"
    # )
    parent_parser.add_argument(
        "--local_output_dir",
        type=str,
        default="./output",
        help="Local directory for outputs when training locally",
    )
    parent_parser.add_argument(
        "--algorithm_path",
        type=str,
        required=True,
        help="Algorithm to use for training",
    )

    # # GCP training subparser
    # gcp_parser = subparsers.add_parser(
    #     "gcp", parents=[parent_parser], help="Run as if on gcp"
    # )
    # gcp_parser.add_argument(
    #     "--training_id", type=str, required=True, help="Training ID"
    # )
    # gcp_parser.add_argument(
    #     "--algorithm_id", type=str, required=True, help="Algorithm ID"
    # )
    parent_parser.add_argument(
        "--algorithm_config",
        type=str,
        required=True,
        help="JSON or YAML config containing algorithm and training parameters",
    )

    return parser


def setup_logging(output_dir: Path, rank: int = 0):
    """Setup logging configuration."""
    # Only set up file logging for rank 0
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(output_dir / "train.log"),
            ],
        )
    else:
        # For other ranks, only log to console
        logging.basicConfig(
            level=logging.INFO,
            format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )


def get_model_class(
    args, storage_handler: StorageHandler, logger
) -> type[NeuracoreModel]:
    """Get model class from algorithm string."""
    # Create model
    if args.mode == "local":
        algo_path = Path(args.algorithm_path)
        logger.info(f"Loading algorithm from {algo_path}")
        algorithm_loader = AlgorithmLoader(algo_path)
        model_class = algorithm_loader.load_model()
    else:  # GCP mode
        # Load algorithm from storage using algorithm_id
        try:
            algorithm_dir = f"organizations/shared/algorithms/{args.algorithm_id}"
            ext_algorithm_dir = storage_handler.download_algorithm(algorithm_dir)
        except FileNotFoundError:
            logger.info("Algorithm not found in shared. Looking at private.")
            algorithm_dir = (
                f"organizations/{args.org_id}/algorithms/{args.algorithm_id}"
            )
            ext_algorithm_dir = storage_handler.download_algorithm(algorithm_dir)
        logger.info(f"Algorithm extracted to {ext_algorithm_dir}")
        logger.info("Loading algorithm model class")
        algorithm_loader = AlgorithmLoader(ext_algorithm_dir)
        model_class = algorithm_loader.load_model()
    return model_class


def create_storage_handler(args):
    if args.mode == "local":
        storage_handler = LocalStorageHandler(local_dir=args.local_output_dir)
    else:  # GCP mode
        storage_handler = GCPStorageHandler(
            bucket_name=args.bucket,
            database_name=args.database_name,
            org_id=args.org_id,
            training_id=args.training_id,
        )
    return storage_handler


def extract_data_types(args) -> tuple[list[str], list[str]]:
    input_data_types = args.input_data_types
    if len(input_data_types) == 1:
        input_data_types = input_data_types[0].split(" ")

    output_data_types = args.output_data_types
    if len(output_data_types) == 1:
        output_data_types = output_data_types[0].split(" ")

    return input_data_types, output_data_types


def run_training(
    rank,
    world_size,
    args,
    algorithm_config,
    batch_size: int,
    synchronized_dataset: SynchronizedDataset,
):
    """Run the training process for a single GPU."""
    # Setup for distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Setup output directory
    output_dir = Path(args.local_output_dir if args.mode == "local" else "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (different file per process)
    setup_logging(output_dir, rank)
    logger = logging.getLogger(__name__)

    # Set random seed (different for each process to ensure different data sampling)
    torch.manual_seed(args.seed + rank)

    epochs = args.epochs

    try:

        logger.info(f"Using batch size: {batch_size}")

        # Initialize storage handler based on mode
        storage_handler = create_storage_handler(args)

        input_data_types, output_data_types = extract_data_types(args)
        input_data_types = [DataType(t) for t in input_data_types]
        output_data_types = [DataType(t) for t in output_data_types]

        # Create model
        model_class = get_model_class(args, storage_handler, logger)
        logger.info(f"Loaded model class: {model_class}")

        # Setup dataset
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=synchronized_dataset,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=args.output_prediction_horizon,
            tokenize_text=model_class.tokenize_text,
        )

        # Split dataset
        dataset_size = len(dataset)
        train_split = 1 - args.validation_split
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size

        # Use random split with fixed seed for deterministic behavior
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        num_workers = min(os.cpu_count() // 2, 4)

        # Create samplers and data loaders
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=args.seed,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=dataset.collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=dataset.collate_fn,
            )
        else:
            # Regular data loaders for single GPU training
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=dataset.collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=dataset.collate_fn,
            )

        # Log data loader information
        logger.info(
            f"Created data loaders with {len(train_loader.dataset)} training samples "
            f"and {len(val_loader.dataset)} validation samples"
        )

        model_init_description = ModelInitDescription(
            dataset_description=dataset.dataset_description,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=args.output_prediction_horizon,
        )

        model = model_class(
            model_init_description=model_init_description,
            **algorithm_config,
        )
        logger.info(
            f"Created model with "
            f"{sum(p.numel() for p in model.parameters()):,} parameters"
        )

        # Setup metrics logger based on mode
        if args.mode == "local":
            metrics_logger = LocalMetricsLogger(local_dir=args.local_output_dir)
        else:  # GCP mode
            metrics_logger = GCPMetricsLogger(
                database_name=args.database_name,
                org_id=args.org_id,
                training_id=args.training_id,
            )

        # Create trainer
        trainer = DistributedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            metrics_logger=metrics_logger,
            storage_handler=storage_handler,
            output_dir=output_dir,
            algorithm_config=algorithm_config,
            num_epochs=epochs,
            rank=rank,
            world_size=world_size,
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            try:
                checkpoint = trainer.load_checkpoint(args.resume)
                start_epoch = checkpoint.get("epoch", 0) + 1
                logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
            except Exception as e:
                # Log error and continue training from scratch
                # Assume that the user stopped training before the first epoch
                logger.error(f"Failed to load checkpoint: {str(e)}")

        # Start training
        try:
            logger.info("Starting training...")
            trainer.train(start_epoch=start_epoch)
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    finally:
        # Clean up distributed process group
        if world_size > 1:
            cleanup_distributed()

        logger.info(f"Process {rank} completed")


def determine_optimal_batch_size(
    args,
    algorithm_config: dict,
    storage_handler: StorageHandler,
    synchronized_dataset: SynchronizedDataset,
):
    """Run batch size autotuning on a single GPU and return the result."""

    logger = logging.getLogger(__name__)
    logger.info("Starting batch size autotuning on GPU 0...")

    input_data_types, output_data_types = extract_data_types(args)
    input_data_types = [DataType(t) for t in input_data_types]
    output_data_types = [DataType(t) for t in output_data_types]

    # Setup dataset for autotuning
    dataset = PytorchSynchronizedDataset(
        synchronized_dataset=synchronized_dataset,
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        output_prediction_horizon=args.output_prediction_horizon,
    )

    # Create a smaller subset for autotuning
    train_size = len(dataset)
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))

    model_class = get_model_class(args, storage_handler, logger)

    model_init_description = ModelInitDescription(
        dataset_description=dataset.dataset_description,
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        output_prediction_horizon=args.output_prediction_horizon,
    )

    model = model_class(
        model_init_description=model_init_description,
        **algorithm_config,
    )

    # Determine per-GPU batch size
    optimal_batch_size = find_optimal_batch_size(
        dataset=train_dataset,
        model=model,
        model_kwargs=algorithm_config,
        min_batch_size=2,
        max_batch_size=4096,
        gpu_id=0,
        dataloader_kwargs={
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "collate_fn": dataset.collate_fn,
        },
    )

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info(
        f"Autotuning complete. Optimal batch size per GPU: {optimal_batch_size}"
    )
    return optimal_batch_size


def main():
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    batch_size = args.batch_size
    algorithm_config = (
        json.loads(args.algorithm_config)
        if isinstance(args.algorithm_config, str)
        else args.algorithm_config
    )

    data_types_to_sync = args.input_data_types + args.output_data_types
    dataset_name_to_train_on = args.dataset_name
    frequency = args.frequency

    nc.login()
    dataset = nc.get_dataset(dataset_name_to_train_on)
    synchronized_dataset = dataset.synchronize(
        frequency=frequency, data_types=data_types_to_sync
    )

    # Create output directory
    output_dir = Path(args.local_output_dir if args.mode == "local" else "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging for main process
    setup_logging(output_dir)

    # Check if distributed training is enabled and multiple GPUs are available
    world_size = torch.cuda.device_count()

    # Handle batch size configuration
    if type(batch_size) == str:
        batch_size = batch_size.lower()
    if batch_size == "auto":
        # Run autotuning before launching distributed training
        storage_handler = create_storage_handler(args)
        optimal_batch_size = determine_optimal_batch_size(
            args, dict(algorithm_config), storage_handler
        )
        # Update algorithm_config with the determined batch size
        batch_size = optimal_batch_size
    else:
        batch_size = int(batch_size)

    if world_size > 1:
        # Use multiprocessing to launch multiple processes
        mp.spawn(
            run_training,
            args=(world_size, args, algorithm_config, batch_size, synchronized_dataset),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU or CPU training
        run_training(0, 1, args, algorithm_config, batch_size, synchronized_dataset)


if __name__ == "__main__":
    main()
