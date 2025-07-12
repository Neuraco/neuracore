"""Model archive (MAR) creation utility for Neuracore model deployment.

This module provides functionality to package Neuracore models into TorchServe
Model Archive (.mar) files for deployment. It handles model serialization,
dependency management, and packaging of all required files for inference.
"""

import inspect
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import torch
from model_archiver.model_archiver import ModelArchiver
from model_archiver.model_archiver_config import ModelArchiverConfig

from neuracore.core.nc_types import ModelDevice, ModelInitDescription
from neuracore.ml.core.neuracore_model import NeuracoreModel
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader


def create_mar(
    model: NeuracoreModel, output_dir: Path, algorithm_config: dict = {}
) -> None:
    """Create a TorchServe Model Archive (MAR) file from a Neuracore model.

    Packages a trained Neuracore model into a deployable MAR file that includes
    the model weights, algorithm code, configuration metadata, and dependencies.
    The resulting MAR file can be deployed to TorchServe for inference.

    Args:
        model: Trained Neuracore model instance to package for deployment.
        output_dir: Directory path where the MAR file will be created.
        algorithm_config: Custom configuration for the algorithm.
    """
    algorithm_file = Path(inspect.getfile(model.__class__))
    algorithm_loader = AlgorithmLoader(algorithm_file.parent)
    algo_files = algorithm_loader.get_all_files()
    extra_files = [str(f) for f in algo_files]
    requirements_file_path = algorithm_loader.algorithm_dir / "requirements.txt"
    if requirements_file_path.exists():
        requirements_file = str(requirements_file_path.resolve())
        extra_files.append(requirements_file)
    else:
        requirements_file = None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        torch.save(model.state_dict(), temp_path / "model.pt")
        with open(temp_path / "model_init_description.json", "w") as f:
            json.dump(model.model_init_description.model_dump(), f, indent=2)
        extra_files.append(str(temp_path / "model_init_description.json"))
        if algorithm_config is not None:
            with open(temp_path / "algorithm_config.json", "w") as f:
                json.dump(algorithm_config, f, indent=2)
            extra_files.append(str(temp_path / "algorithm_config.json"))

        FILE_PATH = Path(__file__).parent / "handlers.py"
        # All the files are "baked" into the model archive
        ModelArchiver.generate_model_archive(
            ModelArchiverConfig(
                model_name="model",
                version="1.0",
                model_file=str(algorithm_file),
                serialized_file=str(temp_path / "model.pt"),
                handler=str(FILE_PATH.resolve()),
                export_path=str(output_dir),
                extra_files=",".join(extra_files),
                force=True,
                requirements_file=requirements_file,
            )
        )


def extract_mar(mar_file: Path, output_dir: Path) -> dict[str, Path]:
    """Extract all contents from a TorchServe Model Archive (MAR) file.

    Extracts all files from a MAR archive including model weights, algorithm code,
    configuration files, and dependencies. The MAR file is essentially a ZIP file
    with a specific structure.

    Args:
        mar_file: Path to the MAR file to extract.
        output_dir: Directory where extracted files will be saved.

    Returns:
        Dictionary mapping file types to their extracted paths.

    Raises:
        FileNotFoundError: If the MAR file doesn't exist.
        zipfile.BadZipFile: If the MAR file is corrupted or not a valid ZIP.
    """
    if not mar_file.exists():
        raise FileNotFoundError(f"MAR file not found: {mar_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = {}

    with zipfile.ZipFile(mar_file, "r") as zip_ref:
        # Extract all files
        zip_ref.extractall(output_dir)

        # Catalog the extracted files
        for file_info in zip_ref.filelist:
            file_path = output_dir / file_info.filename

            # Categorize files based on their names/extensions
            if file_info.filename.endswith(".pt"):
                extracted_files["model_weights"] = file_path
            elif file_info.filename == "model_init_description.json":
                extracted_files["model_init_description"] = file_path
            elif file_info.filename == "algorithm_config.json":
                extracted_files["algorithm_config"] = file_path
            elif file_info.filename.endswith(".py"):
                if "handler" not in extracted_files:
                    extracted_files["handlers"] = []
                extracted_files.setdefault("python_files", []).append(file_path)
            elif file_info.filename == "requirements.txt":
                extracted_files["requirements"] = file_path
            elif file_info.filename == "MANIFEST.json":
                extracted_files["manifest"] = file_path
            else:
                extracted_files.setdefault("other_files", []).append(file_path)

    return extracted_files


def load_model_from_mar(
    mar_file: Path, extract_to: Optional[Path] = None
) -> NeuracoreModel:
    """Load a Neuracore model from a MAR file.

    Extracts the MAR file and reconstructs the original Neuracore model instance
    with its trained weights and configuration.

    Args:
        mar_file: Path to the MAR file.
        extract_to: Optional directory to extract files to. If None, uses a temporary directory.

    Returns:
        NeuracoreModel: The reconstructed model instance ready for inference.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_to = Path(temp_dir)

        # Extract the MAR file
        extracted_files = extract_mar(mar_file, extract_to)

        # Load model initialization description
        if "model_init_description" not in extracted_files:
            raise FileNotFoundError("model_init_description.json not found in MAR file")

        with open(extracted_files["model_init_description"], "r") as f:
            model_init_description = json.load(f)
        model_init_description = ModelInitDescription.model_validate(
            model_init_description
        )
        model_init_description.device = ModelDevice.AUTO

        # Load algorithm config if present
        algorithm_config = {}
        if "algorithm_config" in extracted_files:
            with open(extracted_files["algorithm_config"], "r") as f:
                algorithm_config = json.load(f)

        # Find the algorithm file (main model Python file)
        algorithm_file = None
        if "python_files" in extracted_files:
            for py_file in extracted_files["python_files"]:
                if py_file.name != "handlers.py":  # Skip the handler file
                    algorithm_file = py_file
                    break

        if algorithm_file is None:
            raise FileNotFoundError("Algorithm Python file not found in MAR file")

        # Load the algorithm using AlgorithmLoader
        algorithm_loader = AlgorithmLoader(algorithm_file.parent)
        algorithm_loader.install_requirements()
        model_class = algorithm_loader.load_model()

        # Create model instance
        model = model_class(model_init_description, **algorithm_config)
        model.to(model.device)  # Move model to the appropriate device

        # Load trained weights if present
        if "model_weights" in extracted_files:
            state_dict = torch.load(
                extracted_files["model_weights"],
                map_location=model.device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)

        return model
