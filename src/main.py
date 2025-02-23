import argparse
import zarr
import os
import time
import logging
import numpy as np
import pandas as pd
from mldesigner import command_component, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes

from config import Configuration

# Define how we mount the input datastore as a filesystem (read-only)
INPUT_MODE = InputOutputModes.RO_MOUNT

# Define how we mount the output datastores as a filesystem (read-write)
OUTPUT_MODE = InputOutputModes.RW_MOUNT

# Tell Azure that the type of data we're pointing to is a folder
IN_DATA_TYPE = AssetTypes.URI_FOLDER
OUT_DATA_TYPE = AssetTypes.URI_FOLDER

@command_component(
    name="preprocess_data",
    version="1",
    display_name="Preprocess data",
    description="Downsample the original 0.25-degree atmospheric data to 1-degree, and split into train, test and validation sets",
    environment=dict(
        conda_file="./environment.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
    inputs={
        "dataset_path": Input(type=IN_DATA_TYPE, mode=INPUT_MODE),
        "config_path": Input(type=AssetTypes.URI_FILE, mode=INPUT_MODE),
    },
    outputs={
        "train_output": Output(type=OUT_DATA_TYPE, mode=OUTPUT_MODE),
        "validation_output": Output(type=OUT_DATA_TYPE, mode=OUTPUT_MODE),
        "test_output": Output(type=OUT_DATA_TYPE, mode=OUTPUT_MODE),
    },
)

def preprocess_data(
    dataset_path: str,
    config_path: str,
    train_output: str,
    validation_output: str,
    test_output: str,
    logs_dir: str = "./logs",
    log_level: str = "INFO",
):
    """Preprocess the dataset and save the splits to the specified output paths."""
    # Setup logging
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    config = Configuration.load(config_path)

    ds = zarr.open(dataset_path, mode="r")

    train_indices = get_period_indices(config.data.train_period)
    val_indices = get_period_indices(config.data.val_period)
    test_indices = get_period_indices(config.data.test_period)

    # Process splits using calculated indices
    splits = {
        "train": {var: ds[var][train_indices] for var in ds.keys()},
        "validation": {var: ds[var][val_indices] for var in ds.keys()},
        "test": {var: ds[var][test_indices] for var in ds.keys()}
    }

    processed_splits = {}
    target_lat = config.preprocessor.target_lat
    target_lon = config.preprocessor.target_lon
    for split_name, split_data in splits.items():
        processed_splits[split_name] = {
            var: downsample_spatial_resolution(arr, target_lat, target_lon)
            for var, arr in split_data.items()
        }

    # Save processed splits to the output paths provided by Azure ML
    save_zarr(processed_splits["train"], train_output)
    save_zarr(processed_splits["validation"], validation_output)
    save_zarr(processed_splits["test"], test_output)

    logging.info(f"Train split saved to {train_output}")
    logging.info(f"Validation split saved to {validation_output}")
    logging.info(f"Test split saved to {test_output}")

def get_period_indices(period: dict[str, str]) -> list[int]:
    """Get indices for a given period."""
    start = pd.to_datetime(period["start"])
    end = pd.to_datetime(period["end"])
    date_range = pd.date_range(start, end, freq='ME')
    base_date = pd.to_datetime('2019-01-01')
    
    return [(date.year - base_date.year) * 12 + date.month - base_date.month 
            for date in date_range]

def downsample_spatial_resolution(
    data: np.ndarray,
    target_lat: int,
    target_lon: int
) -> np.ndarray:
    """Downsample spatial dimensions of a numpy array."""
    orig_lat, orig_lon = data.shape[1], data.shape[2]
    lat_idx = np.linspace(0, orig_lat-1, target_lat, dtype=int)
    lon_idx = np.linspace(0, orig_lon-1, target_lon, dtype=int)
    return data[:, lat_idx][:, :, lon_idx]

def save_zarr(data: dict, path: str):
    """Save a dictionary of numpy arrays to a Zarr store."""
    os.makedirs(path, exist_ok=True)
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store)
    for var, arr in data.items():
        root.create_dataset(var, data=arr, chunks=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', metavar="config_path", default='./config/config.yaml')
    parser.add_argument('--dataset-path', metavar="dataset_path", default='./ERA5_data/zarr/full_dataset.zarr')
    parser.add_argument('--train-output', metavar="train_output", required=True)
    parser.add_argument('--validation-output', metavar="validation_output", required=True)
    parser.add_argument('--test-output', metavar="test_output", required=True)
    parser.add_argument('--logs-dir', metavar="logs_dir", default='./logs')
    parser.add_argument('--log-level', metavar="log_level", default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    preprocess_data(
        dataset_path=args.dataset_path,
        config_path=args.config_path,
        train_output=args.train_output,
        validation_output=args.validation_output,
        test_output=args.test_output,
        logs_dir=args.logs_dir,
        log_level=args.log_level,
    )
