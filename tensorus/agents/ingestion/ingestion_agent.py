"""
Ingestion agent for Tensorus.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from tensorus.agents import Agent
from tensorus.storage.client import TensorusStorageClient
from tensorus.storage.schema import DatasetSchema, DType, TensorSchema


class IngestionStatus(str, Enum):
    """Status of an ingestion operation."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    status: IngestionStatus
    records_processed: int
    records_ingested: int
    execution_time: float
    dataset_id: str
    errors: List[str]
    metadata: Dict[str, Any]


class IngestionAgent(Agent):
    """Agent for autonomously ingesting data into Tensorus."""
    
    def __init__(
        self,
        storage_client: TensorusStorageClient,
        watch_paths: Optional[List[str]] = None,
        polling_interval: int = 60,
        transform_functions: Optional[Dict[str, Callable]] = None,
        logger: Optional[logging.Logger] = None,
        auto_create_dataset: bool = True,
    ):
        """Initialize the ingestion agent.
        
        Args:
            storage_client: Storage client for interacting with the Tensorus storage engine.
            watch_paths: Paths to watch for new data.
            polling_interval: Interval in seconds for polling watch paths.
            transform_functions: Functions for transforming data before ingestion.
            logger: Logger for the agent.
            auto_create_dataset: Whether to automatically create datasets if they don't exist.
        """
        super().__init__(
            storage_client=storage_client,
            name="ingestion_agent",
            description="Agent for autonomously ingesting data into Tensorus.",
            logger=logger,
        )
        
        self.watch_paths = watch_paths or []
        self.polling_interval = polling_interval
        self.transform_functions = transform_functions or {}
        self.auto_create_dataset = auto_create_dataset
        
        # Keep track of processed files
        self.processed_files: List[str] = []
        
        # Last poll time
        self.last_poll_time = 0
        
    def _ingest_file(
        self,
        file_path: str,
        dataset_id: str,
        tensor_mapping: Dict[str, str],
        transform_function: Optional[Callable] = None,
    ) -> IngestionResult:
        """Ingest a file into a dataset.
        
        Args:
            file_path: Path to the file to ingest.
            dataset_id: ID of the dataset to ingest into.
            tensor_mapping: Mapping from file columns/keys to tensor names.
            transform_function: Function for transforming data before ingestion.
            
        Returns:
            Result of the ingestion operation.
            
        Raises:
            ValueError: If the file could not be ingested.
        """
        start_time = time.time()
        
        try:
            # Log start of ingestion
            self.log_event("info", f"Ingesting file {file_path} into dataset {dataset_id}")
            
            # Check if dataset exists
            dataset_info = self.storage_client.get_dataset_info(dataset_id)
            if dataset_info is None:
                if not self.auto_create_dataset:
                    return IngestionResult(
                        status=IngestionStatus.FAILURE,
                        records_processed=0,
                        records_ingested=0,
                        execution_time=time.time() - start_time,
                        dataset_id=dataset_id,
                        errors=[f"Dataset {dataset_id} does not exist, and auto_create_dataset is False"],
                        metadata={"file_path": file_path},
                    )
                    
                # Create dataset
                self.log_event("info", f"Creating dataset {dataset_id}")
                self.storage_client.create_dataset(
                    name=dataset_id,
                    description=f"Auto-created dataset for {os.path.basename(file_path)}",
                )
                
            # Load data from file
            data = self._load_data_from_file(file_path)
            
            # Apply transform function if provided
            if transform_function is not None:
                data = transform_function(data)
                
            # Get tensors from data
            tensors = self._extract_tensors(data, tensor_mapping)
            
            # Ingest tensors
            records_ingested = 0
            errors = []
            
            for tensor_name, tensor_data in tensors.items():
                try:
                    # Check if tensor exists
                    tensor_info = self.storage_client.get_tensor_info(dataset_id, tensor_name)
                    if tensor_info is None:
                        # Create tensor
                        self._create_tensor_from_data(dataset_id, tensor_name, tensor_data)
                        
                    # Append data to tensor
                    self.storage_client.append_tensor_data(dataset_id, tensor_name, tensor_data)
                    records_ingested += 1
                except Exception as e:
                    errors.append(f"Failed to ingest tensor {tensor_name}: {str(e)}")
                    
            # Determine status
            status = IngestionStatus.SUCCESS
            if errors:
                if records_ingested == 0:
                    status = IngestionStatus.FAILURE
                else:
                    status = IngestionStatus.PARTIAL
                    
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Add to processed files
            self.processed_files.append(file_path)
            
            # Log completion
            self.log_event("info", f"Ingestion completed with status {status}", metadata={
                "file_path": file_path,
                "dataset_id": dataset_id,
                "records_ingested": records_ingested,
                "execution_time": execution_time,
            })
            
            return IngestionResult(
                status=status,
                records_processed=len(tensors),
                records_ingested=records_ingested,
                execution_time=execution_time,
                dataset_id=dataset_id,
                errors=errors,
                metadata={"file_path": file_path},
            )
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            self.log_event("error", f"Failed to ingest file {file_path}: {str(e)}")
            
            return IngestionResult(
                status=IngestionStatus.FAILURE,
                records_processed=0,
                records_ingested=0,
                execution_time=execution_time,
                dataset_id=dataset_id,
                errors=[str(e)],
                metadata={"file_path": file_path},
            )
            
    def _load_data_from_file(self, file_path: str) -> Any:
        """Load data from a file.
        
        Args:
            file_path: Path to the file to load.
            
        Returns:
            Loaded data.
            
        Raises:
            ValueError: If the file could not be loaded.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == ".csv":
                return pd.read_csv(file_path)
            elif file_ext == ".json":
                with open(file_path, "r") as f:
                    return json.load(f)
            elif file_ext == ".npy":
                return np.load(file_path)
            elif file_ext == ".npz":
                return np.load(file_path)
            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                return np.array(Image.open(file_path))
            elif file_ext == ".pt" or file_ext == ".pth":
                return torch.load(file_path)
            elif file_ext == ".parquet":
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {str(e)}")
            
    def _extract_tensors(
        self,
        data: Any,
        tensor_mapping: Dict[str, str],
    ) -> Dict[str, np.ndarray]:
        """Extract tensors from data.
        
        Args:
            data: Data to extract tensors from.
            tensor_mapping: Mapping from data columns/keys to tensor names.
            
        Returns:
            Dictionary of tensor names to tensor data.
            
        Raises:
            ValueError: If tensors could not be extracted.
        """
        tensors = {}
        
        try:
            if isinstance(data, pd.DataFrame):
                # Extract tensors from DataFrame
                for column, tensor_name in tensor_mapping.items():
                    if column in data.columns:
                        tensors[tensor_name] = np.array(data[column])
            elif isinstance(data, dict):
                # Extract tensors from dictionary
                for key, tensor_name in tensor_mapping.items():
                    if key in data:
                        tensors[tensor_name] = np.array(data[key])
            elif isinstance(data, np.ndarray):
                # Single tensor
                tensor_name = next(iter(tensor_mapping.values()), "data")
                tensors[tensor_name] = data
            elif isinstance(data, torch.Tensor):
                # Single tensor
                tensor_name = next(iter(tensor_mapping.values()), "data")
                tensors[tensor_name] = data.detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
            return tensors
        except Exception as e:
            raise ValueError(f"Failed to extract tensors: {str(e)}")
            
    def _create_tensor_from_data(
        self,
        dataset_id: str,
        tensor_name: str,
        data: np.ndarray,
    ) -> str:
        """Create a tensor from data.
        
        Args:
            dataset_id: ID of the dataset to create the tensor in.
            tensor_name: Name of the tensor to create.
            data: Data to initialize the tensor with.
            
        Returns:
            ID of the created tensor.
            
        Raises:
            ValueError: If the tensor could not be created.
        """
        # Determine shape
        shape = list(data.shape)
        
        # First dimension is variable
        shape[0] = None
        
        # Determine dtype
        dtype_map = {
            np.dtype("float32"): "float32",
            np.dtype("float64"): "float64",
            np.dtype("int32"): "int32",
            np.dtype("int64"): "int64",
            np.dtype("uint8"): "uint8",
            np.dtype("bool"): "bool",
        }
        dtype = dtype_map.get(data.dtype, "float32")
        
        # Create tensor
        return self.storage_client.create_tensor(
            dataset_id=dataset_id,
            tensor_name=tensor_name,
            shape=shape,
            dtype=dtype,
        )
        
    def _scan_watch_paths(self) -> List[Tuple[str, Dict[str, str]]]:
        """Scan watch paths for new files.
        
        Returns:
            List of tuples of (file path, tensor mapping).
        """
        new_files = []
        
        for watch_path in self.watch_paths:
            # Check if it's a directory or a file pattern
            if os.path.isdir(watch_path):
                # It's a directory, scan all files
                for root, _, files in os.walk(watch_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file_path not in self.processed_files:
                            # Infer dataset ID and tensor mapping from file
                            dataset_id, tensor_mapping = self._infer_dataset_and_tensors(file_path)
                            if dataset_id:
                                new_files.append((file_path, dataset_id, tensor_mapping))
            else:
                # It's a file pattern, check if it exists
                if os.path.exists(watch_path) and watch_path not in self.processed_files:
                    # Infer dataset ID and tensor mapping from file
                    dataset_id, tensor_mapping = self._infer_dataset_and_tensors(watch_path)
                    if dataset_id:
                        new_files.append((watch_path, dataset_id, tensor_mapping))
                        
        return new_files
        
    def _infer_dataset_and_tensors(self, file_path: str) -> Tuple[str, Dict[str, str]]:
        """Infer dataset ID and tensor mapping from a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Tuple of (dataset ID, tensor mapping).
        """
        # Extract file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Use parent directory name as dataset ID
        parent_dir = os.path.basename(os.path.dirname(file_path))
        
        # If parent_dir is empty or root, use file_name as dataset ID
        dataset_id = parent_dir if parent_dir else file_name
        
        # Initialize tensor mapping
        tensor_mapping = {}
        
        # Try to infer tensor mapping from file content
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == ".csv":
                # For CSV, use each column as a tensor
                df = pd.read_csv(file_path, nrows=1)
                for column in df.columns:
                    tensor_mapping[column] = column
            elif file_ext == ".json":
                # For JSON, use top-level keys as tensors
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key in data.keys():
                        tensor_mapping[key] = key
            elif file_ext in [".npy", ".npz", ".jpg", ".jpeg", ".png", ".bmp", ".pt", ".pth"]:
                # For numpy arrays and images, use file name as tensor name
                tensor_mapping["data"] = file_name
        except Exception:
            # If we can't infer tensor mapping, use a default
            tensor_mapping["data"] = file_name
            
        return dataset_id, tensor_mapping
        
    def poll_watch_paths(self) -> List[IngestionResult]:
        """Poll watch paths for new files and ingest them.
        
        Returns:
            List of ingestion results.
        """
        # Update last poll time
        self.last_poll_time = time.time()
        
        # Scan watch paths
        new_files = self._scan_watch_paths()
        
        # Ingest new files
        results = []
        for file_path, dataset_id, tensor_mapping in new_files:
            # Get transform function if available
            transform_function = self.transform_functions.get(dataset_id)
            
            # Ingest file
            result = self._ingest_file(
                file_path=file_path,
                dataset_id=dataset_id,
                tensor_mapping=tensor_mapping,
                transform_function=transform_function,
            )
            
            results.append(result)
            
        return results
        
    def run(self, continuous: bool = False) -> List[IngestionResult]:
        """Run the ingestion agent.
        
        Args:
            continuous: Whether to run continuously or just once.
            
        Returns:
            List of ingestion results.
        """
        if continuous:
            # Run continuously
            try:
                while True:
                    results = self.poll_watch_paths()
                    # Sleep until next poll
                    time.sleep(max(0, self.polling_interval - (time.time() - self.last_poll_time)))
            except KeyboardInterrupt:
                self.log_event("info", "Ingestion agent stopped")
                return []
        else:
            # Run once
            return self.poll_watch_paths()
            
    def add_watch_path(self, path: str) -> None:
        """Add a path to watch.
        
        Args:
            path: Path to watch.
        """
        if path not in self.watch_paths:
            self.watch_paths.append(path)
            
    def remove_watch_path(self, path: str) -> bool:
        """Remove a path from watch.
        
        Args:
            path: Path to remove.
            
        Returns:
            True if the path was removed, False otherwise.
        """
        if path in self.watch_paths:
            self.watch_paths.remove(path)
            return True
        return False
        
    def add_transform_function(self, dataset_id: str, transform_function: Callable) -> None:
        """Add a transform function for a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            transform_function: Function for transforming data before ingestion.
        """
        self.transform_functions[dataset_id] = transform_function
        
    def remove_transform_function(self, dataset_id: str) -> bool:
        """Remove a transform function for a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            True if the function was removed, False otherwise.
        """
        if dataset_id in self.transform_functions:
            del self.transform_functions[dataset_id]
            return True
        return False
        
    def clear_processed_files(self) -> None:
        """Clear the list of processed files."""
        self.processed_files = []
        
    def status(self) -> Dict[str, Any]:
        """Get the status of the agent.
        
        Returns:
            Agent status.
        """
        status = super().status()
        status.update({
            "watch_paths": self.watch_paths,
            "polling_interval": self.polling_interval,
            "processed_files_count": len(self.processed_files),
            "last_poll_time": self.last_poll_time,
        })
        return status 