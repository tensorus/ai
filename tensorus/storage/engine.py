"""
Core storage engine for Tensorus.

This module implements the low-level tensor storage functionality,
handling chunking, compression, and backend storage operations.
"""

import json
import os
import shutil
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import zarr
from fsspec import filesystem
from fsspec.implementations.local import LocalFileSystem

from tensorus.storage.metadata import DatasetMetadata, TensorMetadata
from tensorus.storage.schema import DatasetSchema, DType, TensorSchema


class BackendType(str, Enum):
    """Supported backend storage types."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class CompressionType(str, Enum):
    """Supported compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    BLOSC = "blosc"
    LZ4 = "lz4"
    ZSTD = "zstd"


class ChunkingStrategy(str, Enum):
    """Strategies for chunking tensors."""
    AUTO = "auto"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class TensorStorage:
    """Core tensor storage engine."""
    
    def __init__(
        self,
        base_path: str,
        backend_type: BackendType = BackendType.LOCAL,
        backend_options: Optional[Dict[str, Any]] = None,
        default_compression: CompressionType = CompressionType.BLOSC,
        metadata_path: Optional[str] = None,
    ):
        """Initialize the tensor storage engine.
        
        Args:
            base_path: Base path for storing data.
            backend_type: Type of backend storage.
            backend_options: Options for the backend storage.
            default_compression: Default compression algorithm.
            metadata_path: Path for storing metadata (defaults to base_path/metadata).
        """
        self.base_path = base_path
        self.backend_type = backend_type
        self.backend_options = backend_options or {}
        self.default_compression = default_compression
        self.metadata_path = metadata_path or os.path.join(base_path, "metadata")
        
        # Initialize filesystem
        self.fs = self._init_filesystem()
        
        # Create base directories
        self._ensure_directories()
        
        # Cache for dataset metadata
        self._dataset_metadata_cache: Dict[str, DatasetMetadata] = {}
        
    def _init_filesystem(self) -> Any:
        """Initialize the appropriate filesystem backend.
        
        Returns:
            Filesystem object.
        """
        if self.backend_type == BackendType.LOCAL:
            return LocalFileSystem()
        elif self.backend_type == BackendType.S3:
            import s3fs
            aws_access_key = self.backend_options.get("aws_access_key_id")
            aws_secret_key = self.backend_options.get("aws_secret_access_key")
            endpoint_url = self.backend_options.get("endpoint_url")
            
            return s3fs.S3FileSystem(
                key=aws_access_key,
                secret=aws_secret_key,
                endpoint_url=endpoint_url,
                client_kwargs=self.backend_options.get("client_kwargs", {}),
            )
        elif self.backend_type == BackendType.GCS:
            import gcsfs
            return gcsfs.GCSFileSystem(**self.backend_options)
        elif self.backend_type == BackendType.AZURE:
            import adlfs
            account_name = self.backend_options.get("account_name")
            account_key = self.backend_options.get("account_key")
            
            return adlfs.AzureBlobFileSystem(
                account_name=account_name,
                account_key=account_key,
                **{k: v for k, v in self.backend_options.items() if k not in ["account_name", "account_key"]},
            )
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")
            
    def _ensure_directories(self) -> None:
        """Ensure that the necessary directories exist."""
        if self.backend_type == BackendType.LOCAL:
            os.makedirs(self.base_path, exist_ok=True)
            os.makedirs(self.metadata_path, exist_ok=True)
            os.makedirs(os.path.join(self.base_path, "datasets"), exist_ok=True)
            
    def _get_dataset_path(self, dataset_id: str) -> str:
        """Get the path for a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            Path for the dataset.
        """
        return os.path.join(self.base_path, "datasets", dataset_id)
        
    def _get_tensor_path(self, dataset_id: str, tensor_name: str) -> str:
        """Get the path for a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            Path for the tensor.
        """
        return os.path.join(self._get_dataset_path(dataset_id), "tensors", tensor_name)
        
    def _get_metadata_path(self, dataset_id: str) -> str:
        """Get the path for dataset metadata.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            Path for the dataset metadata.
        """
        return os.path.join(self.metadata_path, "datasets", dataset_id, "metadata.json")
        
    def _get_tensor_metadata_path(self, dataset_id: str, tensor_name: str) -> str:
        """Get the path for tensor metadata.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            Path for the tensor metadata.
        """
        return os.path.join(self.metadata_path, "datasets", dataset_id, "tensors", tensor_name, "metadata.json")
        
    def _save_metadata(self, metadata: Union[DatasetMetadata, TensorMetadata]) -> None:
        """Save metadata to disk.
        
        Args:
            metadata: Metadata to save.
        """
        if isinstance(metadata, DatasetMetadata):
            path = self._get_metadata_path(metadata.id)
            # Update cache
            self._dataset_metadata_cache[metadata.id] = metadata
        else:
            path = self._get_tensor_metadata_path(metadata.dataset_id, metadata.name)
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save metadata
        with open(path, "w") as f:
            f.write(metadata.json(indent=2))
            
    def _load_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Load dataset metadata from disk.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            Dataset metadata, or None if not found.
        """
        # Check cache first
        if dataset_id in self._dataset_metadata_cache:
            return self._dataset_metadata_cache[dataset_id]
            
        path = self._get_metadata_path(dataset_id)
        
        if os.path.exists(path):
            with open(path, "r") as f:
                metadata = DatasetMetadata.parse_raw(f.read())
                # Update cache
                self._dataset_metadata_cache[dataset_id] = metadata
                return metadata
                
        return None
        
    def _load_tensor_metadata(self, dataset_id: str, tensor_name: str) -> Optional[TensorMetadata]:
        """Load tensor metadata from disk.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            Tensor metadata, or None if not found.
        """
        path = self._get_tensor_metadata_path(dataset_id, tensor_name)
        
        if os.path.exists(path):
            with open(path, "r") as f:
                return TensorMetadata.parse_raw(f.read())
                
        return None
        
    def _determine_chunks(self, shape: List[Optional[int]], dtype: str) -> List[int]:
        """Determine appropriate chunk sizes for a tensor.
        
        Args:
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
            
        Returns:
            List of chunk sizes for each dimension.
        """
        # Get element size in bytes
        if dtype in ["float32", "int32"]:
            element_size = 4
        elif dtype in ["float64", "int64"]:
            element_size = 8
        elif dtype in ["float16", "int16"]:
            element_size = 2
        elif dtype in ["int8", "uint8", "bool"]:
            element_size = 1
        else:
            # Default for string and other types
            element_size = 4
            
        # Target chunk size: 16MB
        target_chunk_size = 16 * 1024 * 1024
        
        # Fixed dimensions (non-None)
        fixed_dims = [dim for dim in shape if dim is not None]
        fixed_dims_product = 1
        for dim in fixed_dims:
            fixed_dims_product *= dim
            
        # Number of variable dimensions (None)
        num_variable_dims = sum(1 for dim in shape if dim is None)
        
        # Replace None with a reasonable size
        chunks = []
        for dim in shape:
            if dim is None:
                if num_variable_dims == 1:
                    # If there's only one variable dimension, calculate based on target size
                    chunk_size = max(1, target_chunk_size // (fixed_dims_product * element_size))
                    chunks.append(chunk_size)
                else:
                    # For multiple variable dimensions, use a reasonable default
                    chunks.append(1000)
                    num_variable_dims -= 1
            else:
                # For fixed dimensions, use the actual size or a smaller chunk
                chunks.append(min(dim, 100))
                
        return chunks
        
    def _get_zarr_store(self, path: str) -> zarr.storage:
        """Get a Zarr store for the given path.
        
        Args:
            path: Path to the store.
            
        Returns:
            Zarr store.
        """
        if self.backend_type == BackendType.LOCAL:
            return zarr.DirectoryStore(path)
        else:
            # Use fsspec store for cloud backends
            return zarr.storage.FSStore(path, fs=self.fs)
            
    def _create_zarr_array(
        self,
        path: str,
        shape: List[Optional[int]],
        dtype: str,
        chunks: Optional[List[int]] = None,
        compression: Optional[str] = None,
    ) -> zarr.Array:
        """Create a Zarr array.
        
        Args:
            path: Path to the array.
            shape: Shape of the array.
            dtype: Data type of the array.
            chunks: Chunk sizes for each dimension.
            compression: Compression algorithm to use.
            
        Returns:
            Zarr array.
        """
        # Replace None in shape with 0 for initial creation
        initial_shape = [dim if dim is not None else 0 for dim in shape]
        
        # Determine chunks if not provided
        if chunks is None:
            chunks = self._determine_chunks(shape, dtype)
            
        # Determine compression method
        if compression is None or compression == "default":
            compression = self.default_compression
            
        compressor = None
        if compression == CompressionType.NONE:
            compressor = None
        elif compression == CompressionType.ZLIB:
            compressor = zarr.codecs.Zlib(level=1)
        elif compression == CompressionType.BLOSC:
            compressor = zarr.codecs.Blosc(cname="lz4", clevel=5, shuffle=zarr.codecs.Blosc.SHUFFLE)
        elif compression == CompressionType.LZ4:
            compressor = zarr.codecs.Blosc(cname="lz4", clevel=5, shuffle=zarr.codecs.Blosc.SHUFFLE)
        elif compression == CompressionType.ZSTD:
            compressor = zarr.codecs.Blosc(cname="zstd", clevel=1, shuffle=zarr.codecs.Blosc.SHUFFLE)
            
        # Create the store
        store = self._get_zarr_store(path)
        
        # Create array with initial shape
        return zarr.create(
            shape=initial_shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            store=store,
        )
        
    def create_dataset(self, schema: DatasetSchema) -> DatasetMetadata:
        """Create a new dataset.
        
        Args:
            schema: Schema of the dataset.
            
        Returns:
            Metadata of the created dataset.
        """
        # Create dataset metadata
        dataset_metadata = DatasetMetadata(
            name=schema.name,
            description=schema.description,
            schema=schema,
            path=self._get_dataset_path(schema.name),
            version=schema.version,
        )
        
        # Create dataset directory
        dataset_path = self._get_dataset_path(dataset_metadata.id)
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "tensors"), exist_ok=True)
        
        # Save dataset metadata
        self._save_metadata(dataset_metadata)
        
        return dataset_metadata
        
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            True if the dataset was deleted, False otherwise.
        """
        dataset_path = self._get_dataset_path(dataset_id)
        metadata_path = os.path.dirname(self._get_metadata_path(dataset_id))
        
        if os.path.exists(dataset_path):
            if self.backend_type == BackendType.LOCAL:
                shutil.rmtree(dataset_path)
                shutil.rmtree(metadata_path)
            else:
                self.fs.rm(dataset_path, recursive=True)
                self.fs.rm(metadata_path, recursive=True)
                
            # Remove from cache
            if dataset_id in self._dataset_metadata_cache:
                del self._dataset_metadata_cache[dataset_id]
                
            return True
            
        return False
        
    def create_tensor(
        self,
        dataset_id: str,
        tensor_name: str,
        schema: TensorSchema,
    ) -> TensorMetadata:
        """Create a new tensor in a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            schema: Schema of the tensor.
            
        Returns:
            Metadata of the created tensor.
        """
        # Load dataset metadata
        dataset_metadata = self._load_dataset_metadata(dataset_id)
        if dataset_metadata is None:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Check if tensor already exists
        if tensor_name in dataset_metadata.tensors:
            raise ValueError(f"Tensor {tensor_name} already exists in dataset {dataset_id}")
            
        # Create tensor path
        tensor_path = self._get_tensor_path(dataset_id, tensor_name)
        os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
        
        # Create zarr array
        self._create_zarr_array(
            path=tensor_path,
            shape=schema.shape,
            dtype=str(schema.dtype.value),
            chunks=schema.chunks,
            compression=schema.compression,
        )
        
        # Create tensor metadata
        tensor_metadata = TensorMetadata(
            name=tensor_name,
            dataset_id=dataset_id,
            schema=schema,
            path=tensor_path,
        )
        
        # Save tensor metadata
        self._save_metadata(tensor_metadata)
        
        # Update dataset metadata
        dataset_metadata.add_tensor(tensor_metadata)
        self._save_metadata(dataset_metadata)
        
        return tensor_metadata
        
    def delete_tensor(self, dataset_id: str, tensor_name: str) -> bool:
        """Delete a tensor from a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            True if the tensor was deleted, False otherwise.
        """
        # Load dataset metadata
        dataset_metadata = self._load_dataset_metadata(dataset_id)
        if dataset_metadata is None:
            return False
            
        # Check if tensor exists
        if tensor_name not in dataset_metadata.tensors:
            return False
            
        # Delete tensor data
        tensor_path = self._get_tensor_path(dataset_id, tensor_name)
        metadata_path = os.path.dirname(self._get_tensor_metadata_path(dataset_id, tensor_name))
        
        if os.path.exists(tensor_path):
            if self.backend_type == BackendType.LOCAL:
                shutil.rmtree(tensor_path)
                shutil.rmtree(metadata_path)
            else:
                self.fs.rm(tensor_path, recursive=True)
                self.fs.rm(metadata_path, recursive=True)
                
        # Update dataset metadata
        dataset_metadata.remove_tensor(tensor_name)
        self._save_metadata(dataset_metadata)
        
        return True
        
    def append_tensor_data(
        self,
        dataset_id: str,
        tensor_name: str,
        data: Union[np.ndarray, torch.Tensor],
    ) -> TensorMetadata:
        """Append data to a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            data: Data to append.
            
        Returns:
            Updated tensor metadata.
        """
        # Load dataset metadata
        dataset_metadata = self._load_dataset_metadata(dataset_id)
        if dataset_metadata is None:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Check if tensor exists
        if tensor_name not in dataset_metadata.tensors:
            raise ValueError(f"Tensor {tensor_name} not found in dataset {dataset_id}")
            
        # Load tensor metadata
        tensor_metadata = self._load_tensor_metadata(dataset_id, tensor_name)
        if tensor_metadata is None:
            raise ValueError(f"Tensor {tensor_name} metadata not found")
            
        # Convert torch tensor to numpy if necessary
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        # Open zarr array
        tensor_path = self._get_tensor_path(dataset_id, tensor_name)
        store = self._get_zarr_store(tensor_path)
        array = zarr.open(store)
        
        # Get current size
        variable_dim_index = next((i for i, dim in enumerate(tensor_metadata.schema.shape) if dim is None), None)
        if variable_dim_index is None:
            raise ValueError(f"Tensor {tensor_name} has no variable dimension for appending")
            
        current_size = array.shape[variable_dim_index]
        new_size = current_size + data.shape[0]
        
        # Resize array
        new_shape = list(array.shape)
        new_shape[variable_dim_index] = new_size
        array.resize(new_shape)
        
        # Write data
        index_slice = tuple(slice(current_size, new_size) if i == variable_dim_index else slice(None) for i in range(len(array.shape)))
        array[index_slice] = data
        
        # Update tensor metadata
        tensor_metadata.size_bytes = array.nbytes
        tensor_metadata.num_chunks = array.nchunks
        tensor_metadata.update_version()
        self._save_metadata(tensor_metadata)
        
        # Update dataset metadata
        dataset_metadata.update_tensor(tensor_metadata)
        dataset_metadata.num_records = new_size
        self._save_metadata(dataset_metadata)
        
        return tensor_metadata
        
    def get_tensor_data(
        self,
        dataset_id: str,
        tensor_name: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        indices: Optional[List[int]] = None,
        to_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Get data from a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            start: Start index (for variable dimension).
            end: End index (for variable dimension).
            indices: Specific indices to retrieve (for variable dimension).
            to_torch: Whether to return a torch Tensor instead of a numpy array.
            
        Returns:
            Tensor data.
        """
        # Load dataset metadata
        dataset_metadata = self._load_dataset_metadata(dataset_id)
        if dataset_metadata is None:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Check if tensor exists
        if tensor_name not in dataset_metadata.tensors:
            raise ValueError(f"Tensor {tensor_name} not found in dataset {dataset_id}")
            
        # Load tensor metadata
        tensor_metadata = self._load_tensor_metadata(dataset_id, tensor_name)
        if tensor_metadata is None:
            raise ValueError(f"Tensor {tensor_name} metadata not found")
            
        # Update access stats
        tensor_metadata.update_access_stats()
        dataset_metadata.update_access_stats()
        self._save_metadata(tensor_metadata)
        self._save_metadata(dataset_metadata)
        
        # Open zarr array
        tensor_path = self._get_tensor_path(dataset_id, tensor_name)
        store = self._get_zarr_store(tensor_path)
        array = zarr.open(store)
        
        # Get variable dimension index
        variable_dim_index = next((i for i, dim in enumerate(tensor_metadata.schema.shape) if dim is None), None)
        
        # Prepare slice
        if indices is not None:
            # Convert indices to a numpy array for fancy indexing
            indices = np.array(indices)
            index_slice = tuple(indices if i == variable_dim_index else slice(None) for i in range(len(array.shape)))
        else:
            # Create a slice from start to end
            if start is None:
                start = 0
            if end is None:
                end = array.shape[variable_dim_index]
                
            index_slice = tuple(slice(start, end) if i == variable_dim_index else slice(None) for i in range(len(array.shape)))
            
        # Get data
        data = array[index_slice]
        
        # Convert to torch if requested
        if to_torch:
            return torch.from_numpy(data)
            
        return data 