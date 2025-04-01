"""
Storage client for Tensorus.

This module provides a high-level client for interacting with the Tensorus storage engine.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from tensorus.storage.engine import BackendType, CompressionType, TensorStorage
from tensorus.storage.metadata import DatasetMetadata, TensorMetadata
from tensorus.storage.schema import DatasetSchema, DType, TensorSchema
from tensorus.storage.version import VersionInfo, VersionManager


class TensorusStorageClient:
    """High-level client for interacting with the Tensorus storage engine."""
    
    def __init__(
        self,
        storage_path: str,
        backend_type: str = "local",
        backend_options: Optional[Dict[str, Any]] = None,
        default_compression: str = "blosc",
    ):
        """Initialize the storage client.
        
        Args:
            storage_path: Path to the storage directory.
            backend_type: Type of backend storage ("local", "s3", "gcs", or "azure").
            backend_options: Options for the backend storage.
            default_compression: Default compression algorithm.
        """
        self.storage = TensorStorage(
            base_path=storage_path,
            backend_type=BackendType(backend_type),
            backend_options=backend_options,
            default_compression=CompressionType(default_compression),
        )
        
        self.version_manager = VersionManager(
            base_path=storage_path,
        )
        
    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new dataset.
        
        Args:
            name: Name of the dataset.
            description: Description of the dataset.
            version: Version of the dataset.
            metadata: Additional metadata for the dataset.
            
        Returns:
            ID of the created dataset.
        """
        # Create dataset schema
        schema = DatasetSchema(
            name=name,
            description=description,
            version=version,
            tensors={},
            metadata=metadata or {},
        )
        
        # Create dataset
        dataset_metadata = self.storage.create_dataset(schema)
        return dataset_metadata.id
        
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            True if the dataset was deleted, False otherwise.
        """
        return self.storage.delete_dataset(dataset_id)
        
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets.
        
        Returns:
            List of dataset summaries.
        """
        # This is a placeholder implementation
        # In a real implementation, this would scan the metadata directory
        return []
        
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            Dataset information, or None if not found.
        """
        metadata = self.storage._load_dataset_metadata(dataset_id)
        if metadata is None:
            return None
            
        return metadata.to_summary_dict()
        
    def create_tensor(
        self,
        dataset_id: str,
        tensor_name: str,
        shape: List[Optional[int]],
        dtype: str,
        compression: Optional[str] = None,
        chunks: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new tensor in a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
            compression: Compression algorithm to use.
            chunks: Chunk sizes for each dimension.
            metadata: Additional metadata for the tensor.
            
        Returns:
            ID of the created tensor.
        """
        # Create tensor schema
        schema = TensorSchema(
            shape=shape,
            dtype=DType(dtype),
            compression=compression,
            chunks=chunks,
            metadata=metadata or {},
        )
        
        # Create tensor
        tensor_metadata = self.storage.create_tensor(
            dataset_id=dataset_id,
            tensor_name=tensor_name,
            schema=schema,
        )
        
        return tensor_metadata.id
        
    def delete_tensor(self, dataset_id: str, tensor_name: str) -> bool:
        """Delete a tensor from a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            True if the tensor was deleted, False otherwise.
        """
        return self.storage.delete_tensor(
            dataset_id=dataset_id,
            tensor_name=tensor_name,
        )
        
    def list_tensors(self, dataset_id: str) -> List[str]:
        """List all tensors in a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            List of tensor names.
        """
        metadata = self.storage._load_dataset_metadata(dataset_id)
        if metadata is None:
            return []
            
        return list(metadata.get_tensor_names())
        
    def get_tensor_info(self, dataset_id: str, tensor_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            Tensor information, or None if not found.
        """
        metadata = self.storage._load_tensor_metadata(dataset_id, tensor_name)
        if metadata is None:
            return None
            
        return {
            "id": metadata.id,
            "name": metadata.name,
            "dataset_id": metadata.dataset_id,
            "shape": metadata.schema.shape,
            "dtype": metadata.schema.dtype.value,
            "compression": metadata.schema.compression,
            "chunks": metadata.schema.chunks,
            "version": metadata.version,
            "size_bytes": metadata.size_bytes,
            "num_chunks": metadata.num_chunks,
            "created_at": metadata.created_at,
            "updated_at": metadata.updated_at,
            "access_count": metadata.access_count,
            "tags": metadata.tags,
            "metadata": metadata.custom_metadata,
        }
        
    def append_tensor_data(
        self,
        dataset_id: str,
        tensor_name: str,
        data: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """Append data to a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            data: Data to append.
            
        Returns:
            Updated tensor metadata.
        """
        metadata = self.storage.append_tensor_data(
            dataset_id=dataset_id,
            tensor_name=tensor_name,
            data=data,
        )
        
        return {
            "id": metadata.id,
            "name": metadata.name,
            "version": metadata.version,
            "size_bytes": metadata.size_bytes,
            "num_chunks": metadata.num_chunks,
            "updated_at": metadata.updated_at,
        }
        
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
        return self.storage.get_tensor_data(
            dataset_id=dataset_id,
            tensor_name=tensor_name,
            start=start,
            end=end,
            indices=indices,
            to_torch=to_torch,
        )
        
    def create_dataset_version(
        self,
        dataset_id: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create a new version of a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            description: Description of this version.
            created_by: User who created this version.
            tags: Tags for this version.
            custom_metadata: Additional metadata for this version.
            
        Returns:
            Version information.
        """
        # Load dataset metadata
        dataset_metadata = self.storage._load_dataset_metadata(dataset_id)
        if dataset_metadata is None:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Create version
        return self.version_manager.create_dataset_version(
            dataset_metadata=dataset_metadata,
            description=description,
            created_by=created_by,
            tags=tags,
            custom_metadata=custom_metadata,
        )
        
    def create_tensor_version(
        self,
        dataset_id: str,
        tensor_name: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create a new version of a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            description: Description of this version.
            created_by: User who created this version.
            tags: Tags for this version.
            custom_metadata: Additional metadata for this version.
            
        Returns:
            Version information.
        """
        # Load tensor metadata
        tensor_metadata = self.storage._load_tensor_metadata(dataset_id, tensor_name)
        if tensor_metadata is None:
            raise ValueError(f"Tensor {tensor_name} not found in dataset {dataset_id}")
            
        # Create version
        return self.version_manager.create_tensor_version(
            tensor_metadata=tensor_metadata,
            description=description,
            created_by=created_by,
            tags=tags,
            custom_metadata=custom_metadata,
        )
        
    def get_dataset_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            List of version information.
        """
        versions = self.version_manager.get_dataset_versions(dataset_id)
        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "created_by": v.created_by,
                "description": v.description,
                "size_bytes": v.size_bytes,
                "parent_version": v.parent_version,
                "tags": v.tags,
                "metadata": v.metadata,
            }
            for v in versions
        ]
        
    def get_tensor_versions(self, dataset_id: str, tensor_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            List of version information.
        """
        versions = self.version_manager.get_tensor_versions(dataset_id, tensor_name)
        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "created_by": v.created_by,
                "description": v.description,
                "size_bytes": v.size_bytes,
                "parent_version": v.parent_version,
                "tags": v.tags,
                "metadata": v.metadata,
            }
            for v in versions
        ]
        
    def get_dataset_version(self, dataset_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get a specific version of a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            version: Version to get.
            
        Returns:
            Version information, or None if not found.
        """
        v = self.version_manager.get_dataset_version(dataset_id, version)
        if v is None:
            return None
            
        return {
            "version": v.version,
            "created_at": v.created_at,
            "created_by": v.created_by,
            "description": v.description,
            "size_bytes": v.size_bytes,
            "parent_version": v.parent_version,
            "tags": v.tags,
            "metadata": v.metadata,
        }
        
    def get_tensor_version(self, dataset_id: str, tensor_name: str, version: int) -> Optional[Dict[str, Any]]:
        """Get a specific version of a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            version: Version to get.
            
        Returns:
            Version information, or None if not found.
        """
        v = self.version_manager.get_tensor_version(dataset_id, tensor_name, version)
        if v is None:
            return None
            
        return {
            "version": v.version,
            "created_at": v.created_at,
            "created_by": v.created_by,
            "description": v.description,
            "size_bytes": v.size_bytes,
            "parent_version": v.parent_version,
            "tags": v.tags,
            "metadata": v.metadata,
        }