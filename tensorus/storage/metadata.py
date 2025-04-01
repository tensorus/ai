"""
Metadata management for Tensorus datasets and tensors.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from tensorus.storage.schema import DatasetSchema, TensorSchema


class TensorMetadata(BaseModel):
    """Metadata for a tensor."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the tensor.")
    name: str = Field(..., description="Name of the tensor within the dataset.")
    dataset_id: str = Field(..., description="ID of the dataset this tensor belongs to.")
    schema: TensorSchema = Field(..., description="Schema of the tensor.")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp.")
    updated_at: float = Field(default_factory=time.time, description="Last update timestamp.")
    version: int = Field(default=1, description="Version of the tensor.")
    size_bytes: int = Field(default=0, description="Size of the tensor in bytes.")
    num_chunks: int = Field(default=0, description="Number of chunks the tensor is divided into.")
    path: str = Field(..., description="Storage path for the tensor data.")
    access_count: int = Field(default=0, description="Number of times the tensor has been accessed.")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the tensor.")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for the tensor.")
    
    def update_access_stats(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        
    def update_version(self) -> None:
        """Increment the version of the tensor."""
        self.version += 1
        self.updated_at = time.time()


class DatasetMetadata(BaseModel):
    """Metadata for a dataset."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the dataset.")
    name: str = Field(..., description="Name of the dataset.")
    description: Optional[str] = Field(None, description="Description of the dataset.")
    schema: DatasetSchema = Field(..., description="Schema of the dataset.")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp.")
    updated_at: float = Field(default_factory=time.time, description="Last update timestamp.")
    version: str = Field(default="1.0.0", description="Version of the dataset.")
    size_bytes: int = Field(default=0, description="Total size of all tensors in the dataset in bytes.")
    num_tensors: int = Field(default=0, description="Number of tensors in the dataset.")
    num_records: int = Field(default=0, description="Number of records in the dataset.")
    path: str = Field(..., description="Storage path for the dataset.")
    tensors: Dict[str, TensorMetadata] = Field(default_factory=dict, description="Metadata for each tensor in the dataset.")
    access_count: int = Field(default=0, description="Number of times the dataset has been accessed.")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the dataset.")
    owner: Optional[str] = Field(None, description="Owner of the dataset.")
    permissions: Dict[str, List[str]] = Field(
        default_factory=dict, description="Access permissions for the dataset."
    )
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for the dataset.")
    
    def update_access_stats(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.updated_at = time.time()
        
    def add_tensor(self, tensor_metadata: TensorMetadata) -> None:
        """Add a tensor to the dataset."""
        self.tensors[tensor_metadata.name] = tensor_metadata
        self.num_tensors = len(self.tensors)
        self.size_bytes += tensor_metadata.size_bytes
        self.updated_at = time.time()
        
    def remove_tensor(self, tensor_name: str) -> None:
        """Remove a tensor from the dataset."""
        if tensor_name in self.tensors:
            tensor = self.tensors[tensor_name]
            self.size_bytes -= tensor.size_bytes
            del self.tensors[tensor_name]
            self.num_tensors = len(self.tensors)
            self.updated_at = time.time()
            
    def update_tensor(self, tensor_metadata: TensorMetadata) -> None:
        """Update a tensor in the dataset."""
        if tensor_metadata.name in self.tensors:
            old_tensor = self.tensors[tensor_metadata.name]
            self.size_bytes = self.size_bytes - old_tensor.size_bytes + tensor_metadata.size_bytes
            self.tensors[tensor_metadata.name] = tensor_metadata
            self.updated_at = time.time()
            
    def get_tensor_names(self) -> Set[str]:
        """Get the names of all tensors in the dataset."""
        return set(self.tensors.keys())
    
    def human_readable_size(self) -> str:
        """Get a human-readable string for the dataset size."""
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        size = self.size_bytes
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
            
        return f"{size:.2f} {units[unit_index]}"
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "updated_at": datetime.fromtimestamp(self.updated_at).isoformat(),
            "size": self.human_readable_size(),
            "num_tensors": self.num_tensors,
            "num_records": self.num_records,
            "tags": self.tags,
            "owner": self.owner,
        } 