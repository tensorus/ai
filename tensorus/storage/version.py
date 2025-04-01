"""
Version management for Tensorus datasets and tensors.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from tensorus.storage.metadata import DatasetMetadata, TensorMetadata


class VersionInfo(BaseModel):
    """Information about a specific version."""
    
    version: Union[int, str] = Field(..., description="Version number or string.")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp.")
    created_by: Optional[str] = Field(None, description="User who created this version.")
    description: Optional[str] = Field(None, description="Description of the version.")
    size_bytes: int = Field(default=0, description="Size in bytes of this version.")
    parent_version: Optional[Union[int, str]] = Field(
        None, description="Parent version, if any."
    )
    tags: List[str] = Field(default_factory=list, description="Tags associated with this version.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for this version.")


class VersionManager:
    """Manager for versioning datasets and tensors."""
    
    def __init__(self, base_path: str):
        """Initialize the version manager.
        
        Args:
            base_path: Base path for storing version information.
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def _get_dataset_version_path(self, dataset_id: str) -> str:
        """Get the path for storing dataset version information.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            Path for storing dataset version information.
        """
        path = os.path.join(self.base_path, "datasets", dataset_id, "versions")
        os.makedirs(path, exist_ok=True)
        return path
        
    def _get_tensor_version_path(self, dataset_id: str, tensor_name: str) -> str:
        """Get the path for storing tensor version information.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            Path for storing tensor version information.
        """
        path = os.path.join(self.base_path, "datasets", dataset_id, "tensors", tensor_name, "versions")
        os.makedirs(path, exist_ok=True)
        return path
        
    def create_dataset_version(
        self,
        dataset_metadata: DatasetMetadata,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create a new version of a dataset.
        
        Args:
            dataset_metadata: Metadata of the dataset.
            description: Description of this version.
            created_by: User who created this version.
            tags: Tags for this version.
            custom_metadata: Additional metadata for this version.
            
        Returns:
            Version information.
        """
        # Parse current version and increment
        version_parts = dataset_metadata.version.split(".")
        major, minor, patch = map(int, version_parts)
        new_version = f"{major}.{minor}.{patch + 1}"
        
        # Create version info
        version_info = VersionInfo(
            version=new_version,
            created_at=time.time(),
            created_by=created_by,
            description=description,
            size_bytes=dataset_metadata.size_bytes,
            parent_version=dataset_metadata.version,
            tags=tags or [],
            metadata=custom_metadata or {},
        )
        
        # Save version info
        version_path = self._get_dataset_version_path(dataset_metadata.id)
        with open(os.path.join(version_path, f"{new_version}.json"), "w") as f:
            f.write(version_info.json(indent=2))
            
        return version_info
    
    def create_tensor_version(
        self,
        tensor_metadata: TensorMetadata,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create a new version of a tensor.
        
        Args:
            tensor_metadata: Metadata of the tensor.
            description: Description of this version.
            created_by: User who created this version.
            tags: Tags for this version.
            custom_metadata: Additional metadata for this version.
            
        Returns:
            Version information.
        """
        # Increment version
        new_version = tensor_metadata.version + 1
        
        # Create version info
        version_info = VersionInfo(
            version=new_version,
            created_at=time.time(),
            created_by=created_by,
            description=description,
            size_bytes=tensor_metadata.size_bytes,
            parent_version=tensor_metadata.version,
            tags=tags or [],
            metadata=custom_metadata or {},
        )
        
        # Save version info
        version_path = self._get_tensor_version_path(tensor_metadata.dataset_id, tensor_metadata.name)
        with open(os.path.join(version_path, f"{new_version}.json"), "w") as f:
            f.write(version_info.json(indent=2))
            
        return version_info
    
    def get_dataset_versions(self, dataset_id: str) -> List[VersionInfo]:
        """Get all versions of a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            List of version information.
        """
        version_path = self._get_dataset_version_path(dataset_id)
        versions = []
        
        if os.path.exists(version_path):
            for filename in os.listdir(version_path):
                if filename.endswith(".json"):
                    with open(os.path.join(version_path, filename), "r") as f:
                        version_info = VersionInfo.parse_raw(f.read())
                        versions.append(version_info)
                        
        # Sort by creation time
        versions.sort(key=lambda v: v.created_at)
        return versions
    
    def get_tensor_versions(self, dataset_id: str, tensor_name: str) -> List[VersionInfo]:
        """Get all versions of a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            
        Returns:
            List of version information.
        """
        version_path = self._get_tensor_version_path(dataset_id, tensor_name)
        versions = []
        
        if os.path.exists(version_path):
            for filename in os.listdir(version_path):
                if filename.endswith(".json"):
                    with open(os.path.join(version_path, filename), "r") as f:
                        version_info = VersionInfo.parse_raw(f.read())
                        versions.append(version_info)
                        
        # Sort by version number
        versions.sort(key=lambda v: v.version)
        return versions
    
    def get_dataset_version(self, dataset_id: str, version: str) -> Optional[VersionInfo]:
        """Get a specific version of a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            version: Version to get.
            
        Returns:
            Version information, or None if not found.
        """
        version_path = self._get_dataset_version_path(dataset_id)
        version_file = os.path.join(version_path, f"{version}.json")
        
        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                return VersionInfo.parse_raw(f.read())
                
        return None
    
    def get_tensor_version(self, dataset_id: str, tensor_name: str, version: int) -> Optional[VersionInfo]:
        """Get a specific version of a tensor.
        
        Args:
            dataset_id: ID of the dataset.
            tensor_name: Name of the tensor.
            version: Version to get.
            
        Returns:
            Version information, or None if not found.
        """
        version_path = self._get_tensor_version_path(dataset_id, tensor_name)
        version_file = os.path.join(version_path, f"{version}.json")
        
        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                return VersionInfo.parse_raw(f.read())
                
        return None 