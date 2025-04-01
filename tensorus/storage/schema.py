"""
Schemas for Tensorus datasets and tensors.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator


class DType(str, Enum):
    """Supported tensor data types."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOL = "bool"
    STRING = "string"
    BYTES = "bytes"
    
    # Special types for specific domains
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    EMBEDDING = "embedding"


class TensorSchema(BaseModel):
    """Schema definition for a tensor."""
    shape: List[Optional[int]] = Field(
        ..., description="Shape of the tensor. Use None for variable dimensions."
    )
    dtype: DType = Field(..., description="Data type of the tensor.")
    compression: Optional[str] = Field(
        "default", description="Compression algorithm to use."
    )
    chunks: Optional[List[int]] = Field(
        None, description="Chunk sizes for each dimension. If None, auto-determined."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the tensor."
    )
    
    @validator("shape")
    def validate_shape(cls, v):
        """Validate that the shape is valid."""
        if not v:
            raise ValueError("Shape cannot be empty")
        return v
    
    @validator("chunks")
    def validate_chunks(cls, v, values):
        """Validate that chunks match the shape dimensions."""
        if v is not None:
            shape = values.get("shape", [])
            if len(v) != len(shape):
                raise ValueError("Chunks must have same dimensionality as shape")
        return v


class DatasetSchema(BaseModel):
    """Schema definition for a dataset, consisting of multiple tensors."""
    name: str = Field(..., description="Name of the dataset.")
    description: Optional[str] = Field(None, description="Description of the dataset.")
    version: str = Field("1.0.0", description="Dataset version.")
    tensors: Dict[str, TensorSchema] = Field(
        ..., description="Dictionary of tensor names to their schemas."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the dataset."
    )
    
    @validator("name")
    def validate_name(cls, v):
        """Validate that the name is valid."""
        if not v or not v.isalnum():
            raise ValueError("Name must be alphanumeric")
        return v
    
    @validator("tensors")
    def validate_tensors(cls, v):
        """Validate that there is at least one tensor."""
        if not v:
            raise ValueError("Dataset must contain at least one tensor")
        return v
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "name": "image_dataset",
                "description": "Dataset of images and labels",
                "version": "1.0.0",
                "tensors": {
                    "images": {
                        "shape": [None, 3, 224, 224],
                        "dtype": "float32",
                        "compression": "lz4",
                        "chunks": [1, 3, 224, 224],
                        "metadata": {"domain": "computer_vision"}
                    },
                    "labels": {
                        "shape": [None],
                        "dtype": "int64",
                        "metadata": {"classes": ["cat", "dog", "bird"]}
                    }
                },
                "metadata": {
                    "author": "Tensorus Team",
                    "created_at": "2023-08-01"
                }
            }
        } 