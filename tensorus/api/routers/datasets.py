"""
Datasets router for Tensorus API.

This router handles dataset and tensor operations.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile, status
from pydantic import BaseModel, Field

from tensorus.api.routers.security import get_current_active_user
from tensorus.storage.client import TensorusStorageClient
from tensorus.storage.schema import DatasetSchema, TensorSchema


# Create router
router = APIRouter()


# Models for request and response
class CreateDatasetRequest(BaseModel):
    """Request model for creating a dataset."""
    name: str = Field(..., description="Name of the dataset.")
    description: Optional[str] = Field(None, description="Description of the dataset.")
    version: str = Field("1.0.0", description="Version of the dataset.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the dataset.")


class CreateDatasetResponse(BaseModel):
    """Response model for creating a dataset."""
    id: str = Field(..., description="ID of the created dataset.")
    name: str = Field(..., description="Name of the dataset.")
    description: Optional[str] = Field(None, description="Description of the dataset.")
    version: str = Field(..., description="Version of the dataset.")


class CreateTensorRequest(BaseModel):
    """Request model for creating a tensor."""
    name: str = Field(..., description="Name of the tensor.")
    shape: List[Optional[int]] = Field(..., description="Shape of the tensor.")
    dtype: str = Field(..., description="Data type of the tensor.")
    compression: Optional[str] = Field(None, description="Compression algorithm to use.")
    chunks: Optional[List[int]] = Field(None, description="Chunk sizes for each dimension.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the tensor.")


class CreateTensorResponse(BaseModel):
    """Response model for creating a tensor."""
    id: str = Field(..., description="ID of the created tensor.")
    name: str = Field(..., description="Name of the tensor.")
    dataset_id: str = Field(..., description="ID of the dataset.")
    shape: List[Optional[int]] = Field(..., description="Shape of the tensor.")
    dtype: str = Field(..., description="Data type of the tensor.")


class AppendTensorDataResponse(BaseModel):
    """Response model for appending data to a tensor."""
    id: str = Field(..., description="ID of the tensor.")
    name: str = Field(..., description="Name of the tensor.")
    version: int = Field(..., description="Version of the tensor.")
    size_bytes: int = Field(..., description="Size of the tensor in bytes.")
    num_chunks: int = Field(..., description="Number of chunks the tensor is divided into.")
    updated_at: float = Field(..., description="Last update timestamp.")


# Dependency for getting the storage client
def get_storage_client() -> TensorusStorageClient:
    """Get the storage client.
    
    Returns:
        TensorusStorageClient instance.
    """
    # In a real application, this would be instantiated and configured properly
    # Here, we create a new client each time, which is not efficient
    # This would typically be set up in the app.py file
    return TensorusStorageClient(
        storage_path="/tmp/tensorus",
        backend_type="local",
    )


@router.post("/create", response_model=CreateDatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    request: CreateDatasetRequest,
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> CreateDatasetResponse:
    """Create a new dataset.
    
    Args:
        request: Request data.
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        Created dataset information.
        
    Raises:
        HTTPException: If the dataset could not be created.
    """
    try:
        dataset_id = storage_client.create_dataset(
            name=request.name,
            description=request.description,
            version=request.version,
            metadata=request.metadata,
        )
        
        # Get dataset info
        dataset_info = storage_client.get_dataset_info(dataset_id)
        
        return CreateDatasetResponse(
            id=dataset_id,
            name=request.name,
            description=request.description,
            version=request.version,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dataset: {str(e)}",
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str = Path(..., description="ID of the dataset."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> None:
    """Delete a dataset.
    
    Args:
        dataset_id: ID of the dataset.
        storage_client: Storage client.
        user: Current user.
        
    Raises:
        HTTPException: If the dataset could not be deleted.
    """
    try:
        success = storage_client.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}",
        )


@router.get("/", response_model=List[Dict[str, Any]])
async def list_datasets(
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """List all datasets.
    
    Args:
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        List of dataset summaries.
        
    Raises:
        HTTPException: If the datasets could not be listed.
    """
    try:
        return storage_client.list_datasets()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}",
        )


@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset_info(
    dataset_id: str = Path(..., description="ID of the dataset."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get information about a dataset.
    
    Args:
        dataset_id: ID of the dataset.
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        Dataset information.
        
    Raises:
        HTTPException: If the dataset could not be found.
    """
    try:
        dataset_info = storage_client.get_dataset_info(dataset_id)
        if dataset_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found",
            )
        return dataset_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset info: {str(e)}",
        )


@router.post("/{dataset_id}/tensors/create", response_model=CreateTensorResponse, status_code=status.HTTP_201_CREATED)
async def create_tensor(
    request: CreateTensorRequest,
    dataset_id: str = Path(..., description="ID of the dataset."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> CreateTensorResponse:
    """Create a new tensor in a dataset.
    
    Args:
        request: Request data.
        dataset_id: ID of the dataset.
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        Created tensor information.
        
    Raises:
        HTTPException: If the tensor could not be created.
    """
    try:
        tensor_id = storage_client.create_tensor(
            dataset_id=dataset_id,
            tensor_name=request.name,
            shape=request.shape,
            dtype=request.dtype,
            compression=request.compression,
            chunks=request.chunks,
            metadata=request.metadata,
        )
        
        return CreateTensorResponse(
            id=tensor_id,
            name=request.name,
            dataset_id=dataset_id,
            shape=request.shape,
            dtype=request.dtype,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tensor: {str(e)}",
        )


@router.delete("/{dataset_id}/tensors/{tensor_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tensor(
    dataset_id: str = Path(..., description="ID of the dataset."),
    tensor_name: str = Path(..., description="Name of the tensor."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> None:
    """Delete a tensor from a dataset.
    
    Args:
        dataset_id: ID of the dataset.
        tensor_name: Name of the tensor.
        storage_client: Storage client.
        user: Current user.
        
    Raises:
        HTTPException: If the tensor could not be deleted.
    """
    try:
        success = storage_client.delete_tensor(dataset_id, tensor_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tensor {tensor_name} not found in dataset {dataset_id}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete tensor: {str(e)}",
        )


@router.get("/{dataset_id}/tensors", response_model=List[str])
async def list_tensors(
    dataset_id: str = Path(..., description="ID of the dataset."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> List[str]:
    """List all tensors in a dataset.
    
    Args:
        dataset_id: ID of the dataset.
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        List of tensor names.
        
    Raises:
        HTTPException: If the tensors could not be listed.
    """
    try:
        return storage_client.list_tensors(dataset_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tensors: {str(e)}",
        )


@router.get("/{dataset_id}/tensors/{tensor_name}", response_model=Dict[str, Any])
async def get_tensor_info(
    dataset_id: str = Path(..., description="ID of the dataset."),
    tensor_name: str = Path(..., description="Name of the tensor."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get information about a tensor.
    
    Args:
        dataset_id: ID of the dataset.
        tensor_name: Name of the tensor.
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        Tensor information.
        
    Raises:
        HTTPException: If the tensor could not be found.
    """
    try:
        tensor_info = storage_client.get_tensor_info(dataset_id, tensor_name)
        if tensor_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tensor {tensor_name} not found in dataset {dataset_id}",
            )
        return tensor_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get tensor info: {str(e)}",
        )


@router.post("/{dataset_id}/tensors/{tensor_name}/append", response_model=AppendTensorDataResponse)
async def append_tensor_data(
    dataset_id: str = Path(..., description="ID of the dataset."),
    tensor_name: str = Path(..., description="Name of the tensor."),
    file: UploadFile = File(..., description="Numpy array file (.npy) to append."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> AppendTensorDataResponse:
    """Append data to a tensor.
    
    Args:
        dataset_id: ID of the dataset.
        tensor_name: Name of the tensor.
        file: Numpy array file (.npy) to append.
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        Updated tensor metadata.
        
    Raises:
        HTTPException: If the data could not be appended.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Convert to numpy array
        data = np.load(content)
        
        # Append data
        metadata = storage_client.append_tensor_data(dataset_id, tensor_name, data)
        
        return AppendTensorDataResponse(
            id=metadata["id"],
            name=metadata["name"],
            version=metadata["version"],
            size_bytes=metadata["size_bytes"],
            num_chunks=metadata["num_chunks"],
            updated_at=metadata["updated_at"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to append data: {str(e)}",
        )


@router.get("/{dataset_id}/tensors/{tensor_name}/data")
async def get_tensor_data(
    dataset_id: str = Path(..., description="ID of the dataset."),
    tensor_name: str = Path(..., description="Name of the tensor."),
    start: Optional[int] = Query(None, description="Start index (for variable dimension)."),
    end: Optional[int] = Query(None, description="End index (for variable dimension)."),
    indices: Optional[List[int]] = Query(None, description="Specific indices to retrieve (for variable dimension)."),
    storage_client: TensorusStorageClient = Depends(get_storage_client),
    user = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get data from a tensor.
    
    Args:
        dataset_id: ID of the dataset.
        tensor_name: Name of the tensor.
        start: Start index (for variable dimension).
        end: End index (for variable dimension).
        indices: Specific indices to retrieve (for variable dimension).
        storage_client: Storage client.
        user: Current user.
        
    Returns:
        Tensor data.
        
    Raises:
        HTTPException: If the data could not be retrieved.
    """
    try:
        # Get tensor data
        data = storage_client.get_tensor_data(
            dataset_id=dataset_id,
            tensor_name=tensor_name,
            start=start,
            end=end,
            indices=indices,
        )
        
        # Convert to list for JSON serialization
        if isinstance(data, np.ndarray):
            data_list = data.tolist()
        else:
            data_list = data.numpy().tolist()
            
        return {"data": data_list}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get tensor data: {str(e)}",
        ) 