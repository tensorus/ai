"""
Query router for Tensorus API.

This router handles Natural Query Language (NQL) queries.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from tensorus.api.routers.security import get_current_active_user
from tensorus.agents.query.nql_agent import NQLAgent
from tensorus.storage.client import TensorusStorageClient


# Create router
router = APIRouter()


class NQLQueryRequest(BaseModel):
    """Request model for NQL queries."""
    query: str = Field(..., description="Natural language query.")
    dataset_id: Optional[str] = Field(None, description="ID of the dataset to query (if specific).")
    timeout: Optional[int] = Field(30, description="Timeout in seconds.")
    max_results: Optional[int] = Field(100, description="Maximum number of results to return.")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query.")


class NQLQueryResponse(BaseModel):
    """Response model for NQL queries."""
    results: Any = Field(..., description="Query results.")
    execution_time: float = Field(..., description="Execution time in seconds.")
    dataset_id: Optional[str] = Field(None, description="ID of the dataset that was queried.")
    explanation: Optional[str] = Field(None, description="Explanation of how the query was executed.")


# Dependency for getting the storage client
def get_storage_client() -> TensorusStorageClient:
    """Get the storage client.
    
    Returns:
        TensorusStorageClient instance.
    """
    # In a real application, this would be instantiated and configured properly
    return TensorusStorageClient(
        storage_path="/tmp/tensorus",
        backend_type="local",
    )


# Dependency for getting the NQL agent
def get_nql_agent(storage_client: TensorusStorageClient = Depends(get_storage_client)) -> NQLAgent:
    """Get the NQL agent.
    
    Args:
        storage_client: Storage client.
        
    Returns:
        NQLAgent instance.
    """
    return NQLAgent(storage_client=storage_client)


@router.post("/nql", response_model=NQLQueryResponse)
async def execute_nql_query(
    request: NQLQueryRequest,
    nql_agent: NQLAgent = Depends(get_nql_agent),
    user = Depends(get_current_active_user),
) -> NQLQueryResponse:
    """Execute a Natural Query Language (NQL) query.
    
    Args:
        request: Query request.
        nql_agent: NQL agent.
        user: Current user.
        
    Returns:
        Query results.
        
    Raises:
        HTTPException: If the query could not be executed.
    """
    try:
        # Execute query
        result = nql_agent.execute_query(
            query=request.query,
            dataset_id=request.dataset_id,
            timeout=request.timeout,
            max_results=request.max_results,
            context=request.context,
        )
        
        return NQLQueryResponse(
            results=result.results,
            execution_time=result.execution_time,
            dataset_id=result.dataset_id,
            explanation=result.explanation,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute query: {str(e)}",
        )


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_query_history(
    limit: int = Query(10, description="Maximum number of history items to return."),
    nql_agent: NQLAgent = Depends(get_nql_agent),
    user = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Get query history.
    
    Args:
        limit: Maximum number of history items to return.
        nql_agent: NQL agent.
        user: Current user.
        
    Returns:
        Query history.
        
    Raises:
        HTTPException: If the history could not be retrieved.
    """
    try:
        return nql_agent.get_query_history(limit=limit, user_id=user.username)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get query history: {str(e)}",
        ) 