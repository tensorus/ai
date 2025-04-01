"""
Main FastAPI application module for Tensorus API.
"""

import os
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tensorus.api.routers import datasets, query, security
from tensorus.storage.client import TensorusStorageClient


def create_app(storage_path: Optional[str] = None, debug: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        storage_path: Path to the storage directory.
        debug: Whether to enable debug mode.
        
    Returns:
        Configured FastAPI application.
    """
    # Create FastAPI app
    app = FastAPI(
        title="Tensorus API",
        description="API for interacting with Tensorus tensor database/data lake",
        version="0.1.0",
        debug=debug,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Determine storage path
    if storage_path is None:
        storage_path = os.environ.get(
            "TENSORUS_STORAGE_PATH", os.path.expanduser("~/.tensorus/storage")
        )
    
    # Create storage client
    storage_client = TensorusStorageClient(
        storage_path=storage_path,
        backend_type=os.environ.get("TENSORUS_BACKEND_TYPE", "local"),
        backend_options={
            "aws_access_key_id": os.environ.get("TENSORUS_AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("TENSORUS_AWS_SECRET_ACCESS_KEY"),
            "endpoint_url": os.environ.get("TENSORUS_ENDPOINT_URL"),
        },
        default_compression=os.environ.get("TENSORUS_DEFAULT_COMPRESSION", "blosc"),
    )
    
    # Dependency for getting the storage client
    def get_storage_client() -> TensorusStorageClient:
        return storage_client
    
    # Add routers
    app.include_router(
        datasets.router,
        prefix="/datasets",
        tags=["datasets"],
        dependencies=[Depends(get_storage_client)],
    )
    app.include_router(
        query.router,
        prefix="/query",
        tags=["query"],
        dependencies=[Depends(get_storage_client)],
    )
    app.include_router(
        security.router,
        prefix="/auth",
        tags=["auth"],
    )
    
    # Root endpoint
    @app.get("/", tags=["default"])
    async def root() -> Dict[str, str]:
        """Root endpoint.
        
        Returns:
            Simple welcome message.
        """
        return {
            "name": "Tensorus API",
            "version": "0.1.0",
            "status": "OK",
        }
    
    # Health check endpoint
    @app.get("/health", tags=["default"])
    async def health() -> Dict[str, str]:
        """Health check endpoint.
        
        Returns:
            Health status.
        """
        return {
            "status": "healthy",
        }
    
    # Custom exception handler for HTTPException
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Custom exception handler for HTTPException.
        
        Args:
            request: Request that caused the exception.
            exc: The exception.
            
        Returns:
            JSON response with error details.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "http_exception",
                }
            },
        )
    
    # Custom exception handler for generic exceptions
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Custom exception handler for generic exceptions.
        
        Args:
            request: Request that caused the exception.
            exc: The exception.
            
        Returns:
            JSON response with error details.
        """
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": str(exc),
                    "type": "server_error",
                }
            },
        )
    
    # Configure OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
            
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Custom schema modifications can go here
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
        
    app.openapi = custom_openapi
    
    return app 