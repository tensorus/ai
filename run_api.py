"""
Run the Tensorus API server.
"""

import argparse
import os
import sys
import uvicorn

from tensorus.api.app import create_app


def main():
    """Run the Tensorus API server."""
    parser = argparse.ArgumentParser(description="Run Tensorus API server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--storage-path",
        default=os.environ.get("TENSORUS_STORAGE_PATH", os.path.expanduser("~/.tensorus/storage")),
        help="Path to the storage directory",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )
    args = parser.parse_args()
    
    print(f"Starting Tensorus API server on http://{args.host}:{args.port}")
    print(f"Storage path: {args.storage_path}")
    
    # Create API app
    app = create_app(storage_path=args.storage_path, debug=args.debug)
    
    # Run the server
    uvicorn.run(
        "tensorus.api.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
        log_level="debug" if args.debug else "info",
        kwargs={"storage_path": args.storage_path, "debug": args.debug},
    )


if __name__ == "__main__":
    main() 