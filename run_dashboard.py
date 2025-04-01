"""
Run the Tensorus Dashboard UI.
"""

import argparse
import os
import sys

from tensorus.ui.dashboard import TensorusDashboard


def main():
    """Run the Tensorus Dashboard UI."""
    parser = argparse.ArgumentParser(description="Run Tensorus Dashboard UI")
    parser.add_argument(
        "--storage-path",
        default=os.environ.get("TENSORUS_STORAGE_PATH", os.path.expanduser("~/.tensorus/storage")),
        help="Path to the storage directory",
    )
    parser.add_argument(
        "--page-title",
        default="Tensorus Dashboard",
        help="Dashboard page title",
    )
    parser.add_argument(
        "--page-icon",
        default="📊",
        help="Dashboard page icon",
    )
    args = parser.parse_args()
    
    print(f"Starting Tensorus Dashboard")
    print(f"Storage path: {args.storage_path}")
    
    # Create and run dashboard
    dashboard = TensorusDashboard(
        storage_path=args.storage_path,
        page_title=args.page_title,
        page_icon=args.page_icon,
    )
    dashboard.run()


if __name__ == "__main__":
    main() 