#!/usr/bin/env python3
"""
Script to verify the Tensorus project structure.
"""

import os
import sys
from pathlib import Path


def check_file_exists(path, required=True):
    """Check if a file exists and print status."""
    exists = os.path.isfile(path)
    status = "✅" if exists else "❌" if required else "⚠️"
    print(f"{status} {path}")
    return exists


def check_directory_exists(path, required=True):
    """Check if a directory exists and print status."""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌" if required else "⚠️"
    print(f"{status} {path}")
    return exists


def main():
    """Run the project structure check."""
    project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Checking project structure in: {project_root}\n")
    
    # Check key project files
    print("Checking key project files:")
    files_to_check = {
        "README.md": True,
        "LICENSE": True,
        "setup.py": True,
        "requirements.txt": True,
        ".gitignore": True,
        "Dockerfile": True,
        "docker-compose.yml": True,
        "run_api.py": True,
        "run_dashboard.py": True,
        "run_tests.py": True,
        "CONTRIBUTING.md": True,
    }
    
    for file_path, required in files_to_check.items():
        check_file_exists(os.path.join(project_root, file_path), required)
    
    # Check directory structure
    print("\nChecking directory structure:")
    directories_to_check = {
        "tensorus": True,
        "tensorus/api": True,
        "tensorus/api/routers": True,
        "tensorus/storage": True,
        "tensorus/agents": True,
        "tensorus/agents/ingestion": True,
        "tensorus/agents/query": True,
        "tensorus/agents/optimizer": True,
        "tensorus/ui": True,
        "tests": True,
        "examples": True,
        ".github/workflows": True,
    }
    
    for dir_path, required in directories_to_check.items():
        check_directory_exists(os.path.join(project_root, dir_path), required)
    
    # Check key module files
    print("\nChecking key module files:")
    module_files_to_check = {
        "tensorus/__init__.py": True,
        "tensorus/api/__init__.py": True,
        "tensorus/api/app.py": True,
        "tensorus/api/routers/__init__.py": True,
        "tensorus/api/routers/query.py": True,
        "tensorus/storage/__init__.py": True,
        "tensorus/storage/engine.py": True,
        "tensorus/storage/client.py": True,
        "tensorus/storage/version.py": True,
        "tensorus/agents/__init__.py": True,
        "tensorus/agents/ingestion/__init__.py": True,
        "tensorus/agents/ingestion/ingestion_agent.py": True,
        "tensorus/agents/query/__init__.py": True,
        "tensorus/agents/query/nql_agent.py": True,
        "tensorus/agents/optimizer/__init__.py": True,
        "tensorus/agents/optimizer/optimizer_agent.py": True,
        "tensorus/ui/__init__.py": True,
        "tensorus/ui/dashboard.py": True,
        "tests/__init__.py": True,
        "tests/test_storage.py": True,
        "tests/test_nql.py": True,
        "examples/basic_usage.py": True,
    }
    
    for file_path, required in module_files_to_check.items():
        check_file_exists(os.path.join(project_root, file_path), required)
    
    # Summary
    print("\nProject structure check complete!")


if __name__ == "__main__":
    main() 