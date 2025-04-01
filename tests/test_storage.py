"""
Tests for the Tensorus storage functionality.
"""

import os
import tempfile
import unittest
from typing import Optional

import numpy as np

from tensorus.storage.client import TensorusStorageClient


class TestTensorusStorage(unittest.TestCase):
    """Test case for the Tensorus storage functionality."""
    
    def setUp(self) -> None:
        """Set up the test case."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = self.temp_dir.name
        self.client = TensorusStorageClient(storage_path=self.storage_path)
        
        # Create test dataset
        self.dataset_id = self.client.create_dataset(
            name="test_dataset",
            description="Test dataset for unit tests",
            metadata={"test": True}
        )
        
        # Create test tensors
        self.client.create_tensor(
            dataset_id=self.dataset_id,
            tensor_name="test_tensor_1",
            shape=[None, 10],
            dtype="float32",
        )
        
        self.client.create_tensor(
            dataset_id=self.dataset_id,
            tensor_name="test_tensor_2",
            shape=[None],
            dtype="int64",
        )
    
    def tearDown(self) -> None:
        """Tear down the test case."""
        # Delete test dataset
        self.client.delete_dataset(self.dataset_id)
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_dataset_creation(self) -> None:
        """Test that datasets can be created."""
        # Check that dataset exists
        dataset_info = self.client.get_dataset_info(self.dataset_id)
        self.assertIsNotNone(dataset_info)
        self.assertEqual(dataset_info["name"], "test_dataset")
        self.assertEqual(dataset_info["description"], "Test dataset for unit tests")
        self.assertEqual(dataset_info["num_tensors"], 2)
    
    def test_tensor_creation(self) -> None:
        """Test that tensors can be created."""
        # List tensors
        tensor_names = self.client.list_tensors(self.dataset_id)
        self.assertEqual(len(tensor_names), 2)
        self.assertIn("test_tensor_1", tensor_names)
        self.assertIn("test_tensor_2", tensor_names)
        
        # Get tensor info
        tensor_info = self.client.get_tensor_info(self.dataset_id, "test_tensor_1")
        self.assertIsNotNone(tensor_info)
        self.assertEqual(tensor_info["name"], "test_tensor_1")
        self.assertEqual(tensor_info["shape"], [None, 10])
        self.assertEqual(tensor_info["dtype"], "float32")
    
    def test_tensor_data_operations(self) -> None:
        """Test that data can be added to and retrieved from tensors."""
        # Create test data
        test_data_1 = np.random.rand(5, 10).astype(np.float32)
        test_data_2 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        
        # Add data to tensors
        self.client.append_tensor_data(self.dataset_id, "test_tensor_1", test_data_1)
        self.client.append_tensor_data(self.dataset_id, "test_tensor_2", test_data_2)
        
        # Get tensor data
        retrieved_data_1 = self.client.get_tensor_data(self.dataset_id, "test_tensor_1")
        retrieved_data_2 = self.client.get_tensor_data(self.dataset_id, "test_tensor_2")
        
        # Check data
        self.assertTrue(np.array_equal(test_data_1, retrieved_data_1))
        self.assertTrue(np.array_equal(test_data_2, retrieved_data_2))
        
        # Test slicing
        retrieved_slice_1 = self.client.get_tensor_data(self.dataset_id, "test_tensor_1", start=1, end=3)
        retrieved_slice_2 = self.client.get_tensor_data(self.dataset_id, "test_tensor_2", start=1, end=3)
        
        # Check sliced data
        self.assertTrue(np.array_equal(test_data_1[1:3], retrieved_slice_1))
        self.assertTrue(np.array_equal(test_data_2[1:3], retrieved_slice_2))
    
    def test_version_management(self) -> None:
        """Test the version management functionality."""
        # Add data to tensors
        test_data = np.random.rand(5, 10).astype(np.float32)
        self.client.append_tensor_data(self.dataset_id, "test_tensor_1", test_data)
        
        # Create a new version
        version_info = self.client.create_tensor_version(
            dataset_id=self.dataset_id,
            tensor_name="test_tensor_1",
            description="Test version",
        )
        
        # Check version info
        self.assertEqual(version_info.version, 2)
        self.assertEqual(version_info.description, "Test version")
        
        # Get versions
        versions = self.client.get_tensor_versions(self.dataset_id, "test_tensor_1")
        self.assertEqual(len(versions), 1)  # The initial version isn't stored explicitly
        
        # Get specific version
        version = self.client.get_tensor_version(self.dataset_id, "test_tensor_1", 2)
        self.assertIsNotNone(version)
        self.assertEqual(version["version"], 2)
        self.assertEqual(version["description"], "Test version")
    
    def test_dataset_deletion(self) -> None:
        """Test that datasets can be deleted."""
        # Create a temporary dataset for deletion
        temp_dataset_id = self.client.create_dataset(name="temp_dataset")
        
        # Check that it exists
        dataset_info = self.client.get_dataset_info(temp_dataset_id)
        self.assertIsNotNone(dataset_info)
        
        # Delete it
        success = self.client.delete_dataset(temp_dataset_id)
        self.assertTrue(success)
        
        # Check that it's gone
        dataset_info = self.client.get_dataset_info(temp_dataset_id)
        self.assertIsNone(dataset_info)


if __name__ == "__main__":
    unittest.main() 