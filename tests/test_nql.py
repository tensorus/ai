"""
Tests for the Tensorus NQL query functionality.
"""

import os
import tempfile
import unittest
from typing import Optional

import numpy as np

from tensorus.storage.client import TensorusStorageClient
from tensorus.agents.query import NQLAgent


class TestNQLQuery(unittest.TestCase):
    """Test case for the Tensorus NQL query functionality."""
    
    def setUp(self) -> None:
        """Set up the test case."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = self.temp_dir.name
        self.client = TensorusStorageClient(storage_path=self.storage_path)
        
        # Create test dataset
        self.dataset_id = self.client.create_dataset(
            name="test_mnist",
            description="Test MNIST dataset for NQL tests",
            metadata={"domain": "computer_vision", "test": True}
        )
        
        # Create test tensors
        self.client.create_tensor(
            dataset_id=self.dataset_id,
            tensor_name="images",
            shape=[None, 28, 28],
            dtype="float32",
            metadata={"content_type": "image"}
        )
        
        self.client.create_tensor(
            dataset_id=self.dataset_id,
            tensor_name="labels",
            shape=[None],
            dtype="int64",
            metadata={"content_type": "label", "classes": list(range(10))}
        )
        
        # Create test data
        num_samples = 100
        images = np.random.rand(num_samples, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int64)
        
        # Make sure we have some samples with label 7 for testing
        labels[0:10] = 7
        
        # Add data to tensors
        self.client.append_tensor_data(self.dataset_id, "images", images)
        self.client.append_tensor_data(self.dataset_id, "labels", labels)
        
        # Initialize NQL agent
        self.nql_agent = NQLAgent(storage_client=self.client)
    
    def tearDown(self) -> None:
        """Tear down the test case."""
        # Delete test dataset
        self.client.delete_dataset(self.dataset_id)
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_basic_query(self) -> None:
        """Test basic NQL query functionality."""
        # Execute a simple query
        result = self.nql_agent.execute_query(
            query="Find all images with label 7",
            dataset_id=self.dataset_id
        )
        
        # Check that the query returned results
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.results)
        
        # Execution time should be positive
        self.assertGreater(result.execution_time, 0)
        
        # We should have found some images
        if isinstance(result.results, dict) and "data" in result.results:
            self.assertGreater(len(result.results["data"]), 0)
    
    def test_query_with_context(self) -> None:
        """Test NQL query with additional context."""
        # Execute a query with context
        result = self.nql_agent.execute_query(
            query="Find images with this specific label",
            dataset_id=self.dataset_id,
            context={"label": 7}
        )
        
        # Check that the query returned results
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.results)
    
    def test_aggregate_query(self) -> None:
        """Test aggregate NQL query functionality."""
        # Execute an aggregate query
        result = self.nql_agent.execute_query(
            query="Count the number of images for each label",
            dataset_id=self.dataset_id
        )
        
        # Check that the query returned results
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.results)
        
        # If the result contains a dictionary with counts, check that it has entries
        if isinstance(result.results, dict) and "counts" in result.results:
            self.assertGreaterEqual(len(result.results["counts"]), 1)
    
    def test_query_history(self) -> None:
        """Test query history functionality."""
        # Execute a query
        self.nql_agent.execute_query(
            query="Find all images with label 7",
            dataset_id=self.dataset_id
        )
        
        # Get query history
        history = self.nql_agent.get_query_history()
        
        # Check that the history contains the query
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["query"], "Find all images with label 7")
        self.assertEqual(history[0]["dataset_id"], self.dataset_id)
        self.assertTrue(history[0]["successful"])


if __name__ == "__main__":
    unittest.main() 