"""
Basic usage example for Tensorus.

This script demonstrates basic operations with the Tensorus system:
1. Creating datasets and tensors
2. Adding data to tensors
3. Retrieving data from tensors
4. Running NQL queries
5. Using the optimizer agent
"""

import os
import time
import numpy as np

from tensorus.storage.client import TensorusStorageClient
from tensorus.agents.query import NQLAgent
from tensorus.agents.optimizer import OptimizerAgent


def main():
    """Run the example."""
    # Set up storage path
    storage_path = os.path.expanduser("~/.tensorus/examples")
    os.makedirs(storage_path, exist_ok=True)
    
    print(f"Using storage path: {storage_path}")
    
    # Initialize client
    client = TensorusStorageClient(storage_path=storage_path)
    
    # Step 1: Create a dataset
    print("\n--- Creating dataset ---")
    dataset_id = client.create_dataset(
        name="mnist_example",
        description="Example MNIST dataset",
        metadata={"domain": "computer_vision", "example": True}
    )
    print(f"Created dataset with ID: {dataset_id}")
    
    # Step 2: Create tensors in the dataset
    print("\n--- Creating tensors ---")
    
    # Create images tensor
    client.create_tensor(
        dataset_id=dataset_id,
        tensor_name="images",
        shape=[None, 28, 28],
        dtype="float32",
        compression="lz4",
        metadata={"content": "handwritten_digits"}
    )
    print("Created 'images' tensor")
    
    # Create labels tensor
    client.create_tensor(
        dataset_id=dataset_id,
        tensor_name="labels",
        shape=[None],
        dtype="int64",
        metadata={"content": "digit_labels"}
    )
    print("Created 'labels' tensor")
    
    # Step 3: Generate and add some example data
    print("\n--- Adding data to tensors ---")
    
    # Generate random images and labels
    num_samples = 1000
    images = np.random.rand(num_samples, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int64)
    
    # Add data to tensors
    client.append_tensor_data(dataset_id, "images", images)
    client.append_tensor_data(dataset_id, "labels", labels)
    print(f"Added {num_samples} samples to the dataset")
    
    # Step 4: Retrieve data from tensors
    print("\n--- Retrieving data from tensors ---")
    
    # Retrieve the first 5 images
    retrieved_images = client.get_tensor_data(dataset_id, "images", start=0, end=5)
    print(f"Retrieved images shape: {retrieved_images.shape}")
    
    # Retrieve the corresponding labels
    retrieved_labels = client.get_tensor_data(dataset_id, "labels", start=0, end=5)
    print(f"Retrieved labels: {retrieved_labels}")
    
    # Step 5: Use the NQL agent for queries
    print("\n--- Running NQL queries ---")
    
    # Initialize NQL agent
    nql_agent = NQLAgent(storage_client=client)
    
    # Run a simple query
    query_result = nql_agent.execute_query(
        query="Find all images with label 7",
        dataset_id=dataset_id
    )
    
    print(f"Query execution time: {query_result.execution_time:.2f} seconds")
    print(f"Number of results: {len(query_result.results) if hasattr(query_result.results, '__len__') else 'N/A'}")
    if query_result.explanation:
        print(f"Explanation: {query_result.explanation}")
    
    # Step 6: Use the optimizer agent
    print("\n--- Running optimizer ---")
    
    # Initialize optimizer agent
    optimizer = OptimizerAgent(
        storage_client=client,
        auto_apply=True
    )
    
    # Get optimization opportunities
    opportunities = optimizer.get_optimization_opportunities(dataset_id)
    print(f"Found {len(opportunities)} optimization opportunities")
    
    # Print opportunities
    for i, opportunity in enumerate(opportunities):
        print(f"Opportunity {i+1}: {opportunity.action_type} for {opportunity.tensor_name or 'dataset'}")
        print(f"  Reason: {opportunity.reason}")
        print(f"  Estimated impact: {opportunity.estimated_impact}")
    
    # Run optimizations
    results = optimizer.optimize_dataset(dataset_id)
    print(f"Applied {len(results)} optimizations")
    
    # Step 7: Clean up
    print("\n--- Cleaning up ---")
    client.delete_dataset(dataset_id)
    print(f"Deleted dataset {dataset_id}")


if __name__ == "__main__":
    main() 