#!/usr/bin/env python3
"""
Tensorus Quick Start Example

This script demonstrates the basic usage of Tensorus in just a few lines of code.
It's designed to be a minimal example that shows the core functionality.
"""

import os
import numpy as np
from tensorus import TensorusStorageClient, NQLAgent

# Set up storage path
storage_path = os.path.expanduser("~/.tensorus/quickstart")
os.makedirs(storage_path, exist_ok=True)
print(f"Using storage path: {storage_path}")

# Initialize client
client = TensorusStorageClient(storage_path=storage_path)

# Create a dataset
dataset_id = client.create_dataset(
    name="quickstart",
    description="Quick start example dataset"
)
print(f"Created dataset with ID: {dataset_id}")

# Create a tensor for images
client.create_tensor(
    dataset_id=dataset_id,
    tensor_name="images",
    shape=[None, 28, 28],
    dtype="float32"
)

# Create a tensor for labels
client.create_tensor(
    dataset_id=dataset_id,
    tensor_name="labels",
    shape=[None],
    dtype="int64"
)
print("Created tensors: 'images' and 'labels'")

# Generate and add some example data
num_samples = 100
images = np.random.rand(num_samples, 28, 28).astype(np.float32)
labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int64)

# Add data to tensors
client.append_tensor_data(dataset_id, "images", images)
client.append_tensor_data(dataset_id, "labels", labels)
print(f"Added {num_samples} samples to the dataset")

# Retrieve some data
retrieved_images = client.get_tensor_data(dataset_id, "images", start=0, end=5)
retrieved_labels = client.get_tensor_data(dataset_id, "labels", start=0, end=5)
print(f"Retrieved data shapes: {retrieved_images.shape}, {retrieved_labels.shape}")
print(f"Sample labels: {retrieved_labels}")

# Use the NQL agent
nql_agent = NQLAgent(storage_client=client)
result = nql_agent.execute_query(
    query="Find images with label 7",
    dataset_id=dataset_id
)
print(f"Query execution time: {result.execution_time:.2f} seconds")
print(f"Explanation: {result.explanation}")

# Clean up
if input("Delete the dataset? (y/n): ").lower() == 'y':
    client.delete_dataset(dataset_id)
    print(f"Deleted dataset {dataset_id}")
else:
    print(f"Dataset {dataset_id} preserved at {storage_path}")

print("\nTensorus quick start complete!")
print("For more examples, see the 'examples' directory.")
print("For documentation, visit https://github.com/tensorus/foundation") 