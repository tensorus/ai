# Tensorus: Agentic Tensor Database/Data Lake

Tensorus is a powerful, AI-driven tensor database and data lake system designed for efficient storage, retrieval, and analysis of tensor data.

## Features

- **Tensor Storage Engine**: Efficiently store and retrieve multi-dimensional tensor data with support for various backends and compression methods.
- **Natural Query Language (NQL)**: Query tensor data using natural language, powered by advanced language models.
- **Autonomous Agents**: A collection of agents for data ingestion, query optimization, and storage management.
- **Versioning**: Built-in version control for datasets and tensors.
- **Dashboard UI**: Interactive dashboard for monitoring and managing tensor data.
- **RESTful API**: Programmatic access to all Tensorus functionality.

## Components

- **Storage Engine**: Core engine for tensor storage and retrieval
- **NQL Interface**: Process natural language queries
- **Agents**: Autonomous agents for tensor data management
  - Ingestion Agent: Automatically ingest data from various sources
  - Query Agent: Process and optimize NQL queries
  - Optimizer Agent: Continuously optimize tensor storage
- **API**: RESTful API for Tensorus functionality
- **UI**: Dashboard for monitoring and control

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/tensorus.git
cd tensorus

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
python run_api.py --host 0.0.0.0 --port 8000 --storage-path /path/to/storage
```

### Running the Dashboard

```bash
python run_dashboard.py --storage-path /path/to/storage
```

## Usage Examples

### Creating and Managing Datasets

```python
from tensorus.storage.client import TensorusStorageClient

# Initialize client
client = TensorusStorageClient(storage_path="/path/to/storage")

# Create a dataset
dataset_id = client.create_dataset("my_dataset", metadata={"description": "My first Tensorus dataset"})

# Add tensors to the dataset
import numpy as np
images = np.random.rand(1000, 28, 28)
labels = np.random.randint(0, 10, size=(1000,))

client.add_tensor(dataset_id, "images", images, metadata={"type": "image"})
client.add_tensor(dataset_id, "labels", labels, metadata={"type": "label"})

# Retrieve data
subset_images = client.get_tensor(dataset_id, "images", slice_spec=(slice(0, 10),))
```

### Using NQL Queries

```python
from tensorus.agents.query import NQLAgent

# Initialize the NQL agent
agent = NQLAgent(storage_client=client)

# Execute a natural language query
result = agent.execute_query("Find all images with a label of 7")

# Display results
print(f"Found {len(result.results)} matching items")
print(f"Execution time: {result.execution_time}s")
```

### Using the Optimizer

```python
from tensorus.agents.optimizer import OptimizerAgent

# Initialize the optimizer agent
optimizer = OptimizerAgent(storage_client=client, auto_apply=True)

# Run optimization on a specific dataset
results = optimizer.optimize_dataset("my_dataset")

# Display optimization results
for result in results:
    print(f"Applied {result.action.action_type} to {result.action.tensor_name or 'dataset'}")
    print(f"Success: {result.success}, Execution time: {result.execution_time}s")
```

## Project Structure

```
tensorus/
├── api/                 # API implementation
│   ├── app.py           # FastAPI application
│   └── routers/         # API routers
├── storage/             # Storage implementation
│   ├── engine.py        # Storage engine
│   ├── client.py        # Client for storage engine
│   └── version.py       # Version management
├── agents/              # Autonomous agents
│   ├── ingestion/       # Data ingestion agents
│   ├── query/           # Query processing agents
│   └── optimizer/       # Storage optimization agents
├── ui/                  # User interface
│   └── dashboard.py     # Streamlit dashboard
├── run_api.py           # Script to run the API
└── run_dashboard.py     # Script to run the dashboard
```

## License

[MIT License](LICENSE)

## Contributing

We welcome contributions to Tensorus! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details. 