"""
Optimizer agent for Tensorus.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from tensorus.agents import Agent
from tensorus.storage.client import TensorusStorageClient
from tensorus.storage.engine import BackendType, CompressionType, TensorStorage


class OptimizationActionType(str, Enum):
    """Type of optimization action."""
    RECOMPRESSION = "recompression"
    RECHUNKING = "rechunking"
    TENSOR_CACHING = "tensor_caching"
    DATASET_COMPACTION = "dataset_compaction"
    INDEX_CREATION = "index_creation"
    INDEX_RECREATION = "index_recreation"
    PARTITION_ADJUSTMENT = "partition_adjustment"
    STATISTICS_UPDATE = "statistics_update"


@dataclass
class OptimizationAction:
    """Description of an optimization action."""
    action_type: OptimizationActionType
    dataset_id: str
    tensor_name: Optional[str]
    parameters: Dict[str, Any]
    estimated_impact: Dict[str, Any]
    reason: str


@dataclass
class OptimizationResult:
    """Result of an optimization action."""
    action: OptimizationAction
    success: bool
    execution_time: float
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    error_message: Optional[str] = None


class OptimizerAgent(Agent):
    """Agent for autonomously optimizing Tensorus datasets."""
    
    def __init__(
        self,
        storage_client: TensorusStorageClient,
        optimizer_config_path: Optional[str] = None,
        execution_interval: int = 3600,  # 1 hour
        monitor_datasets: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        auto_apply: bool = False,
    ):
        """Initialize the optimizer agent.
        
        Args:
            storage_client: Storage client for interacting with the Tensorus storage engine.
            optimizer_config_path: Path to optimizer configuration file.
            execution_interval: Interval in seconds between optimization runs.
            monitor_datasets: List of dataset IDs to monitor (None means all).
            logger: Logger for the agent.
            auto_apply: Whether to automatically apply optimizations.
        """
        super().__init__(
            storage_client=storage_client,
            name="optimizer_agent",
            description="Agent for autonomously optimizing Tensorus datasets.",
            logger=logger,
        )
        
        self.optimizer_config_path = optimizer_config_path
        self.execution_interval = execution_interval
        self.monitor_datasets = monitor_datasets
        self.auto_apply = auto_apply
        
        # Load optimizer configuration
        self.config = self._load_optimizer_config()
        
        # Optimization history
        self.history: List[OptimizationResult] = []
        
        # Last optimization time
        self.last_optimization_time = 0
        
    def _load_optimizer_config(self) -> Dict[str, Any]:
        """Load optimizer configuration.
        
        Returns:
            Optimizer configuration.
        """
        default_config = {
            "thresholds": {
                "compression_ratio": 0.7,  # Threshold for recompression
                "chunk_utilization": 0.5,  # Threshold for rechunking
                "access_frequency": 10,    # Access frequency for caching
                "index_size_ratio": 0.3,   # Ratio of index size to data size
            },
            "actions": {
                "recompression": {
                    "enabled": True,
                    "target_compression": "lz4",  # Target compression algorithm
                },
                "rechunking": {
                    "enabled": True,
                    "min_chunk_size": 1024 * 1024,  # 1 MB
                    "max_chunk_size": 32 * 1024 * 1024,  # 32 MB
                },
                "tensor_caching": {
                    "enabled": True,
                    "max_cache_size": 1024 * 1024 * 1024,  # 1 GB
                },
                "dataset_compaction": {
                    "enabled": True,
                    "min_fragmentation": 0.3,  # Minimum fragmentation for compaction
                },
                "index_creation": {
                    "enabled": True,
                    "indexed_dimensions": [0],  # Dimensions to index
                },
                "index_recreation": {
                    "enabled": True,
                    "min_size_increase": 0.5,  # Minimum size increase for recreation
                },
                "partition_adjustment": {
                    "enabled": True,
                    "max_partition_size": 1024 * 1024 * 1024,  # 1 GB
                },
                "statistics_update": {
                    "enabled": True,
                    "update_interval": 24 * 3600,  # 24 hours
                },
            },
        }
        
        # If no config path is provided, use default config
        if not self.optimizer_config_path:
            return default_config
            
        # Try to load config from file
        try:
            with open(self.optimizer_config_path, "r") as f:
                config = json.load(f)
                
            # Merge with default config
            for section in default_config:
                if section not in config:
                    config[section] = default_config[section]
                elif isinstance(default_config[section], dict):
                    for key in default_config[section]:
                        if key not in config[section]:
                            config[section][key] = default_config[section][key]
                            
            self.log_event("info", f"Loaded optimizer configuration from {self.optimizer_config_path}")
            return config
        except Exception as e:
            self.log_event("error", f"Failed to load optimizer configuration: {str(e)}")
            return default_config
            
    def _save_optimizer_config(self) -> None:
        """Save optimizer configuration to file."""
        if not self.optimizer_config_path:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.optimizer_config_path), exist_ok=True)
            
            # Save config to file
            with open(self.optimizer_config_path, "w") as f:
                json.dump(self.config, f, indent=2)
                
            self.log_event("info", f"Saved optimizer configuration to {self.optimizer_config_path}")
        except Exception as e:
            self.log_event("error", f"Failed to save optimizer configuration: {str(e)}")
            
    def _collect_dataset_metrics(self, dataset_id: str) -> Dict[str, Any]:
        """Collect metrics for a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            Dataset metrics.
            
        Raises:
            ValueError: If the dataset could not be found.
        """
        # Get dataset info
        dataset_info = self.storage_client.get_dataset_info(dataset_id)
        if dataset_info is None:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Collect metrics
        metrics = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_info.get("name"),
            "dataset_size": dataset_info.get("size"),
            "num_tensors": dataset_info.get("num_tensors"),
            "num_records": dataset_info.get("num_records"),
            "created_at": dataset_info.get("created_at"),
            "updated_at": dataset_info.get("updated_at"),
            "access_count": dataset_info.get("access_count", 0),
            "tensors": {},
        }
        
        # Collect tensor metrics
        for tensor_name in self.storage_client.list_tensors(dataset_id):
            tensor_info = self.storage_client.get_tensor_info(dataset_id, tensor_name)
            if tensor_info is not None:
                metrics["tensors"][tensor_name] = {
                    "shape": tensor_info.get("shape"),
                    "dtype": tensor_info.get("dtype"),
                    "compression": tensor_info.get("compression"),
                    "chunks": tensor_info.get("chunks"),
                    "size_bytes": tensor_info.get("size_bytes"),
                    "num_chunks": tensor_info.get("num_chunks"),
                    "access_count": tensor_info.get("access_count", 0),
                }
                
        return metrics
        
    def _identify_optimization_opportunities(
        self,
        dataset_metrics: Dict[str, Any],
    ) -> List[OptimizationAction]:
        """Identify optimization opportunities for a dataset.
        
        Args:
            dataset_metrics: Metrics for the dataset.
            
        Returns:
            List of optimization actions.
        """
        opportunities = []
        dataset_id = dataset_metrics["dataset_id"]
        
        # Thresholds
        thresholds = self.config["thresholds"]
        
        # Check each tensor for optimization opportunities
        for tensor_name, tensor_metrics in dataset_metrics["tensors"].items():
            # Recompression
            if self.config["actions"]["recompression"]["enabled"]:
                compression = tensor_metrics.get("compression")
                if compression != self.config["actions"]["recompression"]["target_compression"]:
                    opportunities.append(
                        OptimizationAction(
                            action_type=OptimizationActionType.RECOMPRESSION,
                            dataset_id=dataset_id,
                            tensor_name=tensor_name,
                            parameters={
                                "new_compression": self.config["actions"]["recompression"]["target_compression"],
                            },
                            estimated_impact={
                                "size_reduction": "20-30%",
                                "query_performance": "5-10% improvement",
                            },
                            reason=f"Current compression ({compression}) is not optimal.",
                        )
                    )
                    
            # Rechunking
            if self.config["actions"]["rechunking"]["enabled"]:
                chunks = tensor_metrics.get("chunks")
                if chunks:
                    # This is a simplified check - in a real implementation,
                    # we would analyze access patterns and determine optimal chunk sizes
                    chunk_size = np.prod(chunks) * 4  # Assume 4 bytes per element
                    if (
                        chunk_size < self.config["actions"]["rechunking"]["min_chunk_size"]
                        or chunk_size > self.config["actions"]["rechunking"]["max_chunk_size"]
                    ):
                        opportunities.append(
                            OptimizationAction(
                                action_type=OptimizationActionType.RECHUNKING,
                                dataset_id=dataset_id,
                                tensor_name=tensor_name,
                                parameters={
                                    "new_chunks": self._calculate_optimal_chunks(tensor_metrics),
                                },
                                estimated_impact={
                                    "query_performance": "20-40% improvement",
                                    "storage_efficiency": "5-15% improvement",
                                },
                                reason="Current chunk size is not optimal for access patterns.",
                            )
                        )
                        
            # Tensor caching
            if self.config["actions"]["tensor_caching"]["enabled"]:
                access_count = tensor_metrics.get("access_count", 0)
                if access_count > thresholds["access_frequency"]:
                    opportunities.append(
                        OptimizationAction(
                            action_type=OptimizationActionType.TENSOR_CACHING,
                            dataset_id=dataset_id,
                            tensor_name=tensor_name,
                            parameters={
                                "cache_priority": access_count / tensor_metrics.get("size_bytes", 1),
                            },
                            estimated_impact={
                                "query_performance": "50-90% improvement",
                            },
                            reason=f"High access frequency ({access_count} accesses).",
                        )
                    )
                    
        # Dataset-level optimizations
        
        # Dataset compaction
        if self.config["actions"]["dataset_compaction"]["enabled"]:
            # This is a simplified check - in a real implementation,
            # we would analyze fragmentation
            fragmentation = 0.4  # Dummy value
            if fragmentation > thresholds["compression_ratio"]:
                opportunities.append(
                    OptimizationAction(
                        action_type=OptimizationActionType.DATASET_COMPACTION,
                        dataset_id=dataset_id,
                        tensor_name=None,
                        parameters={},
                        estimated_impact={
                            "size_reduction": "10-20%",
                            "query_performance": "5-10% improvement",
                        },
                        reason=f"High fragmentation ({fragmentation:.2f}).",
                    )
                )
                
        # Statistics update
        if self.config["actions"]["statistics_update"]["enabled"]:
            # Check last statistics update time
            last_update = dataset_metrics.get("updated_at")
            if last_update:
                time_since_update = time.time() - last_update
                if time_since_update > self.config["actions"]["statistics_update"]["update_interval"]:
                    opportunities.append(
                        OptimizationAction(
                            action_type=OptimizationActionType.STATISTICS_UPDATE,
                            dataset_id=dataset_id,
                            tensor_name=None,
                            parameters={},
                            estimated_impact={
                                "query_optimization": "10-20% improvement",
                            },
                            reason=f"Statistics are outdated ({time_since_update / 3600:.1f} hours since last update).",
                        )
                    )
                    
        return opportunities
        
    def _calculate_optimal_chunks(self, tensor_metrics: Dict[str, Any]) -> List[int]:
        """Calculate optimal chunk sizes for a tensor.
        
        Args:
            tensor_metrics: Metrics for the tensor.
            
        Returns:
            List of optimal chunk sizes.
        """
        # This is a simplified implementation
        # In a real implementation, we would analyze access patterns and data distribution
        
        shape = tensor_metrics.get("shape", [])
        dtype = tensor_metrics.get("dtype", "float32")
        
        # Element size in bytes
        element_size = 4  # Default to 4 bytes (float32)
        if dtype == "float64":
            element_size = 8
        elif dtype in ["float16", "int16"]:
            element_size = 2
        elif dtype in ["int8", "uint8", "bool"]:
            element_size = 1
            
        # Target chunk size: 16MB
        target_chunk_size = 16 * 1024 * 1024
        
        # Calculate optimal chunks
        chunks = []
        for dim in shape:
            if dim is None:
                # Variable dimension, use a reasonable default
                chunks.append(1000)
            else:
                # Fixed dimension, use actual size or smaller
                chunks.append(min(dim, 100))
                
        return chunks
        
    def _apply_optimization(self, action: OptimizationAction) -> OptimizationResult:
        """Apply an optimization action.
        
        Args:
            action: Optimization action to apply.
            
        Returns:
            Result of the optimization.
        """
        start_time = time.time()
        
        try:
            # Log start of optimization
            self.log_event("info", f"Applying optimization: {action.action_type} to dataset {action.dataset_id}", metadata={
                "tensor_name": action.tensor_name,
                "parameters": action.parameters,
            })
            
            # Collect metrics before optimization
            metrics_before = {}
            if action.tensor_name:
                tensor_info = self.storage_client.get_tensor_info(action.dataset_id, action.tensor_name)
                if tensor_info:
                    metrics_before = tensor_info
            else:
                dataset_info = self.storage_client.get_dataset_info(action.dataset_id)
                if dataset_info:
                    metrics_before = dataset_info
                    
            # Apply optimization based on action type
            if action.action_type == OptimizationActionType.RECOMPRESSION:
                self._apply_recompression(action)
            elif action.action_type == OptimizationActionType.RECHUNKING:
                self._apply_rechunking(action)
            elif action.action_type == OptimizationActionType.TENSOR_CACHING:
                self._apply_tensor_caching(action)
            elif action.action_type == OptimizationActionType.DATASET_COMPACTION:
                self._apply_dataset_compaction(action)
            elif action.action_type == OptimizationActionType.INDEX_CREATION:
                self._apply_index_creation(action)
            elif action.action_type == OptimizationActionType.INDEX_RECREATION:
                self._apply_index_recreation(action)
            elif action.action_type == OptimizationActionType.PARTITION_ADJUSTMENT:
                self._apply_partition_adjustment(action)
            elif action.action_type == OptimizationActionType.STATISTICS_UPDATE:
                self._apply_statistics_update(action)
            else:
                raise ValueError(f"Unsupported action type: {action.action_type}")
                
            # Collect metrics after optimization
            metrics_after = {}
            if action.tensor_name:
                tensor_info = self.storage_client.get_tensor_info(action.dataset_id, action.tensor_name)
                if tensor_info:
                    metrics_after = tensor_info
            else:
                dataset_info = self.storage_client.get_dataset_info(action.dataset_id)
                if dataset_info:
                    metrics_after = dataset_info
                    
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log completion
            self.log_event("info", f"Optimization completed successfully in {execution_time:.2f} seconds")
            
            return OptimizationResult(
                action=action,
                success=True,
                execution_time=execution_time,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
            )
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log error
            self.log_event("error", f"Failed to apply optimization: {str(e)}")
            
            return OptimizationResult(
                action=action,
                success=False,
                execution_time=execution_time,
                metrics_before=metrics_before,
                metrics_after={},
                error_message=str(e),
            )
            
    def _apply_recompression(self, action: OptimizationAction) -> None:
        """Apply recompression optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would recompress the tensor data
        # For now, we just log that it would be done
        self.log_event("info", f"Would recompress tensor {action.tensor_name} with {action.parameters.get('new_compression')}")
        
        # Here's a sketch of what this might do:
        # 1. Get the current tensor data
        # 2. Create a new zarr array with the new compression
        # 3. Write the data to the new array
        # 4. Update the tensor metadata
        
    def _apply_rechunking(self, action: OptimizationAction) -> None:
        """Apply rechunking optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would rechunk the tensor data
        # For now, we just log that it would be done
        self.log_event("info", f"Would rechunk tensor {action.tensor_name} with {action.parameters.get('new_chunks')}")
        
    def _apply_tensor_caching(self, action: OptimizationAction) -> None:
        """Apply tensor caching optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would set up caching for the tensor
        # For now, we just log that it would be done
        self.log_event("info", f"Would cache tensor {action.tensor_name} with priority {action.parameters.get('cache_priority')}")
        
    def _apply_dataset_compaction(self, action: OptimizationAction) -> None:
        """Apply dataset compaction optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would compact the dataset
        # For now, we just log that it would be done
        self.log_event("info", f"Would compact dataset {action.dataset_id}")
        
    def _apply_index_creation(self, action: OptimizationAction) -> None:
        """Apply index creation optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would create an index for the tensor
        # For now, we just log that it would be done
        self.log_event("info", f"Would create index for tensor {action.tensor_name}")
        
    def _apply_index_recreation(self, action: OptimizationAction) -> None:
        """Apply index recreation optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would recreate an index for the tensor
        # For now, we just log that it would be done
        self.log_event("info", f"Would recreate index for tensor {action.tensor_name}")
        
    def _apply_partition_adjustment(self, action: OptimizationAction) -> None:
        """Apply partition adjustment optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would adjust partitioning for the dataset
        # For now, we just log that it would be done
        self.log_event("info", f"Would adjust partitioning for dataset {action.dataset_id}")
        
    def _apply_statistics_update(self, action: OptimizationAction) -> None:
        """Apply statistics update optimization.
        
        Args:
            action: Optimization action.
            
        Raises:
            ValueError: If the optimization could not be applied.
        """
        # In a real implementation, this would update statistics for the dataset
        # For now, we just log that it would be done
        self.log_event("info", f"Would update statistics for dataset {action.dataset_id}")
        
    def optimize_dataset(self, dataset_id: str) -> List[OptimizationResult]:
        """Optimize a dataset.
        
        Args:
            dataset_id: ID of the dataset to optimize.
            
        Returns:
            List of optimization results.
        """
        try:
            # Collect dataset metrics
            dataset_metrics = self._collect_dataset_metrics(dataset_id)
            
            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities(dataset_metrics)
            
            # Apply optimizations
            results = []
            for opportunity in opportunities:
                if self.auto_apply:
                    # Apply optimization
                    result = self._apply_optimization(opportunity)
                    results.append(result)
                    
                    # Add to history
                    self.history.append(result)
                else:
                    # Log opportunity
                    self.log_event("info", f"Identified optimization opportunity: {opportunity.action_type}", metadata={
                        "dataset_id": opportunity.dataset_id,
                        "tensor_name": opportunity.tensor_name,
                        "reason": opportunity.reason,
                    })
                    
            return results
        except Exception as e:
            self.log_event("error", f"Failed to optimize dataset {dataset_id}: {str(e)}")
            return []
            
    def get_optimization_opportunities(self, dataset_id: str) -> List[OptimizationAction]:
        """Get optimization opportunities for a dataset.
        
        Args:
            dataset_id: ID of the dataset.
            
        Returns:
            List of optimization opportunities.
        """
        try:
            # Collect dataset metrics
            dataset_metrics = self._collect_dataset_metrics(dataset_id)
            
            # Identify optimization opportunities
            return self._identify_optimization_opportunities(dataset_metrics)
        except Exception as e:
            self.log_event("error", f"Failed to get optimization opportunities for dataset {dataset_id}: {str(e)}")
            return []
            
    def run(self, dataset_ids: Optional[List[str]] = None) -> List[OptimizationResult]:
        """Run the optimizer agent.
        
        Args:
            dataset_ids: List of dataset IDs to optimize (None means use monitor_datasets).
            
        Returns:
            List of optimization results.
        """
        # Update last optimization time
        self.last_optimization_time = time.time()
        
        # Determine which datasets to optimize
        if dataset_ids is None:
            dataset_ids = self.monitor_datasets
            
        if dataset_ids is None:
            # Optimize all datasets
            dataset_ids = [d.get("id") for d in self.storage_client.list_datasets()]
            
        # Optimize each dataset
        results = []
        for dataset_id in dataset_ids:
            dataset_results = self.optimize_dataset(dataset_id)
            results.extend(dataset_results)
            
        return results
        
    def run_continuous(self) -> None:
        """Run the optimizer agent continuously."""
        try:
            while True:
                # Run optimization
                self.run()
                
                # Sleep until next run
                time.sleep(self.execution_interval)
        except KeyboardInterrupt:
            self.log_event("info", "Optimizer agent stopped")
            
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history.
        
        Returns:
            List of optimization results.
        """
        return self.history
        
    def clear_optimization_history(self) -> None:
        """Clear optimization history."""
        self.history = []
        
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update optimizer configuration.
        
        Args:
            config_updates: Configuration updates.
        """
        # Update config
        for section, section_updates in config_updates.items():
            if section in self.config:
                if isinstance(self.config[section], dict):
                    for key, value in section_updates.items():
                        self.config[section][key] = value
                else:
                    self.config[section] = section_updates
            else:
                self.config[section] = section_updates
                
        # Save config
        self._save_optimizer_config()
        
    def status(self) -> Dict[str, Any]:
        """Get the status of the agent.
        
        Returns:
            Agent status.
        """
        status = super().status()
        status.update({
            "execution_interval": self.execution_interval,
            "monitor_datasets": self.monitor_datasets,
            "auto_apply": self.auto_apply,
            "last_optimization_time": self.last_optimization_time,
            "optimization_history_count": len(self.history),
        })
        return status 