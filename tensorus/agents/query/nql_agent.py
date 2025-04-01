"""
Natural Query Language agent for Tensorus.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tensorus.agents import Agent
from tensorus.storage.client import TensorusStorageClient


@dataclass
class NQLQueryResult:
    """Result of a NQL query."""
    results: Any
    execution_time: float
    dataset_id: Optional[str]
    explanation: Optional[str]


class NQLAgent(Agent):
    """Natural Query Language agent for interpreting natural language queries."""
    
    def __init__(
        self,
        storage_client: TensorusStorageClient,
        model_name: str = "Llama-3-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None,
        query_history_path: Optional[str] = None,
    ):
        """Initialize the NQL agent.
        
        Args:
            storage_client: Storage client for interacting with the Tensorus storage engine.
            model_name: Name of the model to use for NQL.
            device: Device to run the model on.
            logger: Logger for the agent.
            query_history_path: Path to store query history.
        """
        super().__init__(
            storage_client=storage_client,
            name="nql_agent",
            description="Natural Query Language agent for interpreting natural language queries.",
            logger=logger,
        )
        
        self.model_name = model_name
        self.device = device
        self.query_history_path = query_history_path or "/tmp/tensorus/query_history.jsonl"
        self.query_history: List[Dict[str, Any]] = []
        
        # Load query history
        self._load_query_history()
        
        # Initialize model and tokenizer
        self.log_event("info", f"Initializing NQL agent with model {model_name} on {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device,
                load_in_8bit=device == "cuda",
                low_cpu_mem_usage=True,
            )
            self.log_event("info", "NQL agent initialization successful")
        except Exception as e:
            self.log_event("error", f"Failed to initialize NQL agent: {str(e)}")
            raise
            
    def _load_query_history(self) -> None:
        """Load query history from file."""
        try:
            with open(self.query_history_path, "r") as f:
                self.query_history = [json.loads(line) for line in f]
            self.log_event("info", f"Loaded {len(self.query_history)} query history items")
        except FileNotFoundError:
            self.log_event("info", "Query history file not found, starting with empty history")
            self.query_history = []
        except Exception as e:
            self.log_event("error", f"Failed to load query history: {str(e)}")
            self.query_history = []
            
    def _save_query_history(self) -> None:
        """Save query history to file."""
        try:
            with open(self.query_history_path, "w") as f:
                for item in self.query_history:
                    f.write(json.dumps(item) + "\n")
            self.log_event("info", f"Saved {len(self.query_history)} query history items")
        except Exception as e:
            self.log_event("error", f"Failed to save query history: {str(e)}")
            
    def _add_to_query_history(
        self,
        query: str,
        dataset_id: Optional[str],
        execution_time: float,
        successful: bool,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Add a query to the history.
        
        Args:
            query: The query string.
            dataset_id: ID of the dataset that was queried.
            execution_time: Execution time in seconds.
            successful: Whether the query was successful.
            error_message: Error message if the query failed.
            user_id: ID of the user who executed the query.
        """
        history_item = {
            "query": query,
            "dataset_id": dataset_id,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "successful": successful,
            "error_message": error_message,
            "user_id": user_id,
        }
        self.query_history.append(history_item)
        
        # Keep only the last 1000 items
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
            
        # Save to file
        self._save_query_history()
            
    def _generate_nql_prompt(
        self,
        query: str,
        dataset_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a prompt for the NQL model.
        
        Args:
            query: Natural language query.
            dataset_id: ID of the dataset to query (if specific).
            context: Additional context for the query.
            
        Returns:
            Prompt for the NQL model.
        """
        # Start with a system prompt explaining the task
        prompt = (
            "You are a natural language to query converter for the Tensorus database. "
            "Translate the following natural language query into a Tensorus operation. "
            "Your response should be valid JSON that I can execute.\n\n"
        )
        
        # Add information about available datasets
        if dataset_id is not None:
            prompt += f"Dataset ID: {dataset_id}\n"
            
            # Get dataset info
            dataset_info = self.storage_client.get_dataset_info(dataset_id)
            if dataset_info is not None:
                prompt += f"Dataset Name: {dataset_info.get('name')}\n"
                prompt += f"Tensors: {', '.join(self.storage_client.list_tensors(dataset_id))}\n\n"
                
                # Add information about each tensor
                for tensor_name in self.storage_client.list_tensors(dataset_id):
                    tensor_info = self.storage_client.get_tensor_info(dataset_id, tensor_name)
                    if tensor_info is not None:
                        prompt += f"Tensor: {tensor_name}\n"
                        prompt += f"  Shape: {tensor_info.get('shape')}\n"
                        prompt += f"  Dtype: {tensor_info.get('dtype')}\n\n"
        else:
            # List all available datasets
            prompt += "Available Datasets:\n"
            datasets = self.storage_client.list_datasets()
            if datasets:
                for dataset in datasets:
                    prompt += f"- {dataset.get('name')} (ID: {dataset.get('id')})\n"
            else:
                prompt += "No datasets available.\n"
                
        # Add additional context if provided
        if context:
            prompt += "Additional Context:\n"
            for key, value in context.items():
                prompt += f"{key}: {value}\n"
                
        # Add the query
        prompt += "\nQuery: " + query + "\n\n"
        
        # Add instructions for the format of the response
        prompt += (
            "Respond with a JSON object that specifies:\n"
            "1. The operation to perform (e.g., 'filter', 'aggregate', 'join', 'similarity_search')\n"
            "2. The dataset ID to operate on\n"
            "3. The tensors involved in the operation\n"
            "4. Any parameters for the operation (e.g., filter conditions, aggregation functions)\n"
            "5. An explanation of how you interpreted the query\n\n"
            "Example response format:\n"
            "```json\n"
            "{\n"
            '  "operation": "filter",\n'
            '  "dataset_id": "example_dataset_id",\n'
            '  "tensors": ["images", "labels"],\n'
            '  "parameters": {\n'
            '    "condition": {\n'
            '      "tensor": "labels",\n'
            '      "operator": "equals",\n'
            '      "value": 1\n'
            '    }\n'
            '  },\n'
            '  "explanation": "This query is filtering for images where the label is 1."\n'
            "}\n"
            "```\n"
        )
        
        return prompt
        
    def _parse_nql_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the NQL model.
        
        Args:
            response: Response from the NQL model.
            
        Returns:
            Parsed response.
            
        Raises:
            ValueError: If the response could not be parsed.
        """
        # Extract JSON from the response
        try:
            # Look for JSON in code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                # Try to find a JSON object in the response
                start_idx = response.find("{")
                end_idx = response.rfind("}")
                if start_idx >= 0 and end_idx >= 0:
                    json_str = response[start_idx:end_idx + 1]
                else:
                    raise ValueError("Could not find JSON in response")
                    
            # Parse JSON
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse NQL response: {str(e)}")
            
    def _execute_operation(
        self,
        operation: Dict[str, Any],
    ) -> Any:
        """Execute a Tensorus operation.
        
        Args:
            operation: Operation to execute.
            
        Returns:
            Operation results.
            
        Raises:
            ValueError: If the operation could not be executed.
        """
        op_type = operation.get("operation")
        dataset_id = operation.get("dataset_id")
        tensors = operation.get("tensors", [])
        parameters = operation.get("parameters", {})
        
        if not dataset_id:
            raise ValueError("No dataset ID specified in operation")
            
        if not tensors:
            raise ValueError("No tensors specified in operation")
            
        # Execute the operation based on its type
        if op_type == "filter":
            return self._execute_filter_operation(dataset_id, tensors, parameters)
        elif op_type == "aggregate":
            return self._execute_aggregate_operation(dataset_id, tensors, parameters)
        elif op_type == "similarity_search":
            return self._execute_similarity_search_operation(dataset_id, tensors, parameters)
        elif op_type == "join":
            return self._execute_join_operation(dataset_id, tensors, parameters)
        elif op_type == "get":
            return self._execute_get_operation(dataset_id, tensors, parameters)
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")
            
    def _execute_filter_operation(
        self,
        dataset_id: str,
        tensors: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a filter operation.
        
        Args:
            dataset_id: ID of the dataset.
            tensors: Tensors to filter.
            parameters: Filter parameters.
            
        Returns:
            Filter results.
            
        Raises:
            ValueError: If the filter could not be executed.
        """
        # Get the condition
        condition = parameters.get("condition", {})
        if not condition:
            raise ValueError("No filter condition specified")
            
        tensor_name = condition.get("tensor")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if not tensor_name or not operator:
            raise ValueError("Invalid filter condition")
            
        # Get the tensor data
        tensor_data = self.storage_client.get_tensor_data(dataset_id, tensor_name)
        
        # Apply the filter
        if operator == "equals":
            indices = np.where(tensor_data == value)[0]
        elif operator == "not_equals":
            indices = np.where(tensor_data != value)[0]
        elif operator == "greater_than":
            indices = np.where(tensor_data > value)[0]
        elif operator == "less_than":
            indices = np.where(tensor_data < value)[0]
        elif operator == "greater_than_or_equals":
            indices = np.where(tensor_data >= value)[0]
        elif operator == "less_than_or_equals":
            indices = np.where(tensor_data <= value)[0]
        elif operator == "in":
            indices = np.where(np.isin(tensor_data, value))[0]
        elif operator == "not_in":
            indices = np.where(~np.isin(tensor_data, value))[0]
        else:
            raise ValueError(f"Unsupported operator: {operator}")
            
        # Get the filtered data for each tensor
        result = {}
        for t_name in tensors:
            if t_name == tensor_name:
                result[t_name] = tensor_data[indices]
            else:
                t_data = self.storage_client.get_tensor_data(dataset_id, t_name)
                result[t_name] = t_data[indices]
                
        return {
            "filtered_data": result,
            "indices": indices.tolist(),
            "count": len(indices),
        }
        
    def _execute_aggregate_operation(
        self,
        dataset_id: str,
        tensors: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute an aggregate operation.
        
        Args:
            dataset_id: ID of the dataset.
            tensors: Tensors to aggregate.
            parameters: Aggregation parameters.
            
        Returns:
            Aggregation results.
            
        Raises:
            ValueError: If the aggregation could not be executed.
        """
        # Get the aggregation function
        agg_function = parameters.get("function")
        if not agg_function:
            raise ValueError("No aggregation function specified")
            
        # Get the tensor data
        result = {}
        for tensor_name in tensors:
            tensor_data = self.storage_client.get_tensor_data(dataset_id, tensor_name)
            
            # Apply the aggregation function
            if agg_function == "mean":
                result[tensor_name] = float(np.mean(tensor_data))
            elif agg_function == "sum":
                result[tensor_name] = float(np.sum(tensor_data))
            elif agg_function == "min":
                result[tensor_name] = float(np.min(tensor_data))
            elif agg_function == "max":
                result[tensor_name] = float(np.max(tensor_data))
            elif agg_function == "count":
                result[tensor_name] = int(tensor_data.size)
            elif agg_function == "std":
                result[tensor_name] = float(np.std(tensor_data))
            elif agg_function == "var":
                result[tensor_name] = float(np.var(tensor_data))
            elif agg_function == "median":
                result[tensor_name] = float(np.median(tensor_data))
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_function}")
                
        return {
            "aggregated_data": result,
            "function": agg_function,
        }
        
    def _execute_similarity_search_operation(
        self,
        dataset_id: str,
        tensors: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a similarity search operation.
        
        Args:
            dataset_id: ID of the dataset.
            tensors: Tensors to search.
            parameters: Search parameters.
            
        Returns:
            Search results.
            
        Raises:
            ValueError: If the search could not be executed.
        """
        # Get the query vector
        query_vector = parameters.get("query_vector")
        if query_vector is None:
            raise ValueError("No query vector specified")
            
        # Get the number of results to return
        k = parameters.get("k", 5)
        
        # Get the tensor to search
        if len(tensors) != 1:
            raise ValueError("Similarity search requires exactly one tensor")
            
        tensor_name = tensors[0]
        tensor_data = self.storage_client.get_tensor_data(dataset_id, tensor_name)
        
        # Convert query vector to numpy array
        query_vector = np.array(query_vector)
        
        # Calculate cosine similarity
        # Normalize vectors
        tensor_data_norm = tensor_data / np.linalg.norm(tensor_data, axis=1, keepdims=True)
        query_vector_norm = query_vector / np.linalg.norm(query_vector)
        
        # Calculate similarity
        similarity = np.dot(tensor_data_norm, query_vector_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarity)[-k:][::-1]
        top_scores = similarity[top_indices]
        
        return {
            "indices": top_indices.tolist(),
            "scores": top_scores.tolist(),
            "tensor": tensor_name,
        }
        
    def _execute_join_operation(
        self,
        dataset_id: str,
        tensors: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a join operation.
        
        Args:
            dataset_id: ID of the dataset.
            tensors: Tensors to join.
            parameters: Join parameters.
            
        Returns:
            Join results.
            
        Raises:
            ValueError: If the join could not be executed.
        """
        # This is a simplified implementation
        # Get the tensors to join
        if len(tensors) < 2:
            raise ValueError("Join requires at least two tensors")
            
        # Get the join keys
        join_keys = parameters.get("join_keys", {})
        if not join_keys:
            raise ValueError("No join keys specified")
            
        # Get the join type
        join_type = parameters.get("join_type", "inner")
        
        # Get the tensor data
        tensor_data = {}
        for tensor_name in tensors:
            tensor_data[tensor_name] = self.storage_client.get_tensor_data(dataset_id, tensor_name)
            
        # Perform the join (simplified)
        result = {tensor_name: [] for tensor_name in tensors}
        
        # This is just a placeholder - a real implementation would be more complex
        # and depend on the specific join type and keys
        return {
            "joined_data": result,
            "join_type": join_type,
        }
        
    def _execute_get_operation(
        self,
        dataset_id: str,
        tensors: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a get operation.
        
        Args:
            dataset_id: ID of the dataset.
            tensors: Tensors to get.
            parameters: Get parameters.
            
        Returns:
            Get results.
            
        Raises:
            ValueError: If the get could not be executed.
        """
        # Get the indices
        indices = parameters.get("indices")
        start = parameters.get("start")
        end = parameters.get("end")
        
        # Get the tensor data
        result = {}
        for tensor_name in tensors:
            tensor_data = self.storage_client.get_tensor_data(
                dataset_id=dataset_id,
                tensor_name=tensor_name,
                start=start,
                end=end,
                indices=indices,
            )
            
            # Convert to list for JSON serialization
            if isinstance(tensor_data, np.ndarray):
                result[tensor_name] = tensor_data.tolist()
            elif isinstance(tensor_data, torch.Tensor):
                result[tensor_name] = tensor_data.numpy().tolist()
            else:
                result[tensor_name] = tensor_data
                
        return {
            "data": result,
            "tensor_info": {
                tensor_name: self.storage_client.get_tensor_info(dataset_id, tensor_name)
                for tensor_name in tensors
            },
        }
        
    def execute_query(
        self,
        query: str,
        dataset_id: Optional[str] = None,
        timeout: int = 30,
        max_results: int = 100,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> NQLQueryResult:
        """Execute a Natural Query Language (NQL) query.
        
        Args:
            query: Natural language query.
            dataset_id: ID of the dataset to query (if specific).
            timeout: Timeout in seconds.
            max_results: Maximum number of results to return.
            context: Additional context for the query.
            user_id: ID of the user executing the query.
            
        Returns:
            Query results.
            
        Raises:
            ValueError: If the query could not be executed.
            TimeoutError: If the query timed out.
        """
        start_time = time.time()
        
        try:
            # Generate prompt
            prompt = self._generate_nql_prompt(query, dataset_id, context)
            
            # Log query
            self.log_event("info", f"Executing NQL query: {query}", metadata={
                "dataset_id": dataset_id,
                "user_id": user_id,
            })
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response
            operation = self._parse_nql_response(response)
            
            # Set dataset_id if not provided in the query
            if dataset_id is not None and "dataset_id" not in operation:
                operation["dataset_id"] = dataset_id
                
            # Execute operation
            result = self._execute_operation(operation)
            
            # Get execution time
            execution_time = time.time() - start_time
            
            # Add to query history
            self._add_to_query_history(
                query=query,
                dataset_id=operation.get("dataset_id"),
                execution_time=execution_time,
                successful=True,
                user_id=user_id,
            )
            
            # Log success
            self.log_event("info", f"NQL query executed successfully in {execution_time:.2f} seconds", metadata={
                "dataset_id": operation.get("dataset_id"),
                "operation": operation.get("operation"),
                "user_id": user_id,
            })
            
            return NQLQueryResult(
                results=result,
                execution_time=execution_time,
                dataset_id=operation.get("dataset_id"),
                explanation=operation.get("explanation"),
            )
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Add to query history
            self._add_to_query_history(
                query=query,
                dataset_id=dataset_id,
                execution_time=execution_time,
                successful=False,
                error_message=str(e),
                user_id=user_id,
            )
            
            # Log error
            self.log_event("error", f"Failed to execute NQL query: {str(e)}", metadata={
                "dataset_id": dataset_id,
                "user_id": user_id,
                "query": query,
            })
            
            raise ValueError(f"Failed to execute NQL query: {str(e)}")
            
    def get_query_history(
        self,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get query history.
        
        Args:
            limit: Maximum number of history items to return.
            user_id: ID of the user to filter by.
            
        Returns:
            Query history.
        """
        history = self.query_history
        
        # Filter by user ID if provided
        if user_id is not None:
            history = [item for item in history if item.get("user_id") == user_id]
            
        # Sort by timestamp (newest first)
        history = sorted(history, key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Limit number of results
        history = history[:limit]
        
        return history
        
    def run(self, query: str, *args, **kwargs) -> NQLQueryResult:
        """Run the agent with a query.
        
        Args:
            query: Natural language query.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Query results.
        """
        return self.execute_query(query, *args, **kwargs) 