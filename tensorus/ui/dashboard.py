"""
Tensorus Dashboard UI using Streamlit.
"""

import os
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from tensorus.storage.client import TensorusStorageClient
from tensorus.agents.optimizer import OptimizerAgent


class TensorusDashboard:
    """Tensorus Dashboard UI."""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        page_title: str = "Tensorus Dashboard",
        page_icon: str = "📊",
    ):
        """Initialize the dashboard.
        
        Args:
            storage_path: Path to the storage directory.
            page_title: Title of the dashboard.
            page_icon: Icon for the dashboard.
        """
        self.storage_path = storage_path or os.environ.get(
            "TENSORUS_STORAGE_PATH", 
            os.path.expanduser("~/.tensorus/storage")
        )
        
        # Set up page config
        st.set_page_config(
            page_title=page_title,
            page_icon=page_icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
        # Initialize storage client
        self.storage_client = TensorusStorageClient(storage_path=self.storage_path)
        
        # Initialize optimizer agent
        self.optimizer_agent = OptimizerAgent(
            storage_client=self.storage_client,
            auto_apply=False,
        )
        
        # Set up state
        if "selected_dataset" not in st.session_state:
            st.session_state.selected_dataset = None
        if "selected_tensor" not in st.session_state:
            st.session_state.selected_tensor = None
        if "refresh_counter" not in st.session_state:
            st.session_state.refresh_counter = 0
    
    def _render_sidebar(self) -> None:
        """Render the sidebar."""
        st.sidebar.title("Tensorus")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh"):
            st.session_state.refresh_counter += 1
            
        # Navigation
        st.sidebar.header("Navigation")
        pages = {
            "Overview": self._render_overview_page,
            "Datasets": self._render_datasets_page,
            "Optimization": self._render_optimization_page,
            "Monitoring": self._render_monitoring_page,
            "Settings": self._render_settings_page,
        }
        selected_page = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Render storage info
        st.sidebar.header("Storage Info")
        try:
            storage_info = self.storage_client.get_storage_info()
            total_size = storage_info.get("total_size", 0)
            num_datasets = storage_info.get("num_datasets", 0)
            
            st.sidebar.metric("Storage Used", f"{total_size / (1024**2):.2f} MB")
            st.sidebar.metric("Datasets", num_datasets)
        except Exception as e:
            st.sidebar.error(f"Error loading storage info: {str(e)}")
            
        # Render footer
        st.sidebar.markdown("---")
        st.sidebar.caption("Tensorus Foundation")
        
        # Render selected page
        pages[selected_page]()
    
    def _render_overview_page(self) -> None:
        """Render the overview page."""
        st.title("Overview")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        try:
            # Get storage info
            storage_info = self.storage_client.get_storage_info()
            total_size = storage_info.get("total_size", 0)
            num_datasets = storage_info.get("num_datasets", 0)
            num_tensors = storage_info.get("num_tensors", 0)
            
            # Display metrics
            col1.metric("Total Datasets", num_datasets)
            col2.metric("Total Tensors", num_tensors)
            col3.metric("Total Storage", f"{total_size / (1024**2):.2f} MB")
            
            # Display dataset size distribution
            st.subheader("Dataset Size Distribution")
            
            # Get dataset info
            datasets = self.storage_client.list_datasets()
            
            if not datasets:
                st.info("No datasets found.")
                return
                
            # Create dataframe for visualization
            df = pd.DataFrame([
                {
                    "id": d.get("id"),
                    "name": d.get("name", "Unnamed"),
                    "size_mb": d.get("size", 0) / (1024**2),
                    "num_tensors": d.get("num_tensors", 0),
                }
                for d in datasets
            ])
            
            # Create bar chart
            fig = px.bar(
                df, 
                x="name", 
                y="size_mb",
                color="num_tensors",
                labels={"name": "Dataset", "size_mb": "Size (MB)", "num_tensors": "Number of Tensors"},
                title="Dataset Size Distribution",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recent activity
            st.subheader("Recent Activity")
            
            # This would typically come from an activity log
            # For demonstration, we'll create some dummy activity
            activities = [
                {"time": "2023-09-15 14:32:00", "action": "Dataset created", "details": "Created dataset 'mnist'"},
                {"time": "2023-09-15 14:35:12", "action": "Tensor added", "details": "Added tensor 'images' to dataset 'mnist'"},
                {"time": "2023-09-15 14:36:45", "action": "Tensor added", "details": "Added tensor 'labels' to dataset 'mnist'"},
                {"time": "2023-09-15 15:10:22", "action": "Query executed", "details": "NQL query: 'Find all images with label 7'"},
                {"time": "2023-09-15 15:45:33", "action": "Optimization", "details": "Recompressed tensor 'images' in dataset 'mnist'"},
            ]
            
            activity_df = pd.DataFrame(activities)
            st.dataframe(activity_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering overview: {str(e)}")
    
    def _render_datasets_page(self) -> None:
        """Render the datasets page."""
        st.title("Datasets")
        
        try:
            # List datasets
            datasets = self.storage_client.list_datasets()
            
            if not datasets:
                st.info("No datasets found.")
                st.button("Create Dataset")
                return
                
            # Create dataset selector
            dataset_options = [{"id": d.get("id"), "name": d.get("name", d.get("id"))} for d in datasets]
            selected_dataset_idx = 0
            
            if st.session_state.selected_dataset:
                # Find index of previously selected dataset
                for i, d in enumerate(dataset_options):
                    if d["id"] == st.session_state.selected_dataset:
                        selected_dataset_idx = i
                        break
                        
            selected_dataset = st.selectbox(
                "Select a dataset",
                options=range(len(dataset_options)),
                format_func=lambda i: dataset_options[i]["name"],
                index=selected_dataset_idx,
            )
            
            st.session_state.selected_dataset = dataset_options[selected_dataset]["id"]
            
            # Display dataset info
            dataset_id = dataset_options[selected_dataset]["id"]
            dataset_info = self.storage_client.get_dataset_info(dataset_id)
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Info", "Tensors", "Operations"])
            
            with tab1:
                # Display dataset info
                st.subheader("Dataset Information")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Size", f"{dataset_info.get('size', 0) / (1024**2):.2f} MB")
                col2.metric("Number of Tensors", dataset_info.get("num_tensors", 0))
                col3.metric("Number of Records", dataset_info.get("num_records", 0))
                
                # Display metadata
                st.subheader("Metadata")
                metadata = dataset_info.get("metadata", {})
                st.json(metadata)
                
            with tab2:
                # Display tensor list
                st.subheader("Tensors")
                
                # List tensors
                tensors = self.storage_client.list_tensors(dataset_id)
                
                if not tensors:
                    st.info("No tensors found in this dataset.")
                    return
                    
                # Display tensor info
                tensor_data = []
                for tensor_name in tensors:
                    tensor_info = self.storage_client.get_tensor_info(dataset_id, tensor_name)
                    if tensor_info:
                        tensor_data.append({
                            "name": tensor_name,
                            "shape": str(tensor_info.get("shape", [])),
                            "dtype": tensor_info.get("dtype", "unknown"),
                            "size_mb": tensor_info.get("size_bytes", 0) / (1024**2),
                            "compression": tensor_info.get("compression", "none"),
                        })
                        
                tensor_df = pd.DataFrame(tensor_data)
                st.dataframe(tensor_df, use_container_width=True)
                
                # Select a tensor to view details
                tensor_options = [t["name"] for t in tensor_data]
                selected_tensor_idx = 0
                
                if st.session_state.selected_tensor in tensor_options:
                    selected_tensor_idx = tensor_options.index(st.session_state.selected_tensor)
                    
                selected_tensor = st.selectbox(
                    "Select a tensor for details",
                    options=tensor_options,
                    index=selected_tensor_idx,
                )
                
                st.session_state.selected_tensor = selected_tensor
                
                # Display tensor details
                tensor_info = self.storage_client.get_tensor_info(dataset_id, selected_tensor)
                
                if tensor_info:
                    st.subheader(f"Tensor: {selected_tensor}")
                    
                    # Create columns for tensor details
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Shape", str(tensor_info.get("shape", [])))
                    col1.metric("Data Type", tensor_info.get("dtype", "unknown"))
                    col2.metric("Size", f"{tensor_info.get('size_bytes', 0) / (1024**2):.2f} MB")
                    col2.metric("Chunks", str(tensor_info.get("chunks", [])))
                    col3.metric("Compression", tensor_info.get("compression", "none"))
                    col3.metric("Number of Chunks", tensor_info.get("num_chunks", 0))
                    
                    # Display preview option
                    if st.button("Preview Tensor Data (first 10 elements)"):
                        try:
                            # Get tensor data
                            data = self.storage_client.get_tensor(dataset_id, selected_tensor, slice_spec=(slice(0, 10),))
                            st.write(data)
                        except Exception as e:
                            st.error(f"Error loading tensor data: {str(e)}")
                
            with tab3:
                # Dataset operations
                st.subheader("Operations")
                
                # Operations buttons
                if st.button("Delete Dataset"):
                    st.warning(f"Are you sure you want to delete dataset '{dataset_options[selected_dataset]['name']}'?")
                    if st.button("Confirm Delete"):
                        try:
                            self.storage_client.delete_dataset(dataset_id)
                            st.success(f"Dataset '{dataset_options[selected_dataset]['name']}' deleted.")
                            st.session_state.selected_dataset = None
                        except Exception as e:
                            st.error(f"Error deleting dataset: {str(e)}")
                
                # Export dataset
                if st.button("Export Dataset"):
                    st.info("Export functionality not yet implemented.")
                    
                # Compact dataset
                if st.button("Compact Dataset"):
                    st.info("Compaction functionality not yet implemented.")
                    
        except Exception as e:
            st.error(f"Error rendering datasets: {str(e)}")
    
    def _render_optimization_page(self) -> None:
        """Render the optimization page."""
        st.title("Optimization")
        
        try:
            # Get datasets
            datasets = self.storage_client.list_datasets()
            
            if not datasets:
                st.info("No datasets found to optimize.")
                return
                
            # Dataset selector
            dataset_options = [{"id": d.get("id"), "name": d.get("name", d.get("id"))} for d in datasets]
            
            selected_dataset = st.selectbox(
                "Select a dataset to optimize",
                options=range(len(dataset_options)),
                format_func=lambda i: dataset_options[i]["name"],
            )
            
            dataset_id = dataset_options[selected_dataset]["id"]
            
            # Analyze button
            if st.button("Analyze Optimization Opportunities"):
                # Get optimization opportunities
                opportunities = self.optimizer_agent.get_optimization_opportunities(dataset_id)
                
                if not opportunities:
                    st.success("No optimization opportunities found for this dataset.")
                    return
                    
                st.subheader("Optimization Opportunities")
                
                # Display opportunities
                opportunity_data = []
                for i, opp in enumerate(opportunities):
                    opportunity_data.append({
                        "id": i,
                        "type": opp.action_type,
                        "tensor": opp.tensor_name or "N/A",
                        "reason": opp.reason,
                        "estimated_impact": ", ".join([f"{k}: {v}" for k, v in opp.estimated_impact.items()]),
                    })
                    
                opportunity_df = pd.DataFrame(opportunity_data)
                st.dataframe(opportunity_df, use_container_width=True)
                
                # Apply optimizations
                st.subheader("Apply Optimizations")
                
                # Select optimizations to apply
                selected_optimizations = st.multiselect(
                    "Select optimizations to apply",
                    options=opportunity_data,
                    format_func=lambda x: f"{x['type']} - {x['tensor']} - {x['reason']}",
                )
                
                # Apply button
                if st.button("Apply Selected Optimizations"):
                    st.info("Optimization application not yet implemented.")
            
            # Optimization history
            st.subheader("Optimization History")
            history = self.optimizer_agent.get_optimization_history()
            
            if not history:
                st.info("No optimization history available.")
                return
                
            # Display history
            history_data = []
            for result in history:
                history_data.append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result.execution_time)),
                    "action": result.action.action_type,
                    "dataset": result.action.dataset_id,
                    "tensor": result.action.tensor_name or "N/A",
                    "success": result.success,
                    "execution_time": f"{result.execution_time:.2f}s",
                })
                
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering optimization page: {str(e)}")
    
    def _render_monitoring_page(self) -> None:
        """Render the monitoring page."""
        st.title("Monitoring")
        
        try:
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Usage", "Performance", "Logs"])
            
            with tab1:
                # Usage metrics
                st.subheader("Storage Usage Over Time")
                
                # This would typically come from monitoring data
                # For demonstration, we'll create some dummy data
                usage_data = {
                    "time": pd.date_range(start="2023-09-01", periods=30, freq="D"),
                    "usage_mb": [100 + i*10 + i*i*0.5 for i in range(30)],
                }
                
                usage_df = pd.DataFrame(usage_data)
                
                fig = px.line(
                    usage_df,
                    x="time",
                    y="usage_mb",
                    labels={"time": "Date", "usage_mb": "Storage Usage (MB)"},
                    title="Storage Usage Over Time",
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                # Performance metrics
                st.subheader("Query Performance")
                
                # This would typically come from monitoring data
                # For demonstration, we'll create some dummy data
                perf_data = {
                    "query_type": ["Filter", "Aggregate", "Join", "Similarity", "Get"] * 5,
                    "execution_time": [random() * 100 for _ in range(25)],
                    "time": pd.date_range(start="2023-09-10", periods=25, freq="H"),
                }
                
                perf_df = pd.DataFrame(perf_data)
                
                fig = px.box(
                    perf_df,
                    x="query_type",
                    y="execution_time",
                    labels={"query_type": "Query Type", "execution_time": "Execution Time (ms)"},
                    title="Query Performance by Type",
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                # Logs
                st.subheader("System Logs")
                
                # This would typically come from log data
                # For demonstration, we'll create some dummy logs
                logs = [
                    {"timestamp": "2023-09-15 14:30:00", "level": "INFO", "message": "System started"},
                    {"timestamp": "2023-09-15 14:32:00", "level": "INFO", "message": "Dataset 'mnist' created"},
                    {"timestamp": "2023-09-15 14:35:12", "level": "INFO", "message": "Tensor 'images' added to dataset 'mnist'"},
                    {"timestamp": "2023-09-15 14:36:45", "level": "INFO", "message": "Tensor 'labels' added to dataset 'mnist'"},
                    {"timestamp": "2023-09-15 15:10:22", "level": "INFO", "message": "NQL query executed: 'Find all images with label 7'"},
                    {"timestamp": "2023-09-15 15:45:33", "level": "INFO", "message": "Optimizer: Recompressed tensor 'images' in dataset 'mnist'"},
                    {"timestamp": "2023-09-15 16:20:15", "level": "WARNING", "message": "High memory usage detected"},
                    {"timestamp": "2023-09-15 16:25:42", "level": "ERROR", "message": "Failed to execute query: Out of memory error"},
                    {"timestamp": "2023-09-15 16:30:11", "level": "INFO", "message": "Memory usage normalized"},
                    {"timestamp": "2023-09-15 17:15:00", "level": "INFO", "message": "System shutdown"},
                ]
                
                # Filter logs
                log_level = st.selectbox("Filter by log level", ["All", "INFO", "WARNING", "ERROR"])
                
                filtered_logs = logs
                if log_level != "All":
                    filtered_logs = [log for log in logs if log["level"] == log_level]
                    
                log_df = pd.DataFrame(filtered_logs)
                st.dataframe(log_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering monitoring page: {str(e)}")
    
    def _render_settings_page(self) -> None:
        """Render the settings page."""
        st.title("Settings")
        
        try:
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["General", "Storage", "Optimizer"])
            
            with tab1:
                # General settings
                st.subheader("General Settings")
                
                # Dashboard settings
                st.text_input("Dashboard Title", value="Tensorus Dashboard")
                st.selectbox("Default Page", ["Overview", "Datasets", "Optimization", "Monitoring", "Settings"])
                
                # User settings
                st.subheader("User Settings")
                st.text_input("Username", value="admin")
                st.text_input("Email", value="admin@example.com")
                
                # Save button
                if st.button("Save General Settings"):
                    st.success("Settings saved successfully.")
                    
            with tab2:
                # Storage settings
                st.subheader("Storage Settings")
                
                # Storage path
                storage_path = st.text_input("Storage Path", value=self.storage_path)
                
                # Storage backend
                storage_backend = st.selectbox("Storage Backend", ["Local", "S3", "GCS", "Azure Blob"])
                
                # Backend settings
                if storage_backend == "S3":
                    st.text_input("S3 Bucket", value="tensorus-data")
                    st.text_input("AWS Access Key ID")
                    st.text_input("AWS Secret Access Key", type="password")
                    st.text_input("AWS Region", value="us-west-2")
                elif storage_backend == "GCS":
                    st.text_input("GCS Bucket", value="tensorus-data")
                    st.file_uploader("GCP Service Account Key (JSON)")
                elif storage_backend == "Azure Blob":
                    st.text_input("Azure Container", value="tensorus-data")
                    st.text_input("Azure Connection String")
                    
                # Save button
                if st.button("Save Storage Settings"):
                    st.success("Storage settings saved successfully.")
                    
            with tab3:
                # Optimizer settings
                st.subheader("Optimizer Settings")
                
                # Optimizer config
                if self.optimizer_agent and hasattr(self.optimizer_agent, "config"):
                    optimizer_config = self.optimizer_agent.config
                    
                    # Execution interval
                    execution_interval = st.number_input(
                        "Execution Interval (seconds)",
                        min_value=60,
                        max_value=86400,
                        value=self.optimizer_agent.execution_interval,
                        step=60,
                    )
                    
                    # Auto apply
                    auto_apply = st.checkbox("Auto Apply Optimizations", value=self.optimizer_agent.auto_apply)
                    
                    # Thresholds
                    st.subheader("Optimization Thresholds")
                    
                    thresholds = optimizer_config.get("thresholds", {})
                    compression_ratio = st.slider(
                        "Compression Ratio Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=thresholds.get("compression_ratio", 0.7),
                        step=0.05,
                    )
                    
                    access_frequency = st.number_input(
                        "Access Frequency Threshold",
                        min_value=1,
                        max_value=1000,
                        value=thresholds.get("access_frequency", 10),
                        step=1,
                    )
                    
                    # Actions
                    st.subheader("Optimization Actions")
                    
                    actions = optimizer_config.get("actions", {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        recompression_enabled = st.checkbox(
                            "Enable Recompression",
                            value=actions.get("recompression", {}).get("enabled", True),
                        )
                        
                        rechunking_enabled = st.checkbox(
                            "Enable Rechunking",
                            value=actions.get("rechunking", {}).get("enabled", True),
                        )
                        
                        tensor_caching_enabled = st.checkbox(
                            "Enable Tensor Caching",
                            value=actions.get("tensor_caching", {}).get("enabled", True),
                        )
                        
                        dataset_compaction_enabled = st.checkbox(
                            "Enable Dataset Compaction",
                            value=actions.get("dataset_compaction", {}).get("enabled", True),
                        )
                        
                    with col2:
                        index_creation_enabled = st.checkbox(
                            "Enable Index Creation",
                            value=actions.get("index_creation", {}).get("enabled", True),
                        )
                        
                        index_recreation_enabled = st.checkbox(
                            "Enable Index Recreation",
                            value=actions.get("index_recreation", {}).get("enabled", True),
                        )
                        
                        partition_adjustment_enabled = st.checkbox(
                            "Enable Partition Adjustment",
                            value=actions.get("partition_adjustment", {}).get("enabled", True),
                        )
                        
                        statistics_update_enabled = st.checkbox(
                            "Enable Statistics Update",
                            value=actions.get("statistics_update", {}).get("enabled", True),
                        )
                        
                    # Save button
                    if st.button("Save Optimizer Settings"):
                        # Update optimizer config
                        config_updates = {
                            "thresholds": {
                                "compression_ratio": compression_ratio,
                                "access_frequency": access_frequency,
                            },
                            "actions": {
                                "recompression": {
                                    "enabled": recompression_enabled,
                                },
                                "rechunking": {
                                    "enabled": rechunking_enabled,
                                },
                                "tensor_caching": {
                                    "enabled": tensor_caching_enabled,
                                },
                                "dataset_compaction": {
                                    "enabled": dataset_compaction_enabled,
                                },
                                "index_creation": {
                                    "enabled": index_creation_enabled,
                                },
                                "index_recreation": {
                                    "enabled": index_recreation_enabled,
                                },
                                "partition_adjustment": {
                                    "enabled": partition_adjustment_enabled,
                                },
                                "statistics_update": {
                                    "enabled": statistics_update_enabled,
                                },
                            },
                        }
                        
                        self.optimizer_agent.update_config(config_updates)
                        self.optimizer_agent.execution_interval = execution_interval
                        self.optimizer_agent.auto_apply = auto_apply
                        
                        st.success("Optimizer settings saved successfully.")
                        
                else:
                    st.warning("Optimizer agent not initialized or configuration not available.")
                    
        except Exception as e:
            st.error(f"Error rendering settings page: {str(e)}")
    
    def run(self) -> None:
        """Run the dashboard."""
        # Fix missing function in demo
        from random import random
        
        # Render the sidebar
        self._render_sidebar()


def main():
    """Run the Tensorus Dashboard."""
    dashboard = TensorusDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 