import os
from typing import Dict, List, Optional, Union

import pandas as pd


class DatasetLoader:
    """
    Handles loading, validating, and managing custom datasets for benchmarking.
    Datasets are stored in the 'datasets' directory in the project root.
    """

    def __init__(self, datasets_dir: str = "datasets"):
        # Resolve to absolute path to prevent path traversal
        self.datasets_dir = os.path.abspath(datasets_dir)
        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)

    def _get_file_path(self, filename: str) -> str:
        """
        Get the full file path for a dataset filename.

        Raises:
            ValueError: If the filename contains path traversal attempts
                        or tries to access files outside the datasets directory.
        """
        # Check for path traversal patterns
        if ".." in filename or filename.startswith(("/", "\\")):
            raise ValueError("Path traversal detected: filename must be a simple name, not a path")

        # Join with datasets directory and resolve to absolute path
        full_path = os.path.abspath(os.path.join(self.datasets_dir, filename))

        # Verify the resolved path is within the datasets directory
        if not full_path.startswith(self.datasets_dir):
            raise ValueError(
                f"Path traversal detected: resolved path is outside datasets directory"
            )

        return full_path

    def list_datasets(self) -> list[dict[str, str]]:
        """List all available datasets with metadata."""
        datasets = []
        if not os.path.exists(self.datasets_dir):
            return []

        for f in os.listdir(self.datasets_dir):
            if f.endswith(('.csv', '.json')):
                path = self._get_file_path(f)
                try:
                    size_bytes = os.path.getsize(path)
                    datasets.append({
                        "filename": f,
                        "size": f"{size_bytes / 1024:.1f} KB",
                        "type": f.split('.')[-1].upper()
                    })
                except OSError:
                    continue
        return datasets

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataset has the required structure.
        Must contain a 'prompt' column.
        """
        return 'prompt' in df.columns

    def save_dataset(self, file_obj, filename: str) -> Union[str, None]:
        """
        Save an uploaded file to the datasets directory.
        Returns the error message if failed, None if successful.
        """
        try:
            # Determine file type and load into DataFrame for validation
            if filename.endswith('.csv'):
                df = pd.read_csv(file_obj)
            elif filename.endswith('.json'):
                df = pd.read_json(file_obj)
            else:
                return "Unsupported file format. Please upload .csv or .json."

            # Validate
            if not self.validate_dataset(df):
                return "Invalid dataset format. Must contain a 'prompt' column."

            # Save to disk
            save_path = self._get_file_path(filename)
            if filename.endswith('.csv'):
                df.to_csv(save_path, index=False)
            elif filename.endswith('.json'):
                df.to_json(save_path, orient='records', indent=2)

            return None # Success

        except Exception as e:
            return f"Error saving dataset: {str(e)}"

    def load_dataset(self, filename: str) -> pd.DataFrame | None:
        """Load a dataset into a DataFrame."""
        path = self._get_file_path(filename)
        if not os.path.exists(path):
            return None

        try:
            if filename.endswith('.csv'):
                return pd.read_csv(path)
            elif filename.endswith('.json'):
                return pd.read_json(path)
        except Exception as e:
            print(f"Error loading dataset {filename}: {e}")
            return None
        return None

    def delete_dataset(self, filename: str) -> bool:
        """Delete a dataset file."""
        path = self._get_file_path(filename)
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except OSError:
                return False
        return False

    def get_dataset_preview(self, filename: str, rows: int = 5) -> pd.DataFrame | None:
        """Get a preview of the dataset."""
        df = self.load_dataset(filename)
        if df is not None:
            return df.head(rows)
        return None
