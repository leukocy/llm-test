import glob
import os
from typing import Dict, List

import pandas as pd


class HistoryManager:
    """
    Manages historical benchmark results stored in the 'raw_data' directory.
    """

    def __init__(self, base_dir: str = "raw_data"):
        self.base_dir = base_dir

    def list_history(self) -> list[dict[str, str]]:
        """
        List all available benchmark result CSV files.
        Returns a list of dicts with metadata (filename, model_id, timestamp, path).
        """
        if not os.path.exists(self.base_dir):
            return []

        # Find all CSV files recursively
        csv_files = glob.glob(os.path.join(self.base_dir, "**", "*.csv"), recursive=True)

        history = []
        for path in csv_files:
            try:
                filename = os.path.basename(path)
                # Expected format: benchmark_results_{model_id}_{test_type}_{timestamp}.csv
                # But model_id might contain underscores, so splitting by fixed prefix might be safer if possible.
                # Or just use the filename as display name.

                # Let's try to parse, but fallback to filename
                parts = filename.replace('.csv', '').split('_')

                # Simple parsing logic (heuristic)
                timestamp = parts[-2] + "_" + parts[-1] if len(parts) >= 2 and parts[-1].isdigit() else "Unknown"

                # Get file stats
                stats = os.stat(path)
                size_kb = stats.st_size / 1024

                history.append({
                    "display_name": filename,
                    "path": path,
                    "size_kb": size_kb,
                    "timestamp": timestamp
                })
            except Exception as e:
                print(f"Error parsing file {path}: {e}")
                continue

        # Sort by modification time (newest first)
        history.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
        return history

    def load_results(self, paths: list[str]) -> dict[str, pd.DataFrame]:
        """
        Load multiple result files into DataFrames.
        Returns a dict mapping display_name (or path) to DataFrame.
        """
        results = {}
        for path in paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    # Add a column for source identification if needed
                    filename = os.path.basename(path)
                    df['source_file'] = filename
                    results[filename] = df
            except Exception as e:
                print(f"Error loading {path}: {e}")
        return results
