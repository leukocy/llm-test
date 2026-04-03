"""
Preset Manager for Test Configurations

Manages saving, loading, and deleting test configuration presets.
Presets are stored as JSON files in the presets/ directory.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


def get_presets_dir() -> str:
    """Get the presets directory path, creating it if it doesn't exist."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    presets_dir = os.path.join(project_root, 'presets')

    if not os.path.exists(presets_dir):
        os.makedirs(presets_dir)

    return presets_dir

def _sanitize_preset_name(name: str) -> str:
    """Sanitize preset name to be a valid filename."""
    # Remove special characters and spaces
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_'))
    safe_name = safe_name.strip().replace(' ', '_')
    return safe_name

def save_preset(name: str, config: dict) -> bool:
    """
    Save a test configuration as a preset.

    Args:
        name: Name of the preset
        config: Configuration dictionary to save

    Returns:
        True if saved successfully, False otherwise
    """
    if not name or not name.strip():
        raise ValueError("Preset name cannot be empty")

    safe_name = _sanitize_preset_name(name)
    if not safe_name:
        raise ValueError("Preset name contains only invalid characters")

    presets_dir = get_presets_dir()
    preset_path = os.path.join(presets_dir, f"{safe_name}.json")

    preset_data = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "config": config
    }

    try:
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving preset: {e}")
        return False

def load_preset(name: str) -> dict | None:
    """
    Load a preset by name.

    Args:
        name: Name of the preset (can be display name or sanitized name)

    Returns:
        Preset data dictionary or None if not found
    """
    safe_name = _sanitize_preset_name(name)
    presets_dir = get_presets_dir()
    preset_path = os.path.join(presets_dir, f"{safe_name}.json")

    if not os.path.exists(preset_path):
        return None

    try:
        with open(preset_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading preset: {e}")
        return None

def list_presets() -> list[dict]:
    """
    List all available presets.

    Returns:
        List of preset metadata (name, created_at)
    """
    presets_dir = get_presets_dir()
    presets = []

    try:
        for filename in os.listdir(presets_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(presets_dir, filename)
                try:
                    with open(filepath, encoding='utf-8') as f:
                        preset = json.load(f)
                        presets.append({
                            "name": preset.get("name", filename[:-5]),
                            "created_at": preset.get("created_at", "Unknown"),
                            "filename": filename
                        })
                except Exception as e:
                    print(f"Error reading preset {filename}: {e}")
    except Exception as e:
        print(f"Error listing presets: {e}")

    return sorted(presets, key=lambda x: x['created_at'], reverse=True)

def delete_preset(name: str) -> bool:
    """
    Delete a preset by name.

    Args:
        name: Name of the preset

    Returns:
        True if deleted successfully, False otherwise
    """
    safe_name = _sanitize_preset_name(name)
    presets_dir = get_presets_dir()
    preset_path = os.path.join(presets_dir, f"{safe_name}.json")

    if not os.path.exists(preset_path):
        return False

    try:
        os.remove(preset_path)
        return True
    except Exception as e:
        print(f"Error deleting preset: {e}")
        return False
