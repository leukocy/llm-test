"""
Test Configuration管理模块

提供Test ConfigurationSave、Load、ImportExport功能：
- Save常用Test Configurationis预设
- 从预设Load Config
- Import/ExportConfigure文件
- 管理Config Presets
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from config.session_state import set_current_test_type
from config.test_types import normalize_test_type as normalize_test_type_id
from ui.design_system import material_icon

# ============================================================================
# Config预设类
# ============================================================================


class ConfigPreset:
    """Config PresetsData类"""

    def __init__(
        self,
        name: str,
        description: str,
        config: dict[str, Any],
        tags: list[str] | None = None,
        created_at: str | None = None,
    ):
        self.name = name
        self.description = description
        self.config = config
        self.tags = tags or []
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigPreset":
        """从字典Create"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            config=data["config"],
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
        )


# ============================================================================
# Config管理器
# ============================================================================


class TestConfigManager:
    """Test Configuration管理器"""

    def __init__(self, config_dir: str = "test_presets"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def save_preset(self, preset: ConfigPreset) -> bool:
        """
        Save Config预设

        Args:
            preset: Config Presets对象

        Returns:
            is否Savesucceeded
        """
        try:
            # use安全Filename
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in preset.name
            )
            filename = safe_name.lower().replace(" ", "_") + ".json"
            filepath = self.config_dir / filename

            # 确保Filename唯一
            counter = 1
            while filepath.exists():
                filename = f"{safe_name.lower().replace(' ', '_')}_{counter}.json"
                filepath = self.config_dir / filename
                counter += 1

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            st.error(f"Failed to save preset: {e}")
            return False

    def load_preset(self, name: str) -> ConfigPreset | None:
        """
        Load Config预设

        Args:
            name: Preset name

        Returns:
            Config Presets对象，ifnot存in则Return None
        """
        try:
            # 查找匹配文件
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name
            )
            pattern = safe_name.lower().replace(" ", "_") + "*.json"
            matching_files = list(self.config_dir.glob(pattern))

            if not matching_files:
                return None

            # use一匹配文件
            with open(matching_files[0], encoding="utf-8") as f:
                data = json.load(f)

            return ConfigPreset.from_dict(data)
        except Exception as e:
            st.error(f"Failed to load preset: {e}")
            return None

    def list_presets(self) -> list[dict[str, Any]]:
        """
        列出所hasConfig Presets

        Returns:
            预设信息列表
        """
        presets = []
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, encoding="utf-8") as f:
                    data = json.load(f)

                presets.append(
                    {
                        "name": data.get("name", config_file.stem),
                        "description": data.get("description", ""),
                        "tags": data.get("tags", []),
                        "created_at": data.get("created_at", ""),
                        "file_path": str(config_file),
                    }
                )
            except Exception:
                continue

        return sorted(presets, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_preset(self, name: str) -> bool:
        """
        Delete Config预设

        Args:
            name: Preset name

        Returns:
            is否Deletesucceeded
        """
        try:
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name
            )
            pattern = safe_name.lower().replace(" ", "_") + "*.json"
            matching_files = list(self.config_dir.glob(pattern))

            for filepath in matching_files:
                filepath.unlink()

            return True
        except Exception as e:
            st.error(f"Failed to delete preset: {e}")
            return False

    def export_preset(self, name: str, export_path: str) -> bool:
        """
        ExportConfig Presets到指定路径

        Args:
            name: Preset name
            export_path: Export路径

        Returns:
            is否Exportsucceeded
        """
        try:
            preset = self.load_preset(name)
            if not preset:
                st.error(f"Preset not found: {name}")
                return False

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            st.error(f"Failed to export preset: {e}")
            return False

    def import_preset(self, import_path: str) -> ConfigPreset | None:
        """
        从指定路径ImportConfig Presets

        Args:
            import_path: Import路径

        Returns:
            ImportConfig Presets对象
        """
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                st.error(f"File not found: {import_path}")
                return None

            with open(import_file, encoding="utf-8") as f:
                data = json.load(f)

            preset = ConfigPreset.from_dict(data)

            # SaveImport预设
            self.save_preset(preset)

            return preset
        except Exception as e:
            st.error(f"Failed to import preset: {e}")
            return None


# 全局Configure管理器实例
config_manager = TestConfigManager()


# ============================================================================
# 预设Configure模板
# ============================================================================


def get_builtin_presets() -> list[ConfigPreset]:
    """GetBuilt-in PresetsConfigure"""
    return [
        ConfigPreset(
            name="Quick Test",
            description="Low-concurrency configuration for a fast baseline check.",
            config={
                "test_type": "concurrency",
                "concurrency": 1,
                "max_tokens": 256,
                "temperature": 0.0,
                "thinking_enabled": False,
            },
            tags=["quick", "baseline"],
        ),
        ConfigPreset(
            name="Standard Test",
            description="Balanced concurrency configuration for routine benchmarks.",
            config={
                "test_type": "concurrency",
                "concurrency": 4,
                "max_tokens": 512,
                "temperature": 0.0,
                "thinking_enabled": False,
            },
            tags=["standard", "recommended"],
        ),
        ConfigPreset(
            name="Stress Test",
            description="High-concurrency configuration for load testing.",
            config={
                "test_type": "concurrency",
                "concurrency": 16,
                "max_tokens": 1024,
                "temperature": 0.0,
                "thinking_enabled": False,
            },
            tags=["stress", "performance"],
        ),
        ConfigPreset(
            name="Reasoning Test",
            description="Reasoning-enabled configuration with a high thinking budget.",
            config={
                "test_type": "concurrency",
                "concurrency": 2,
                "max_tokens": 2048,
                "temperature": 0.0,
                "thinking_enabled": True,
                "thinking_budget": 10000,
                "reasoning_effort": "high",
            },
            tags=["reasoning", "quality"],
        ),
        ConfigPreset(
            name="Long Context Test",
            description="Long-context performance configuration.",
            config={
                "test_type": "long_context",
                "concurrency": 1,
                "max_tokens": 512,
                "context_length": 32768,
                "temperature": 0.0,
                "thinking_enabled": False,
            },
            tags=["long-context", "performance"],
        ),
        ConfigPreset(
            name="Prefill Stress Test",
            description="Prefill-stage stress configuration.",
            config={
                "test_type": "prefill",
                "concurrency": 4,
                "max_tokens": 1,
                "temperature": 0.0,
                "thinking_enabled": False,
            },
            tags=["prefill", "stress"],
        ),
        ConfigPreset(
            name="Comprehensive Test",
            description="Multi-dimensional concurrency and context configuration.",
            config={
                "test_type": "matrix",
                "concurrency": 4,
                "max_tokens": 512,
                "temperature": 0.0,
                "thinking_enabled": False,
            },
            tags=["matrix", "comprehensive"],
        ),
        ConfigPreset(
            name="Creative Test",
            description="High-temperature configuration for creative generation.",
            config={
                "test_type": "concurrency",
                "concurrency": 2,
                "max_tokens": 1024,
                "temperature": 0.8,
                "thinking_enabled": False,
            },
            tags=["creative", "high-temperature"],
        ),
    ]


def init_builtin_presets():
    """InitializeBuilt-in Presets（ifnot存in）"""
    existing_presets = config_manager.list_presets()
    existing_names = {p["name"] for p in existing_presets}

    for preset in get_builtin_presets():
        if preset.name not in existing_names:
            config_manager.save_preset(preset)


# ============================================================================
# UI 组件
# ============================================================================


def render_preset_manager():
    """Render预设管理界面"""
    st.subheader("Test Configuration Presets")

    # Get所has预设
    all_presets = config_manager.list_presets()

    # 按LabelGroup
    tag_groups: dict[str, list[dict[str, Any]]] = {}
    for preset in all_presets:
        for tag in preset.get("tags", []):
            if tag not in tag_groups:
                tag_groups[tag] = []
            tag_groups[tag].append(preset)

    # Display预设
    if not all_presets:
        st.info("No saved configuration presets")
        return

    # Label页
    tab_all, tab_by_tag = st.tabs(["All Presets", "Browse by Tag"])

    with tab_all:
        for preset in all_presets:
            with st.expander(preset["name"], expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(preset.get("description", "No description"))

                with col2:
                    # LabelDisplay
                    tags = preset.get("tags", [])
                    if tags:
                        st.markdown(" ".join([f"`{tag}`" for tag in tags]))

                with col3:
                    # 操作按钮
                    col_load, col_del = st.columns(2)

                    with col_load:
                        if st.button(
                            "Apply",
                            key=f"load_{preset['name']}",
                            icon=material_icon("check"),
                        ):
                            if apply_preset(preset["name"]):
                                st.success(f"Applied preset: {preset['name']}")
                                st.rerun()

                    with col_del:
                        if st.button(
                            "Delete",
                            key=f"del_{preset['name']}",
                            help="Delete preset",
                            icon=material_icon("delete"),
                        ):
                            if config_manager.delete_preset(preset["name"]):
                                st.rerun()

    with tab_by_tag:
        for tag, presets in sorted(tag_groups.items()):
            st.markdown(f"**`{tag}`**")
            for preset in presets:
                if st.button(
                    preset["name"],
                    key=f"tag_{tag}_{preset['name']}",
                    use_container_width=True,
                ):
                    if apply_preset(preset["name"]):
                        st.success(f"Applied preset: {preset['name']}")
                        st.rerun()
            st.markdown("---")


def render_save_preset_form():
    """RenderSave预设表单"""
    with st.expander("Save Current Configuration as Preset", expanded=False):
        name = st.text_input(
            "Preset name", key="preset_name", placeholder="For example: Custom Test"
        )
        description = st.text_area(
            "Description (optional)",
            key="preset_desc",
            placeholder="Describe when this configuration should be used.",
        )
        tags_input = st.text_input(
            "Tags (optional, comma-separated)",
            key="preset_tags",
            placeholder="For example: quick, baseline",
        )

        col_save, col_cancel = st.columns(2)

        with col_save:
            if st.button(
                "Save Preset",
                type="primary",
                use_container_width=True,
                icon=material_icon("save"),
            ):
                if not name:
                    st.error("Please enter a preset name")
                else:
                    # Get当前Configure
                    current_config = get_current_config()

                    # ProcessLabel
                    tags = (
                        [t.strip() for t in tags_input.split(",") if t.strip()]
                        if tags_input
                        else []
                    )

                    # Create预设
                    preset = ConfigPreset(
                        name=name,
                        description=description,
                        config=current_config,
                        tags=tags,
                    )

                    if config_manager.save_preset(preset):
                        st.success(f"Preset '{name}' saved")


def apply_preset(name: str) -> bool:
    """
    Apply预设Configure到 session_state

    Args:
        name: Preset name

    Returns:
        is否Applysucceeded
    """
    preset = config_manager.load_preset(name)
    if not preset:
        st.error(f"Preset not found: {name}")
        return False

    # ApplyConfigure到 session_state
    config = preset.config

    selected_test_type = None
    if "test_type" in config:
        selected_test_type = set_current_test_type(config["test_type"])
        st.session_state["_force_test_type_selector"] = selected_test_type

    # 映射Configure到 session_state
    config_mapping = {
        "concurrency": "current_concurrency",
        "max_tokens": "current_max_tokens",
        "temperature": "current_temperature",
        "thinking_enabled": "thinking_enabled",
        "thinking_budget": "thinking_budget",
        "reasoning_effort": "reasoning_effort",
        "context_length": "current_context_length",
        "template_tokens": "template_tokens",
    }

    for key, session_key in config_mapping.items():
        if key in config:
            st.session_state[session_key] = config[key]

    _apply_preset_to_widget_state(config, selected_test_type)

    return True


def _as_positive_int(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _apply_preset_to_widget_state(
    config: dict[str, Any], selected_test_type: str | None
) -> None:
    """Sync preset values to the Streamlit widget keys used by the panels."""
    test_type = normalize_test_type_id(selected_test_type or config.get("test_type"))

    max_tokens = _as_positive_int(config.get("max_tokens"))
    concurrency = _as_positive_int(config.get("concurrency"))
    context_length = _as_positive_int(config.get("context_length"))

    if "template_tokens" in config:
        try:
            st.session_state.template_tokens_input = max(
                0, int(config.get("template_tokens") or 0)
            )
        except (TypeError, ValueError):
            st.session_state.template_tokens_input = 0

    if test_type == "concurrency":
        if concurrency:
            st.session_state.con_concurrencies_select = [concurrency]
        if max_tokens:
            st.session_state.con_max_tokens = max_tokens

    elif test_type == "prefill":
        if max_tokens:
            st.session_state.prefill_isolation_mode = max_tokens == 1
            if max_tokens != 1:
                st.session_state.prefill_max_tokens = max_tokens

    elif test_type == "long_context":
        if context_length:
            st.session_state.long_context_lengths_select = [context_length]
        if max_tokens:
            st.session_state.long_max_tokens = max_tokens

    elif test_type == "matrix":
        if concurrency:
            st.session_state.matrix_concurrencies_select = [concurrency]
        if context_length:
            st.session_state.matrix_context_select = [context_length]
        if max_tokens:
            st.session_state.matrix_max_tokens = max_tokens

    elif test_type == "stability":
        if concurrency:
            st.session_state.stability_concurrency = concurrency
        if max_tokens:
            st.session_state.stability_max_tokens = max_tokens


def get_current_config() -> dict[str, Any]:
    """Get当前Configure"""
    test_type = normalize_test_type_id(st.session_state.get("current_test_type"))

    return {
        "test_type": test_type,
        "concurrency": st.session_state.get("current_concurrency", 1),
        "max_tokens": st.session_state.get("current_max_tokens", 512),
        "temperature": st.session_state.get("current_temperature", 0.0),
        "thinking_enabled": st.session_state.get("thinking_enabled", False),
        "thinking_budget": st.session_state.get("thinking_budget", 0),
        "reasoning_effort": st.session_state.get("reasoning_effort", "medium"),
        "context_length": st.session_state.get("current_context_length", 4096),
        "template_tokens": st.session_state.get("template_tokens", 0),
    }


def render_config_import_export():
    """RenderConfigureImport/Export界面"""
    with st.expander("Import or Export Configuration", expanded=False):
        col_import, col_export = st.columns(2)

        with col_import:
            st.markdown("**Import Configuration**")
            uploaded_file = st.file_uploader(
                "Select configuration file (JSON)",
                type=["json"],
                key="import_config",
                help="Upload a previously exported configuration file",
            )

            if uploaded_file:
                if st.button(
                    "Import",
                    use_container_width=True,
                    icon=material_icon("upload_file"),
                ):
                    # Save临时文件
                    temp_path = Path(f"temp_import_{uploaded_file.name}")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Import预设
                    preset = config_manager.import_preset(str(temp_path))
                    if preset:
                        st.success(f"Imported preset: {preset.name}")
                        temp_path.unlink()

        with col_export:
            st.markdown("**Export Configuration**")
            all_presets = config_manager.list_presets()
            preset_names = [p["name"] for p in all_presets]

            if preset_names:
                selected_preset = st.selectbox(
                    "Select preset to export",
                    preset_names,
                    key="export_preset",
                )

                if st.button(
                    "Export",
                    use_container_width=True,
                    icon=material_icon("download"),
                ):
                    # ExportConfigure
                    export_filename = f"{selected_preset}_config.json"
                    preset_data = config_manager.load_preset(selected_preset)

                    if preset_data:
                        json_data = json.dumps(
                            preset_data.to_dict(), ensure_ascii=False, indent=2
                        )
                        st.download_button(
                            label="Download Configuration File",
                            icon=material_icon("download"),
                            data=json_data,
                            file_name=export_filename,
                            mime="application/json",
                            use_container_width=True,
                        )
