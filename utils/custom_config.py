"""
Custom Configuration管理器
用于管理用户AddCustomAPIProviderandModel
"""
import json
import os

CUSTOM_CONFIG_FILE = "config/custom_config.json"

def load_custom_config():
    """LoadCustom Configuration"""
    if os.path.exists(CUSTOM_CONFIG_FILE):
        try:
            with open(CUSTOM_CONFIG_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"LoadCustom Configuration失败: {e}")
    return {"providers": {}, "models": []}

def save_custom_config(config):
    """SaveCustom Configuration"""
    try:
        os.makedirs(os.path.dirname(CUSTOM_CONFIG_FILE), exist_ok=True)
        with open(CUSTOM_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"SaveCustom Configuration失败: {e}")
        return False

def add_custom_provider(name, url):
    """AddCustomProvider"""
    config = load_custom_config()
    config["providers"][name] = url
    return save_custom_config(config)

def remove_custom_provider(name):
    """DeleteCustomProvider"""
    config = load_custom_config()
    if name in config["providers"]:
        del config["providers"][name]
        return save_custom_config(config)
    return False

def add_custom_model(model_name):
    """AddCustomModel"""
    config = load_custom_config()
    if model_name not in config["models"]:
        config["models"].append(model_name)
        return save_custom_config(config)
    return False

def remove_custom_model(model_name):
    """DeleteCustomModel"""
    config = load_custom_config()
    if model_name in config["models"]:
        config["models"].remove(model_name)
        return save_custom_config(config)
    return False

def get_all_providers():
    """Get所hasProvider（内置+Custom）"""
    from config.settings import PROVIDER_OPTIONS
    custom_config = load_custom_config()
    all_providers = PROVIDER_OPTIONS.copy()
    all_providers.update(custom_config["providers"])
    return all_providers

def get_all_models():
    """Get所hasModel（内置+Custom）"""
    from config.settings import MODEL_OPTIONS
    custom_config = load_custom_config()
    all_models = MODEL_OPTIONS.copy()
    all_models.extend(custom_config["models"])
    return all_models
