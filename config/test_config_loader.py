"""
Utility to load test configurations from test_config.ini
"""
import configparser
import os


def load_test_config(config_name='deepseek_test'):
    """
    Load test configuration from test_config.ini

    Args:
        config_name: Name of the configuration section to load

    Returns:
        dict: Configuration dictionary or None if file doesn't exist
    """
    config_file = os.path.join(os.path.dirname(__file__), '..', 'test_config.ini')

    if not os.path.exists(config_file):
        return None

    config = configparser.ConfigParser()
    config.read(config_file)

    if config_name not in config:
        return None

    return {
        'provider': config[config_name].get('provider', 'OpenAI (兼容)'),
        'api_base_url': config[config_name].get('api_base_url', ''),
        'model_id': config[config_name].get('model_id', ''),
        'api_key': config[config_name].get('api_key', ''),
        'tokenizer_option': config[config_name].get('tokenizer_option', 'API (usage field)')
    }
