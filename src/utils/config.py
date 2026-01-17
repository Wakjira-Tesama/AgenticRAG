"""
Configuration management utilities
"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path: str = "config/agent_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        config = _replace_env_vars(config)
        
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()

def save_config(config: Dict[str, Any], config_path: str = "config/agent_config.yaml"):
    """Save configuration to YAML file"""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def _replace_env_vars(config: Any) -> Any:
    """Recursively replace ${VAR} with environment variables"""
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        if config.startswith('${') and config.endswith('}'):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config
    else:
        return config

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'system': {
            'name': 'Agentic RAG System',
            'version': '1.0.0',
            'description': 'Default configuration'
        },
        'agents': {
            'orchestrator': {
                'temperature': 0.1,
                'max_iterations': 3
            },
            'maker': {
                'temperature': 0.1,
                'max_tokens': 1500
            },
            'checker': {
                'temperature': 0.0,
                'max_tokens': 1000
            },
            'retriever': {
                'top_k': 5,
                'similarity_threshold': 0.7
            }
        },
        'safety': {
            'input_validation': {
                'enabled': True,
                'max_query_length': 1000
            },
            'output_sanitization': {
                'enabled': True
            }
        }
    }

def update_config_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update config with environment-specific settings"""
    env = os.getenv('ENVIRONMENT', 'development')
    
    env_config_path = f"config/agent_config_{env}.yaml"
    if os.path.exists(env_config_path):
        env_config = load_config(env_config_path)
        # Merge configs (env overrides default)
        config = _merge_dicts(config, env_config)
    
    return config

def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result