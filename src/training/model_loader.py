"""FORGEGUARD Model
responbilities: 1. Load YAML config 2.validate config3.return configuration dictionary
"""
import yaml
from pathlib import Path

def load_config(config_path="configs/training_config.yaml"):
    """ Load YAML configuration file.
         
        returns:
         dict:configuration settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"configuration file not found: {config_path}"
        )
        
    with open(config_path, "r", encoding = "utf-8") as file:
        config =  yaml.safe_load(file)
        
    return config

if __name__ == "__main__":
    config = load_config()
    
    print("config load successfully")
    print(config)