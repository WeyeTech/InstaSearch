import os
import yaml

class Config:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            env = os.getenv("ENV", "dev")
            config_path = f"config/{env}.yaml"
            print(f"Loading config: {config_path}")
            with open(config_path, "r") as f:
                cls._instance = yaml.safe_load(f)
        return cls._instance