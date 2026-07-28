import yaml
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        # check if the file exists
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        # load the yaml config
        with open(self.config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        # retrieve value by key, return default if not found
        return self.config_data.get(key, default)

# example usage
if __name__ == "__main__":
    # TODO: update this path to your config file
    config_loader = ConfigLoader('config.yaml')
    print(config_loader.get('some_key', 'default_value'))