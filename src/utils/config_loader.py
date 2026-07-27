import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        # check if the config file exists
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON in config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        # return the value for the given key or default if not found
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        # set a new value for the given key
        self.config_data[key] = value
        self.save_config()  # TODO: implement save_config method

    def save_config(self) -> None:
        # save the current config back to the file
        with open(self.config_path, 'w') as file:
            json.dump(self.config_data, file, indent=4)

# example usage
# config_loader = ConfigLoader('path/to/config.json')
# print(config_loader.get('some_key', 'default_value'))