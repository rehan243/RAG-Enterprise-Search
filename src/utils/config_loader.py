import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return json.load(file)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config_data[key] = value
        self.save_config()

    def save_config(self) -> None:
        with open(self.config_path, 'w') as file:
            json.dump(self.config_data, file, indent=4)

# example usage
if __name__ == "__main__":
    # TODO: update the path to your actual config.json
    loader = ConfigLoader("config.json")
    print(loader.get("some_setting", "default_value"))  # just a test print
    loader.set("new_setting", "new_value")  # update or add a new setting