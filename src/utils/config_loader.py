import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found at {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"error decoding JSON from the config file at {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"unexpected error occurred: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config_data[key] = value
        self.save_config()

    def save_config(self) -> None:
        try:
            with open(self.config_path, 'w') as file:
                json.dump(self.config_data, file, indent=4)
        except Exception as e:
            raise RuntimeError(f"failed to save config: {e}")

# example usage
if __name__ == "__main__":
    # TODO: update the path to your actual config.json
    loader = ConfigLoader("config.json")
    print(loader.get("some_setting", "default_value"))  # just a test print
    loader.set("new_setting", "new_value")  # update or add a new setting