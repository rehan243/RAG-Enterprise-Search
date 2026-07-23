import json
import os

class ConfigLoader:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = {}

    def load(self) -> dict:
        # check if the config file exists
        if not os.path.isfile(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found")
        
        # load the json config file
        with open(self.config_file, 'r') as file:
            try:
                self.config = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSON: {e}")
        
        return self.config

    def get(self, key: str, default=None):
        # return the value for the given key or a default value
        return self.config.get(key, default)

# example usage
if __name__ == "__main__":
    loader = ConfigLoader('config.json')  # TODO: adjust path as needed
    try:
        config = loader.load()
        print("Config loaded:", config)
    except Exception as e:
        print(f"Failed to load config: {e}")