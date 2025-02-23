import yaml
from dataclasses import dataclass

@dataclass
class PreprocessorConfig:
    target_lat: int
    target_lon: int

@dataclass
class DataConfig:
    train_period: dict
    test_period: dict
    validation_period: dict

class Configuration:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.preprocessor = PreprocessorConfig(**config_dict['preprocessor'])
        self.data = DataConfig(**config_dict['data'])
    
    @classmethod
    def load(cls, config_path: str) -> 'Configuration':
        return cls(config_path)