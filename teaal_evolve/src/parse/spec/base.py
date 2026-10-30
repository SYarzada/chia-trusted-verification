

from typing import Optional

from src.common import Base as GlobalBase
from src.parse.common.yaml import Yaml


class Base(GlobalBase):
    

    @staticmethod
    def parse_file(filename: str) -> dict:
        
        return Yaml.parse_file(filename)

    @staticmethod
    def parse_str(string: str) -> dict:
        
        return Yaml.parse_str(string)
