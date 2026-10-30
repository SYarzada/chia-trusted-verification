

from ruamel.yaml import YAML                

from src.parse.common.node import BaseNode, NODES


class Yaml:
    
    @staticmethod
    def parse_str(string: str) -> dict:
        
        yaml = YAML(typ='safe', pure=True)

        for class_ in NODES:
            yaml.register_class(class_)

        return BaseNode.flatten_spec(yaml.load(string))

    @staticmethod
    def parse_file(input_file: str) -> dict:
        
        with open(input_file, 'r') as stream:
            yaml = YAML(typ='safe', pure=True)

            for class_ in NODES:
                yaml.register_class(class_)

            data_loaded = yaml.load(stream)

        return BaseNode.flatten_spec(data_loaded)
