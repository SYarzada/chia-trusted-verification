
from lark.tree import Tree
from typing import Any, Dict, List, Set, Tuple

from src.common import *
from src.ir.gen import *
from src.parse import *


class MemoryComponent(Base):
    

    def __init__(self, memory_parser: MemoryParser):
        self.attributes = memory_parser.get_attributes()
        self.constructors = {
            name: LineGeneration(constructor) for name,
            constructor in memory_parser.get_constructors().items()}
        self.access_template = LineGeneration(
            memory_parser.get_access_template())
        self.reshape_template = LineGeneration(
            memory_parser.get_reshape_template())
        self.name = memory_parser.get_name()

        location = self.name + " node"

        self.ports = None
        if self.attributes.ports:
            self.ports = {port.name: port for port in self.attributes.ports}

        ranks = memory_parser.get_ranks()
        self.ranks = ranks
        ErrorUtils.check_mem_ranks(set(self.ranks), ranks, location)

    def get_access_template(self) -> LineGeneration:
        
        return self.access_template

    def get_bit_widths(self) -> List[int]:
        
        return self.attributes.bit_widths

    def get_constructors(self) -> Dict[str, LineGeneration]:
        
        return self.constructors

    def get_data_types(self) -> List[str]:
        
        return self.attributes.data_types

    def get_name(self) -> str:
        
        return self.name

    def get_ports(self, engine: str) -> Optional[List[str]]:
        
        if self.ports is None:
            return None

        ports = []
        for port in self.ports.values():
            if engine in port.engines:
                ports.append(port.name)

        return ports

    def get_ranks(self) -> List[str]:
        
        return self.ranks

    def get_reshape_template(self) -> LineGeneration:
        
        return self.reshape_template

    def maybe_get_bandwidth(
            self,
            engine: str,
            port: Optional[str],
            einsum: str,
            tensor: str) -> Optional[float]:
        
        maybe_port = self.maybe_get_port(engine, port, einsum, tensor)
        if maybe_port:
            return maybe_port.bandwidth
        return None

    def maybe_get_port(
            self,
            engine: str,
            port: Optional[str],
            einsum: str,
            tensor: str) -> Optional[PortParser]:
        
        if self.ports is None:
            if port is None:
                return None
            else:
                raise UnknownPortError(
                    port,
                    self.get_name(),
                    engine,
                    einsum +
                    " Einsum's " +
                    tensor +
                    " tensor binding")

        parent_loc = " (bound in the " + einsum + \
            " Einsum's " + tensor + " tensor binding)"
        if port:
            if port not in self.ports.keys():
                raise UnknownPortError(
                    port,
                    self.get_name(),
                    engine,
                    einsum +
                    " Einsum's " +
                    tensor +
                    " tensor binding")

            if engine not in self.ports[port].engines:
                raise MissingComponentError(
                    engine, self.name + "'s " + port + " port" + parent_loc)
            else:
                return self.ports[port]

        ports = self.get_ports(engine)
        assert ports is not None

        if len(ports) == 0:
            raise MissingComponentError(
                engine, self.name + " ports" + parent_loc)
        elif len(ports) == 1:
            return self.ports[ports[0]]
        else:
            raise MultiplePortsError(ports, einsum, tensor)

    def key(self) -> Tuple[Any, ...]:
        return self.attributes, self.constructors, self.name, self.ranks, self.access_template, self.reshape_template
