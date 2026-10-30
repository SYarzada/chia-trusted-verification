

from typing import Optional


class GlobalState:
    location_prefix: Optional[str] = None

    @staticmethod
    def clear_location_prefix():
        
        GlobalState.location_prefix = None

    @staticmethod
    def get_location_prefix() -> Optional[str]:
        
        return GlobalState.location_prefix

    @staticmethod
    def get_location_message(location: str) -> str:
        
        prefix = GlobalState.get_location_prefix()
        if prefix:
            extra = prefix + "'s "
        else:
            extra = ""
        return " in the " + extra + location + " specification"

    @staticmethod
    def set_location_prefix(prefix: str):
        
        GlobalState.location_prefix = prefix
