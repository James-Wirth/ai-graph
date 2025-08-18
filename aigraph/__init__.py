from .api import (
    start, step, route, end,
    data, set_data, vars, tool, emit,
    Result, Route, llm,
    App, AppConfig,
)

__all__ = [
    "start", "step", "route", "end",
    "data", "set_data", "vars", "tool", "emit",
    "Result", "Route", "llm",
    "App", "AppConfig",
]

try:
    from importlib.metadata import version, PackageNotFoundError  
except Exception:  
    version = None
    PackageNotFoundError = Exception 

try:
    __version__ = version("aigraph") if version else "0.0.0+local"
except PackageNotFoundError:
    __version__ = "0.0.0+local"