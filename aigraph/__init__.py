from .api import App, Blueprint, AppConfig, Message, Context

__all__ = ["App", "Blueprint", "AppConfig", "Message", "Context"]

try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("aigraph") if version else "0.0.0+local"
except PackageNotFoundError:
    __version__ = "0.0.0+local"
