from .api import App, AppConfig, Message

__all__ = [
    "App",
    "AppConfig",
    "Message",
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
