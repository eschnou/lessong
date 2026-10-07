"""lessong: turn a song into a language lesson."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lessong")
except PackageNotFoundError:      # running from a source checkout that was never installed
    __version__ = "0.0.0+local"
