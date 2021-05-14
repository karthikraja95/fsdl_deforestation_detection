# type: ignore[attr-defined]
"""Predicting deforestation from Satellite Images. Final Project for Full Stack Deep Learning. Authors Karthik Bhaskar and Andre Ferreira"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
