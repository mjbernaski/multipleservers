"""
Version information for Intermediator Dialog (IDi).
"""

# Semantic versioning: MAJOR.MINOR.PATCH
# MAJOR: Breaking changes
# MINOR: New features, backward compatible
# PATCH: Bug fixes, backward compatible
__version__ = "1.0.0"

# Build number: incremented with each commit/release
__build__ = 4

# Full version string
__version_full__ = f"{__version__}-build.{__build__}"

# Release date
__release_date__ = "2025-12-08"

# Module versions (for tracking individual component changes)
MODULE_VERSIONS = {
    'core': '1.0.0',
    'clients': '1.0.0',
    'clients.anthropic': '1.0.0',
    'clients.openai': '1.0.0',
    'clients.gemini': '1.0.0',
    'clients.ollama': '1.0.0',
    'socketio_handlers': '1.0.0',
    'routes': '1.0.0',
    'tts': '1.0.0',
    'pdf_generator': '1.0.0',
}


def get_version():
    """Return the full version string."""
    return __version_full__


def get_version_info():
    """Return detailed version information as a dict."""
    return {
        'version': __version__,
        'build': __build__,
        'full_version': __version_full__,
        'release_date': __release_date__,
        'modules': MODULE_VERSIONS,
    }
