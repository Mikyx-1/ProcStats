"""
procstats - Monitor resource usage of Python processes and functions
"""

# Import the main monitoring function from scripts
try:
    from .scripts.full_monitoring import monitor_function_resources
except ImportError:
    # Fallback import path
    try:
        from procstats.scripts.full_monitoring import monitor_function_resources
    except ImportError:
        # If full_monitoring doesn't exist, you might need to adjust this import
        # based on where your main monitoring function is located
        pass

__version__ = "0.2.0"
__author__ = "Le Hoang Viet"
__email__ = "lehoangviet2k@gmail.com"

# Make the main function available at package level
try:
    __all__ = ["monitor_function_resources"]
except NameError:
    __all__ = []