"""Module to implement logging configurations"""
import logging


def setup_logger(level=None, name=None):
    """Define a logger to be used extensively within smmregrid"""

    if name is None:
        name = "smmregrid"

    loglev = convert_logger(level)

    logger = logging.getLogger(name)  # Create a logger specific to your module

    if logger.handlers:
        logger.warning('Logging is already setup with name %s', name)
        if level != logging.getLevelName(logger.getEffectiveLevel()):
            logger.setLevel(loglev)
            logger.info('Updating the log_level to %s', loglev)
        return logger

    # avoid duplication/propagation of loggers
    logger.propagate = False

    logger.setLevel(loglev)  # Set the desired log level

    # Create a formatter to specify the log format
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)8s -> %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Create a handler for the logger
    handler = logging.StreamHandler()
    #handler.setLevel(loglev)  # Set the desired log level for the handler
    handler.setFormatter(formatter)  # Assign the formatter to the handler
    logger.addHandler(handler)  # Add the handler to the logger

    return logger

def convert_logger(loglev=None):
    """Convert a string or integer to a valid logging level"""

    # Default log level for the AQUA framework
    loglev_default = "WARNING"

    # If loglev is a string, convert it to uppercase
    if isinstance(loglev, str):
        loglev = loglev.upper()

    # If loglev is an integer, convert it to a string
    elif isinstance(loglev, int):
        loglev = logging.getLevelName(loglev)

    # If loglev is None, set it to the default log level
    elif loglev is None:
        loglev = loglev_default

    # If loglev is of an unsupported type, raise a ValueError
    else:
        raise ValueError('Invalid log level type. Must be a string or an integer.')

    # Check if the log level exists and retrieve its integer value
    loglev_int = getattr(logging, loglev, None)

    # If loglev_int is None, the log level doesn't exist
    if loglev_int is None:
        logging.warning("Invalid logging level '%s' specified. Setting it back to default '%s'.",
                        loglev, loglev_default)
        loglev = loglev_default

    return loglev