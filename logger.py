import logging
import sys
import os
from datetime import datetime

def setup_logger(logger_name, log_file=None, level=logging.INFO, console_output=False):
    """
    Sets up a custom logger.

    Args:
        logger_name (str): The name for the logger.
        log_file (str, optional): The file path to save the logs. 
                                  If None, logs will only be sent to the console.
                                  Defaults to None.
        level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG).
                               Defaults to logging.INFO.
        console_output (bool, optional): If True, logs will also be sent to the console.
                                         Defaults to True.


    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get the logger and prevent propagation to the root logger
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # Set the logging level
    logger.setLevel(level)

    # If the logger already has handlers, do not add more to avoid duplicate logs
    if logger.hasHandlers():
        return logger

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- File Handler ---
    # Create a handler for file output if a log file is specified
    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # --- Console Handler (Conditional) ---
    # Create a handler for console output (StreamHandler)
    # Only add the stream handler if console_output is True
    if console_output:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

# --- Example of how to use this logger in another script ---
if __name__ == '__main__':
    # This block will only run when logger.py is executed directly
    # It serves as a simple demonstration of the setup_logger function

    # 1. Create a logger that logs to both a file and the console
    log_file_name = f"example_log_{datetime.now().strftime('%Y%m%d')}.log"
    main_logger = setup_logger("MainAppLogger", log_file=log_file_name, level=logging.DEBUG)

    main_logger.debug("This is a debug message.")
    main_logger.info("This is an info message.")
    main_logger.warning("This is a warning message.")
    main_logger.error("This is an error message.")
    main_logger.critical("This is a critical message.")

    # 2. Create a logger that only logs to the console
    console_only_logger = setup_logger("ConsoleOnlyLogger")
    console_only_logger.info("This message will only appear on the console.")
    
    # 3. Demonstrate that getting the same logger returns the same instance
    another_main_logger = setup_logger("MainAppLogger")
    another_main_logger.info("This log comes from the same logger instance, so no duplicate handlers are added.")