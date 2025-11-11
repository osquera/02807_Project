from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL, Logger
from typing import Optional

from preprocessing.helpers.util_logging import setup_logging

from student_project_template.helpers.example_imports import verify_imports

def main(do_verify_imports: bool = False, logger: Optional[Logger] = None) -> None:
    """
    Main function for the student project template.
    
    Args:
        do_verify_imports (bool): If True, verifies that all example imports work correctly.
        logger (Optional[Logger]): Logger instance for logging messages. If None, a default logger will be used.
        
    Returns:
        None
    """
    if do_verify_imports:
        verify_imports(logger)
        
    if logger is not None:
        logger.info("Hello from student-project-template!")
        
    return None
    
if __name__ == "__main__":
    # Sets up logging to console and file (if log_path is provided)
    logger: Logger = setup_logging(level = INFO, log_path=None)
    main(do_verify_imports = True, logger=logger)