import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union
from value_investor.config import LOG_DIR, OUTPUT_DIR

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: The log record to format.
            
        Returns:
            JSON formatted string of the log record.
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Include exc_info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

class MetricsLogger:
    """Handler for logging metrics and results to JSON files"""
    
    def __init__(self):
        """Initialize the metrics logger."""
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def save_metrics(
        self, 
        metrics: Dict[str, Any], 
        category: str,
        strategy: str
    ) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save.
            category: Category of metrics (e.g., 'training', 'evaluation').
            strategy: Name of the strategy used.
        """
        filename = f"{self.timestamp}_{strategy}_{category}_metrics.json"
        filepath = OUTPUT_DIR / filename
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'category': category,
            'metrics': metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=4)
            
    def save_results(
        self, 
        results: Dict[str, Any],
        experiment_name: str
    ) -> None:
        """
        Save experiment results to a JSON file.
        
        Args:
            results: Dictionary of results to save.
            experiment_name: Name of the experiment.
        """
        filename = f"{self.timestamp}_{experiment_name}_results.json"
        filepath = OUTPUT_DIR / filename
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment': experiment_name,
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=4)

def setup_logger(name: str = 'value_investor') -> logging.Logger:
    """
    Set up and configure the logger.
    
    Args:
        name: Name for the logger.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_log.json"
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    json_formatter = JSONFormatter()
    
    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(json_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create global instances
logger = setup_logger()
metrics_logger = MetricsLogger()