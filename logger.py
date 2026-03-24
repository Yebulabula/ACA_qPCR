import logging
import os
import sys
from datetime import datetime

class TrainingLogger:
    """Centralized logging system for ML training"""
    
    def __init__(self, output_dir, experiment_name=None):
        """
        Initialize logger with file and console output
        
        Args:
            output_dir: Directory to save log files
            experiment_name: Optional name for the experiment
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        # Set up main log file
        self.log_file = os.path.join(output_dir, f"{experiment_name}_training.log")
        
        # Set up root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates on reinitialization
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create handlers
        file_handler = logging.FileHandler(self.log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Store config for logging model configurations
        self.config_file = os.path.join(output_dir, f"{experiment_name}_config.log")
        
        self.logger.info(f"Logger initialized. Logs will be saved to: {self.log_file}")
    
    def log_config(self, config_dict, section_name="Configuration"):
        """Log configuration to separate config file"""
        with open(self.config_file, 'a') as f:
            f.write(f"\n\n{'='*20} {section_name} {'='*20}\n")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if isinstance(config_dict, dict):
                for key, value in config_dict.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(str(config_dict))
        
        self.logger.info(f"{section_name} logged to {self.config_file}")
    
    def info(self, message):
        """Log info level message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning level message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error level message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug level message"""
        self.logger.debug(message)
    
    def log_training(self, epoch, iteration, total_iterations, **metrics):
        """Log training metrics in a consistent format"""
        metrics_str = ', '.join([f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in metrics.items()])
        
        if epoch is not None:
            message = f"Epoch {epoch}, Iter [{iteration}/{total_iterations}] - {metrics_str}"
        else:
            message = f"Iter [{iteration}/{total_iterations}] - {metrics_str}"
            
        self.logger.info(message)
    
    def log_evaluation(self, phase, epoch=None, **metrics):
        """Log evaluation metrics in a consistent format"""
        metrics_str = ', '.join([f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in metrics.items()])
        
        if epoch is not None:
            message = f"[{phase}] Epoch {epoch} - {metrics_str}"
        else:
            message = f"[{phase}] - {metrics_str}"
            
        self.logger.info(message)
    
    def log_model_saving(self, model_path, metrics=None):
        """Log when a model is saved"""
        message = f"Model saved to: {model_path}"
        if metrics:
            metrics_str = ', '.join([f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in metrics.items()])
            message += f" - Best metrics: {metrics_str}"
        self.logger.info(message)