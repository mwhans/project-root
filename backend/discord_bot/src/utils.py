import sys
from loguru import logger
from pathlib import Path
import json
from typing import Dict, Any
from datetime import datetime

def setup_logging():
    """Sets up logging configuration"""
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Add handlers for both file and console output
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Console handler with color
    logger.add(sys.stdout, format=log_format, level="INFO", colorize=True)
    
    # File handler with rotation
    log_file = log_dir / "discord_bot_{time}.log"
    logger.add(
        str(log_file),
        rotation="500 MB",
        retention="10 days",
        format=log_format,
        level="DEBUG",
        compression="zip"
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def create_backup_filename(guild_id: int, backup_type: str) -> str:
    """Creates a unique backup filename"""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return f"backups/{backup_type}_{guild_id}_{timestamp}.json"

def ensure_directory(path: str):
    """Ensures a directory exists, creating it if necessary"""
    Path(path).mkdir(parents=True, exist_ok=True)

def format_time_delta(seconds: float) -> str:
    """Formats a time delta in seconds to a human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours" 