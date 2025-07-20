"""
Configuration management system for PyJoySim.

This module provides centralized configuration management with support for
JSON files, environment variables, and runtime overrides.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Config:
    """
    Central configuration manager for PyJoySim.
    
    Handles loading configuration from multiple sources:
    1. Default values (hardcoded)
    2. User config file (~/.pyjoysim/config.json)
    3. Project config file (./config.json)
    4. Environment variables (PYJOYSIM_*)
    5. Runtime overrides
    """
    
    # Default configuration values
    _defaults = {
        "app": {
            "name": "PyJoySim",
            "version": "0.1.0",
            "debug": False,
            "log_level": "INFO",
            "fps_target": 60,
            "window": {
                "width": 1280,
                "height": 720,
                "fullscreen": False,
                "vsync": True,
            }
        },
        "input": {
            "max_joysticks": 4,
            "deadzone": 0.1,
            "hotplug_enabled": True,
            "keyboard_fallback": True,
        },
        "physics": {
            "timestep": 1.0 / 120.0,  # 120Hz physics
            "iterations": 10,
            "gravity": -9.81,
            "sleep_threshold": 0.1,
        },
        "rendering": {
            "antialias": True,
            "shadow_quality": "medium",
            "texture_quality": "high",
            "particle_count": 1000,
        },
        "audio": {
            "master_volume": 0.8,
            "sfx_volume": 0.7,
            "music_volume": 0.5,
            "mute": False,
        },
        "simulation": {
            "auto_pause": True,
            "performance_mode": False,
            "recording_enabled": False,
        }
    }
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to specific config file
        """
        self._config: Dict[str, Any] = {}
        self._config_file = config_file
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from all sources in priority order."""
        # Start with defaults
        self._config = self._deep_copy(self._defaults)
        
        # Load from user config file
        user_config_path = self._get_user_config_path()
        if user_config_path.exists():
            self._load_from_file(user_config_path)
        
        # Load from project config file
        if self._config_file:
            config_path = Path(self._config_file)
            if config_path.exists():
                self._load_from_file(config_path)
        else:
            # Default project config
            project_config = Path("config.json")
            if project_config.exists():
                self._load_from_file(project_config)
        
        # Load from environment variables
        self._load_from_env()
    
    def _get_user_config_path(self) -> Path:
        """Get the user-specific config file path."""
        home = Path.home()
        config_dir = home / ".pyjoysim"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.json"
    
    def _load_from_file(self, file_path: Path) -> None:
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            self._deep_merge(self._config, file_config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Log warning but continue with existing config
            print(f"Warning: Could not load config from {file_path}: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_prefix = "PYJOYSIM_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert PYJOYSIM_APP_DEBUG to ["app", "debug"]
                config_key = key[len(env_prefix):].lower().split('_')
                self._set_nested_value(self._config, config_key, self._parse_env_value(value))
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first (for complex values)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Handle common boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], keys: list, value: Any) -> None:
        """Set a nested configuration value."""
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy configuration object."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        else:
            return obj
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "app.window.width")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        self._set_nested_value(self._config, keys, value)
    
    def save(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            file_path: Optional path to save to, defaults to user config
        """
        if file_path is None:
            file_path = self._get_user_config_path()
        else:
            file_path = Path(file_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        self._load_configuration()
    
    def get_all(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self._deep_copy(self._config)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self._deep_copy(self._defaults)
    
    # Convenience properties for commonly used values
    @property
    def debug(self) -> bool:
        """Debug mode flag."""
        return self.get("app.debug", False)
    
    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.get("app.log_level", "INFO")
    
    @property
    def fps_target(self) -> int:
        """Target frames per second."""
        return self.get("app.fps_target", 60)
    
    @property
    def window_size(self) -> tuple:
        """Window size as (width, height) tuple."""
        return (
            self.get("app.window.width", 1280),
            self.get("app.window.height", 720)
        )
    
    @property
    def fullscreen(self) -> bool:
        """Fullscreen mode flag."""
        return self.get("app.window.fullscreen", False)


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None