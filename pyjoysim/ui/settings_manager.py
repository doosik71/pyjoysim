"""
Settings Manager Implementation

This module provides comprehensive settings management with:
- Persistent configuration storage
- User preference management
- Default value handling
- Settings validation and migration
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from dataclasses import dataclass, asdict, fields
from enum import Enum
import copy

from pyjoysim.rendering import Color


T = TypeVar('T')


class SettingType(Enum):
    """Setting value types."""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    COLOR = "color"
    LIST = "list"
    DICT = "dict"


@dataclass
class SettingDefinition:
    """Definition of a configuration setting."""
    key: str
    setting_type: SettingType
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    valid_values: Optional[List[Any]] = None
    description: str = ""
    category: str = "general"
    requires_restart: bool = False


@dataclass
class GraphicsSettings:
    """Graphics-related settings."""
    window_width: int = 1024
    window_height: int = 768
    fullscreen: bool = False
    vsync: bool = True
    target_fps: int = 60
    quality_level: str = "medium"  # low, medium, high, ultra
    show_fps: bool = False
    show_debug_overlay: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphicsSettings':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class AudioSettings:
    """Audio-related settings."""
    master_volume: float = 1.0
    effects_volume: float = 1.0
    music_volume: float = 0.7
    mute_on_focus_loss: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioSettings':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class InputSettings:
    """Input-related settings."""
    joystick_enabled: bool = True
    joystick_deadzone: float = 0.1
    joystick_sensitivity: float = 1.0
    invert_y_axis: bool = False
    keyboard_repeat_delay: int = 500
    keyboard_repeat_rate: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputSettings':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class SimulationSettings:
    """Simulation-related settings."""
    physics_timestep: float = 0.016  # 60 FPS
    physics_substeps: int = 8
    collision_detection_mode: str = "continuous"  # discrete, continuous
    gravity_enabled: bool = True
    auto_save_interval: int = 300  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationSettings':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


@dataclass
class UISettings:
    """User interface settings."""
    theme: str = "dark"  # dark, light
    font_size: int = 14
    show_tooltips: bool = True
    animation_speed: float = 1.0
    auto_hide_panels: bool = False
    language: str = "ko"  # ko, en
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UISettings':
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})


class SettingsManager:
    """
    Comprehensive settings management system.
    
    Features:
    - Persistent JSON-based storage
    - Type-safe setting definitions
    - Default value management
    - Setting validation
    - Change notifications
    - Settings migration
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        # Configuration directory
        if config_dir is None:
            config_dir = Path.home() / ".pyjoysim"
        
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "settings.json"
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Settings categories
        self.graphics = GraphicsSettings()
        self.audio = AudioSettings()
        self.input = InputSettings()
        self.simulation = SimulationSettings()
        self.ui = UISettings()
        
        # Change callbacks
        self.change_callbacks = {}  # Dict[str, List[Callable]]
        
        # Setting definitions for validation
        self.setting_definitions = self._create_setting_definitions()
        
        # Load existing settings
        self.load_settings()
    
    def _create_setting_definitions(self) -> Dict[str, SettingDefinition]:
        """Create setting definitions for validation."""
        definitions = {}
        
        # Graphics settings
        definitions.update({
            "graphics.window_width": SettingDefinition(
                "graphics.window_width", SettingType.INTEGER, 1024,
                min_value=640, max_value=3840,
                description="창 너비", category="graphics", requires_restart=True
            ),
            "graphics.window_height": SettingDefinition(
                "graphics.window_height", SettingType.INTEGER, 768,
                min_value=480, max_value=2160,
                description="창 높이", category="graphics", requires_restart=True
            ),
            "graphics.fullscreen": SettingDefinition(
                "graphics.fullscreen", SettingType.BOOLEAN, False,
                description="전체화면 모드", category="graphics", requires_restart=True
            ),
            "graphics.target_fps": SettingDefinition(
                "graphics.target_fps", SettingType.INTEGER, 60,
                min_value=30, max_value=144,
                description="목표 FPS", category="graphics"
            ),
            "graphics.quality_level": SettingDefinition(
                "graphics.quality_level", SettingType.STRING, "medium",
                valid_values=["low", "medium", "high", "ultra"],
                description="그래픽 품질", category="graphics"
            )
        })
        
        # Audio settings
        definitions.update({
            "audio.master_volume": SettingDefinition(
                "audio.master_volume", SettingType.FLOAT, 1.0,
                min_value=0.0, max_value=1.0,
                description="마스터 볼륨", category="audio"
            ),
            "audio.effects_volume": SettingDefinition(
                "audio.effects_volume", SettingType.FLOAT, 1.0,
                min_value=0.0, max_value=1.0,
                description="효과음 볼륨", category="audio"
            )
        })
        
        # Input settings
        definitions.update({
            "input.joystick_deadzone": SettingDefinition(
                "input.joystick_deadzone", SettingType.FLOAT, 0.1,
                min_value=0.0, max_value=0.5,
                description="조이스틱 데드존", category="input"
            ),
            "input.joystick_sensitivity": SettingDefinition(
                "input.joystick_sensitivity", SettingType.FLOAT, 1.0,
                min_value=0.1, max_value=3.0,
                description="조이스틱 감도", category="input"
            )
        })
        
        return definitions
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value by dot-notation key."""
        parts = key.split('.')
        if len(parts) != 2:
            return default
        
        category, setting = parts
        
        if category == "graphics" and hasattr(self.graphics, setting):
            return getattr(self.graphics, setting)
        elif category == "audio" and hasattr(self.audio, setting):
            return getattr(self.audio, setting)
        elif category == "input" and hasattr(self.input, setting):
            return getattr(self.input, setting)
        elif category == "simulation" and hasattr(self.simulation, setting):
            return getattr(self.simulation, setting)
        elif category == "ui" and hasattr(self.ui, setting):
            return getattr(self.ui, setting)
        
        return default
    
    def set_setting(self, key: str, value: Any, validate: bool = True) -> bool:
        """Set a setting value by dot-notation key."""
        if validate and not self._validate_setting(key, value):
            return False
        
        parts = key.split('.')
        if len(parts) != 2:
            return False
        
        category, setting = parts
        old_value = self.get_setting(key)
        
        # Update the setting
        if category == "graphics" and hasattr(self.graphics, setting):
            setattr(self.graphics, setting, value)
        elif category == "audio" and hasattr(self.audio, setting):
            setattr(self.audio, setting, value)
        elif category == "input" and hasattr(self.input, setting):
            setattr(self.input, setting, value)
        elif category == "simulation" and hasattr(self.simulation, setting):
            setattr(self.simulation, setting, value)
        elif category == "ui" and hasattr(self.ui, setting):
            setattr(self.ui, setting, value)
        else:
            return False
        
        # Notify callbacks
        self._notify_setting_changed(key, old_value, value)
        
        return True
    
    def _validate_setting(self, key: str, value: Any) -> bool:
        """Validate a setting value."""
        if key not in self.setting_definitions:
            return True  # Allow unknown settings
        
        definition = self.setting_definitions[key]
        
        # Type validation
        if definition.setting_type == SettingType.BOOLEAN and not isinstance(value, bool):
            return False
        elif definition.setting_type == SettingType.INTEGER and not isinstance(value, int):
            return False
        elif definition.setting_type == SettingType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif definition.setting_type == SettingType.STRING and not isinstance(value, str):
            return False
        
        # Range validation
        if definition.min_value is not None and value < definition.min_value:
            return False
        if definition.max_value is not None and value > definition.max_value:
            return False
        
        # Valid values validation
        if definition.valid_values is not None and value not in definition.valid_values:
            return False
        
        return True
    
    def _notify_setting_changed(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify callbacks of setting changes."""
        if key in self.change_callbacks:
            for callback in self.change_callbacks[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    print(f"Error in setting change callback: {e}")
    
    def register_change_callback(self, key: str, callback: callable) -> None:
        """Register a callback for setting changes."""
        if key not in self.change_callbacks:
            self.change_callbacks[key] = []
        self.change_callbacks[key].append(callback)
    
    def unregister_change_callback(self, key: str, callback: callable) -> None:
        """Unregister a setting change callback."""
        if key in self.change_callbacks:
            try:
                self.change_callbacks[key].remove(callback)
            except ValueError:
                pass
    
    def save_settings(self) -> bool:
        """Save settings to file."""
        try:
            settings_data = {
                "version": "2.0",
                "graphics": self.graphics.to_dict(),
                "audio": self.audio.to_dict(),
                "input": self.input.to_dict(),
                "simulation": self.simulation.to_dict(),
                "ui": self.ui.to_dict()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Failed to save settings: {e}")
            return False
    
    def load_settings(self) -> bool:
        """Load settings from file."""
        try:
            if not self.config_file.exists():
                # Create default settings file
                return self.save_settings()
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                settings_data = json.load(f)
            
            # Migrate settings if necessary
            settings_data = self._migrate_settings(settings_data)
            
            # Load settings into objects
            if "graphics" in settings_data:
                self.graphics = GraphicsSettings.from_dict(settings_data["graphics"])
            
            if "audio" in settings_data:
                self.audio = AudioSettings.from_dict(settings_data["audio"])
            
            if "input" in settings_data:
                self.input = InputSettings.from_dict(settings_data["input"])
            
            if "simulation" in settings_data:
                self.simulation = SimulationSettings.from_dict(settings_data["simulation"])
            
            if "ui" in settings_data:
                self.ui = UISettings.from_dict(settings_data["ui"])
            
            return True
            
        except Exception as e:
            print(f"Failed to load settings: {e}")
            return False
    
    def _migrate_settings(self, settings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate settings from older versions."""
        version = settings_data.get("version", "1.0")
        
        if version == "1.0":
            # Migrate from version 1.0 to 2.0
            # Add any necessary migrations here
            settings_data["version"] = "2.0"
        
        return settings_data
    
    def reset_to_defaults(self, category: Optional[str] = None) -> None:
        """Reset settings to default values."""
        if category is None or category == "graphics":
            self.graphics = GraphicsSettings()
        
        if category is None or category == "audio":
            self.audio = AudioSettings()
        
        if category is None or category == "input":
            self.input = InputSettings()
        
        if category is None or category == "simulation":
            self.simulation = SimulationSettings()
        
        if category is None or category == "ui":
            self.ui = UISettings()
    
    def export_settings(self, file_path: Path) -> bool:
        """Export settings to a file."""
        try:
            settings_data = {
                "version": "2.0",
                "exported_at": str(time.time()),
                "graphics": self.graphics.to_dict(),
                "audio": self.audio.to_dict(),
                "input": self.input.to_dict(),
                "simulation": self.simulation.to_dict(),
                "ui": self.ui.to_dict()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, file_path: Path) -> bool:
        """Import settings from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                settings_data = json.load(f)
            
            # Validate and load imported settings
            if "graphics" in settings_data:
                imported_graphics = GraphicsSettings.from_dict(settings_data["graphics"])
                self.graphics = imported_graphics
            
            if "audio" in settings_data:
                imported_audio = AudioSettings.from_dict(settings_data["audio"])
                self.audio = imported_audio
            
            if "input" in settings_data:
                imported_input = InputSettings.from_dict(settings_data["input"])
                self.input = imported_input
            
            if "simulation" in settings_data:
                imported_simulation = SimulationSettings.from_dict(settings_data["simulation"])
                self.simulation = imported_simulation
            
            if "ui" in settings_data:
                imported_ui = UISettings.from_dict(settings_data["ui"])
                self.ui = imported_ui
            
            return True
            
        except Exception as e:
            print(f"Failed to import settings: {e}")
            return False
    
    def get_settings_summary(self) -> Dict[str, Any]:
        """Get a summary of current settings."""
        return {
            "graphics": self.graphics.to_dict(),
            "audio": self.audio.to_dict(),
            "input": self.input.to_dict(),
            "simulation": self.simulation.to_dict(),
            "ui": self.ui.to_dict()
        }
    
    def validate_all_settings(self) -> List[str]:
        """Validate all current settings and return list of errors."""
        errors = []
        
        for key, definition in self.setting_definitions.items():
            current_value = self.get_setting(key)
            if not self._validate_setting(key, current_value):
                errors.append(f"Invalid value for {key}: {current_value}")
        
        return errors


# Global settings instance
_settings_manager = None


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def save_settings() -> bool:
    """Save current settings to file."""
    return get_settings_manager().save_settings()


def load_settings() -> bool:
    """Load settings from file."""
    return get_settings_manager().load_settings()


# Import time for export functionality
import time