"""
Settings module for PyJoySim.

Provides convenient access to configuration settings with validation and type hints.
"""

from typing import Dict, Any, Optional
from .config import get_config, Config


class Settings:
    """
    High-level settings interface with validation and type safety.
    
    This class provides a more convenient and type-safe way to access
    configuration values compared to the raw Config class.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize settings with optional config instance.
        
        Args:
            config: Optional Config instance, uses global if not provided
        """
        self._config = config or get_config()
    
    # Application settings
    @property
    def app_name(self) -> str:
        """Application name."""
        return self._config.get("app.name", "PyJoySim")
    
    @property
    def app_version(self) -> str:
        """Application version."""
        return self._config.get("app.version", "0.1.0")
    
    @property
    def debug_mode(self) -> bool:
        """Debug mode enabled."""
        return self._config.get("app.debug", False)
    
    @property
    def log_level(self) -> str:
        """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
        level = self._config.get("app.log_level", "INFO").upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        return level if level in valid_levels else "INFO"
    
    @property
    def target_fps(self) -> int:
        """Target frames per second."""
        fps = self._config.get("app.fps_target", 60)
        return max(1, min(fps, 240))  # Clamp between 1 and 240
    
    # Window settings
    @property
    def window_width(self) -> int:
        """Window width in pixels."""
        width = self._config.get("app.window.width", 1280)
        return max(640, min(width, 7680))  # Clamp between 640 and 8K
    
    @property
    def window_height(self) -> int:
        """Window height in pixels."""
        height = self._config.get("app.window.height", 720)
        return max(480, min(height, 4320))  # Clamp between 480 and 8K
    
    @property
    def window_size(self) -> tuple[int, int]:
        """Window size as (width, height) tuple."""
        return (self.window_width, self.window_height)
    
    @property
    def fullscreen(self) -> bool:
        """Fullscreen mode enabled."""
        return self._config.get("app.window.fullscreen", False)
    
    @property
    def vsync_enabled(self) -> bool:
        """Vertical sync enabled."""
        return self._config.get("app.window.vsync", True)
    
    # Input settings
    @property
    def max_joysticks(self) -> int:
        """Maximum number of joysticks to support."""
        count = self._config.get("input.max_joysticks", 4)
        return max(1, min(count, 8))  # Clamp between 1 and 8
    
    @property
    def joystick_deadzone(self) -> float:
        """Joystick analog stick deadzone (0.0 - 1.0)."""
        deadzone = self._config.get("input.deadzone", 0.1)
        return max(0.0, min(deadzone, 0.9))
    
    @property
    def hotplug_enabled(self) -> bool:
        """Joystick hotplug support enabled."""
        return self._config.get("input.hotplug_enabled", True)
    
    @property
    def keyboard_fallback(self) -> bool:
        """Keyboard input fallback enabled."""
        return self._config.get("input.keyboard_fallback", True)
    
    # Physics settings
    @property
    def physics_timestep(self) -> float:
        """Physics simulation timestep in seconds."""
        timestep = self._config.get("physics.timestep", 1.0 / 120.0)
        return max(1.0 / 1000.0, min(timestep, 1.0 / 30.0))  # 30Hz to 1000Hz
    
    @property
    def physics_iterations(self) -> int:
        """Physics solver iterations per step."""
        iterations = self._config.get("physics.iterations", 10)
        return max(1, min(iterations, 50))
    
    @property
    def gravity(self) -> float:
        """Gravity acceleration in m/s²."""
        return self._config.get("physics.gravity", -9.81)
    
    @property
    def sleep_threshold(self) -> float:
        """Physics body sleep threshold."""
        threshold = self._config.get("physics.sleep_threshold", 0.1)
        return max(0.01, min(threshold, 10.0))
    
    # Rendering settings
    @property
    def antialias_enabled(self) -> bool:
        """Anti-aliasing enabled."""
        return self._config.get("rendering.antialias", True)
    
    @property
    def shadow_quality(self) -> str:
        """Shadow quality setting."""
        quality = self._config.get("rendering.shadow_quality", "medium").lower()
        valid_qualities = {"low", "medium", "high", "ultra"}
        return quality if quality in valid_qualities else "medium"
    
    @property
    def texture_quality(self) -> str:
        """Texture quality setting."""
        quality = self._config.get("rendering.texture_quality", "high").lower()
        valid_qualities = {"low", "medium", "high", "ultra"}
        return quality if quality in valid_qualities else "high"
    
    @property
    def max_particles(self) -> int:
        """Maximum number of particles."""
        count = self._config.get("rendering.particle_count", 1000)
        return max(100, min(count, 10000))
    
    # Audio settings
    @property
    def master_volume(self) -> float:
        """Master volume (0.0 - 1.0)."""
        volume = self._config.get("audio.master_volume", 0.8)
        return max(0.0, min(volume, 1.0))
    
    @property
    def sfx_volume(self) -> float:
        """Sound effects volume (0.0 - 1.0)."""
        volume = self._config.get("audio.sfx_volume", 0.7)
        return max(0.0, min(volume, 1.0))
    
    @property
    def music_volume(self) -> float:
        """Music volume (0.0 - 1.0)."""
        volume = self._config.get("audio.music_volume", 0.5)
        return max(0.0, min(volume, 1.0))
    
    @property
    def audio_muted(self) -> bool:
        """Audio muted."""
        return self._config.get("audio.mute", False)
    
    # Simulation settings
    @property
    def auto_pause(self) -> bool:
        """Auto-pause simulation when window loses focus."""
        return self._config.get("simulation.auto_pause", True)
    
    @property
    def performance_mode(self) -> bool:
        """Performance mode enabled (lower quality for better FPS)."""
        return self._config.get("simulation.performance_mode", False)
    
    @property
    def recording_enabled(self) -> bool:
        """Recording/replay functionality enabled."""
        return self._config.get("simulation.recording_enabled", False)
    
    # Convenience methods
    def update_setting(self, key: str, value: Any) -> None:
        """
        Update a setting value.
        
        Args:
            key: Setting key in dot notation
            value: New value
        """
        self._config.set(key, value)
    
    def save_settings(self) -> None:
        """Save current settings to user config file."""
        self._config.save()
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self._config.reset_to_defaults()
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary."""
        return self._config.get_all()
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.debug_mode or self.log_level == "DEBUG"
    
    def get_physics_hz(self) -> float:
        """Get physics update frequency in Hz."""
        return 1.0 / self.physics_timestep
    
    def get_display_info(self) -> Dict[str, Any]:
        """Get display-related settings."""
        return {
            "width": self.window_width,
            "height": self.window_height,
            "fullscreen": self.fullscreen,
            "vsync": self.vsync_enabled,
            "fps_target": self.target_fps,
        }
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """Get quality-related settings."""
        return {
            "antialias": self.antialias_enabled,
            "shadow_quality": self.shadow_quality,
            "texture_quality": self.texture_quality,
            "max_particles": self.max_particles,
            "performance_mode": self.performance_mode,
        }


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings_instance
    _settings_instance = settings


def reset_settings() -> None:
    """Reset the global settings instance."""
    global _settings_instance
    _settings_instance = None