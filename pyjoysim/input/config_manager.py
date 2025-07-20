"""
Input configuration management for PyJoySim.

This module handles joystick-specific configuration including custom key mappings,
axis settings, and input profiles.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .input_processor import AxisMapping, ButtonMapping, AxisType
from ..config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ConfigurationError, InvalidConfigValueError


class InputProfile(Enum):
    """Predefined input profiles."""
    DEFAULT = "default"
    RACING = "racing"
    FLIGHT = "flight"
    ROBOTICS = "robotics"
    GAMING = "gaming"
    CUSTOM = "custom"


@dataclass
class JoystickProfile:
    """Complete configuration profile for a joystick."""
    name: str
    description: str
    profile_type: InputProfile
    
    # Axis mappings
    axis_mappings: Dict[int, Dict[str, Any]]  # Serializable version of AxisMapping
    
    # Button mappings
    button_mappings: Dict[int, Dict[str, Any]]  # Serializable version of ButtonMapping
    
    # General settings
    deadzone: float = 0.1
    sensitivity: float = 1.0
    invert_y: bool = False
    
    # Metadata
    version: str = "1.0"
    created_for_guid: Optional[str] = None
    last_modified: Optional[str] = None
    
    def to_axis_mappings(self) -> Dict[int, AxisMapping]:
        """Convert stored axis mappings to AxisMapping objects."""
        mappings = {}
        for axis_id, config in self.axis_mappings.items():
            mappings[int(axis_id)] = AxisMapping(
                axis_id=config["axis_id"],
                axis_type=AxisType(config["axis_type"]),
                invert=config.get("invert", False),
                scale=config.get("scale", 1.0),
                offset=config.get("offset", 0.0),
                curve=config.get("curve", "linear")
            )
        return mappings
    
    def to_button_mappings(self) -> Dict[int, ButtonMapping]:
        """Convert stored button mappings to ButtonMapping objects."""
        mappings = {}
        for button_id, config in self.button_mappings.items():
            mappings[int(button_id)] = ButtonMapping(
                button_id=config["button_id"],
                repeat_enabled=config.get("repeat_enabled", False),
                repeat_delay=config.get("repeat_delay", 0.5),
                repeat_rate=config.get("repeat_rate", 0.1)
            )
        return mappings
    
    @classmethod
    def from_mappings(cls, 
                     name: str,
                     description: str,
                     profile_type: InputProfile,
                     axis_mappings: Dict[int, AxisMapping],
                     button_mappings: Dict[int, ButtonMapping],
                     **kwargs) -> 'JoystickProfile':
        """Create JoystickProfile from mapping objects."""
        # Convert AxisMapping objects to serializable dicts
        axis_dict = {}
        for axis_id, mapping in axis_mappings.items():
            axis_dict[str(axis_id)] = {
                "axis_id": mapping.axis_id,
                "axis_type": mapping.axis_type.value,
                "invert": mapping.invert,
                "scale": mapping.scale,
                "offset": mapping.offset,
                "curve": mapping.curve
            }
        
        # Convert ButtonMapping objects to serializable dicts
        button_dict = {}
        for button_id, mapping in button_mappings.items():
            button_dict[str(button_id)] = {
                "button_id": mapping.button_id,
                "repeat_enabled": mapping.repeat_enabled,
                "repeat_delay": mapping.repeat_delay,
                "repeat_rate": mapping.repeat_rate
            }
        
        return cls(
            name=name,
            description=description,
            profile_type=profile_type,
            axis_mappings=axis_dict,
            button_mappings=button_dict,
            **kwargs
        )


class InputConfigManager:
    """
    Manager for joystick input configurations and profiles.
    
    Handles:
    - Loading and saving joystick profiles
    - Custom key mappings
    - Predefined input profiles for different simulation types
    - Per-joystick configuration storage
    """
    
    def __init__(self):
        """Initialize the input configuration manager."""
        self.logger = get_logger("input_config")
        self.config = get_config()
        
        # Configuration storage
        self._profiles: Dict[str, JoystickProfile] = {}
        self._joystick_assignments: Dict[str, str] = {}  # GUID -> profile_name
        
        # Paths
        self._config_dir = self._get_config_directory()
        self._profiles_dir = self._config_dir / "input_profiles"
        self._mappings_file = self._config_dir / "joystick_mappings.json"
        
        # Ensure directories exist
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing configurations
        self._load_profiles()
        self._load_joystick_assignments()
        
        self.logger.debug("InputConfigManager initialized", extra={
            "config_dir": str(self._config_dir),
            "profiles_loaded": len(self._profiles)
        })
    
    def _get_config_directory(self) -> Path:
        """Get the configuration directory for input settings."""
        home = Path.home()
        config_dir = home / ".pyjoysim" / "input"
        return config_dir
    
    def _load_profiles(self) -> None:
        """Load all input profiles from disk."""
        # Load built-in profiles first
        self._create_builtin_profiles()
        
        # Load custom profiles from files
        if self._profiles_dir.exists():
            for profile_file in self._profiles_dir.glob("*.json"):
                try:
                    self._load_profile_from_file(profile_file)
                except Exception as e:
                    self.logger.warning("Failed to load profile", extra={
                        "file": str(profile_file),
                        "error": str(e)
                    })
    
    def _create_builtin_profiles(self) -> None:
        """Create built-in input profiles."""
        # Default profile
        default_axis_mappings = {
            0: AxisMapping(0, AxisType.STICK_X),
            1: AxisMapping(1, AxisType.STICK_Y, invert=True),  # Invert Y for typical game controls
            2: AxisMapping(2, AxisType.TRIGGER),
            3: AxisMapping(3, AxisType.STICK_X),
            4: AxisMapping(4, AxisType.STICK_Y, invert=True),
            5: AxisMapping(5, AxisType.TRIGGER),
        }
        
        default_button_mappings = {
            i: ButtonMapping(i) for i in range(16)  # Standard 16 buttons
        }
        
        default_profile = JoystickProfile.from_mappings(
            name="Default",
            description="Standard gamepad configuration",
            profile_type=InputProfile.DEFAULT,
            axis_mappings=default_axis_mappings,
            button_mappings=default_button_mappings
        )
        self._profiles["default"] = default_profile
        
        # Racing profile
        racing_axis_mappings = {
            0: AxisMapping(0, AxisType.LINEAR),  # Steering
            1: AxisMapping(1, AxisType.TRIGGER),  # Throttle
            2: AxisMapping(2, AxisType.TRIGGER),  # Brake
        }
        
        racing_profile = JoystickProfile.from_mappings(
            name="Racing",
            description="Optimized for car simulation",
            profile_type=InputProfile.RACING,
            axis_mappings=racing_axis_mappings,
            button_mappings=default_button_mappings,
            deadzone=0.05,  # Lower deadzone for precision
            sensitivity=1.2
        )
        self._profiles["racing"] = racing_profile
        
        # Flight profile
        flight_axis_mappings = {
            0: AxisMapping(0, AxisType.STICK_X),  # Roll
            1: AxisMapping(1, AxisType.STICK_Y, invert=True),  # Pitch
            2: AxisMapping(2, AxisType.LINEAR),  # Yaw (rudder)
            3: AxisMapping(3, AxisType.LINEAR),  # Throttle
        }
        
        flight_profile = JoystickProfile.from_mappings(
            name="Flight",
            description="Optimized for drone/aircraft simulation",
            profile_type=InputProfile.FLIGHT,
            axis_mappings=flight_axis_mappings,
            button_mappings=default_button_mappings,
            deadzone=0.02,  # Very low deadzone for flight
            sensitivity=0.8,  # Lower sensitivity for stability
            invert_y=True
        )
        self._profiles["flight"] = flight_profile
        
        # Robotics profile
        robotics_axis_mappings = {
            0: AxisMapping(0, AxisType.LINEAR),  # X movement
            1: AxisMapping(1, AxisType.LINEAR),  # Y movement
            2: AxisMapping(2, AxisType.LINEAR),  # Z movement
            3: AxisMapping(3, AxisType.LINEAR),  # Rotation
        }
        
        robotics_button_mappings = {
            i: ButtonMapping(i, repeat_enabled=True, repeat_delay=0.3, repeat_rate=0.1) 
            for i in range(16)
        }
        
        robotics_profile = JoystickProfile.from_mappings(
            name="Robotics",
            description="Optimized for robot control",
            profile_type=InputProfile.ROBOTICS,
            axis_mappings=robotics_axis_mappings,
            button_mappings=robotics_button_mappings,
            deadzone=0.1,
            sensitivity=0.5  # Slower for precision
        )
        self._profiles["robotics"] = robotics_profile
        
        self.logger.debug("Built-in profiles created", extra={
            "profile_count": len(self._profiles)
        })
    
    def _load_profile_from_file(self, file_path: Path) -> None:
        """Load a profile from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate required fields
        required_fields = ["name", "description", "profile_type", "axis_mappings", "button_mappings"]
        for field in required_fields:
            if field not in data:
                raise ConfigurationError(f"Missing required field '{field}' in profile {file_path}")
        
        # Create profile
        profile = JoystickProfile(
            name=data["name"],
            description=data["description"],
            profile_type=InputProfile(data["profile_type"]),
            axis_mappings=data["axis_mappings"],
            button_mappings=data["button_mappings"],
            deadzone=data.get("deadzone", 0.1),
            sensitivity=data.get("sensitivity", 1.0),
            invert_y=data.get("invert_y", False),
            version=data.get("version", "1.0"),
            created_for_guid=data.get("created_for_guid"),
            last_modified=data.get("last_modified")
        )
        
        profile_id = file_path.stem
        self._profiles[profile_id] = profile
        
        self.logger.debug("Profile loaded from file", extra={
            "profile_id": profile_id,
            "profile_name": profile.name,
            "file": str(file_path)
        })
    
    def _load_joystick_assignments(self) -> None:
        """Load joystick-to-profile assignments."""
        if not self._mappings_file.exists():
            return
        
        try:
            with open(self._mappings_file, 'r', encoding='utf-8') as f:
                self._joystick_assignments = json.load(f)
            
            self.logger.debug("Joystick assignments loaded", extra={
                "assignment_count": len(self._joystick_assignments)
            })
        except Exception as e:
            self.logger.warning("Failed to load joystick assignments", extra={
                "error": str(e)
            })
    
    def save_profile(self, profile_id: str, profile: JoystickProfile) -> None:
        """
        Save a profile to disk.
        
        Args:
            profile_id: Unique identifier for the profile
            profile: Profile to save
        """
        try:
            # Update last modified time
            from datetime import datetime
            profile.last_modified = datetime.now().isoformat()
            
            # Convert to dictionary
            profile_data = asdict(profile)
            
            # Save to file
            file_path = self._profiles_dir / f"{profile_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            
            # Store in memory
            self._profiles[profile_id] = profile
            
            self.logger.info("Profile saved", extra={
                "profile_id": profile_id,
                "profile_name": profile.name,
                "file": str(file_path)
            })
            
        except Exception as e:
            self.logger.error("Failed to save profile", extra={
                "profile_id": profile_id,
                "error": str(e)
            })
            raise ConfigurationError(f"Failed to save profile '{profile_id}': {e}")
    
    def load_profile(self, profile_id: str) -> Optional[JoystickProfile]:
        """
        Load a profile by ID.
        
        Args:
            profile_id: ID of the profile to load
            
        Returns:
            JoystickProfile if found, None otherwise
        """
        return self._profiles.get(profile_id)
    
    def get_all_profiles(self) -> Dict[str, JoystickProfile]:
        """Get all available profiles."""
        return self._profiles.copy()
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile.
        
        Args:
            profile_id: ID of the profile to delete
            
        Returns:
            True if deleted, False if not found
        """
        if profile_id not in self._profiles:
            return False
        
        # Don't allow deletion of built-in profiles
        profile = self._profiles[profile_id]
        if profile.profile_type != InputProfile.CUSTOM:
            self.logger.warning("Attempted to delete built-in profile", extra={
                "profile_id": profile_id
            })
            return False
        
        try:
            # Remove file
            file_path = self._profiles_dir / f"{profile_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            # Remove from memory
            del self._profiles[profile_id]
            
            # Remove any joystick assignments
            assignments_to_remove = [guid for guid, assigned_profile in self._joystick_assignments.items() 
                                   if assigned_profile == profile_id]
            for guid in assignments_to_remove:
                del self._joystick_assignments[guid]
            
            self._save_joystick_assignments()
            
            self.logger.info("Profile deleted", extra={"profile_id": profile_id})
            return True
            
        except Exception as e:
            self.logger.error("Failed to delete profile", extra={
                "profile_id": profile_id,
                "error": str(e)
            })
            return False
    
    def assign_profile_to_joystick(self, joystick_guid: str, profile_id: str) -> bool:
        """
        Assign a profile to a specific joystick.
        
        Args:
            joystick_guid: GUID of the joystick
            profile_id: ID of the profile to assign
            
        Returns:
            True if assigned successfully, False otherwise
        """
        if profile_id not in self._profiles:
            self.logger.error("Profile not found for assignment", extra={
                "profile_id": profile_id,
                "joystick_guid": joystick_guid
            })
            return False
        
        self._joystick_assignments[joystick_guid] = profile_id
        self._save_joystick_assignments()
        
        self.logger.info("Profile assigned to joystick", extra={
            "joystick_guid": joystick_guid,
            "profile_id": profile_id
        })
        
        return True
    
    def get_profile_for_joystick(self, joystick_guid: str) -> Optional[JoystickProfile]:
        """
        Get the assigned profile for a joystick.
        
        Args:
            joystick_guid: GUID of the joystick
            
        Returns:
            JoystickProfile if assigned, None otherwise (falls back to default)
        """
        profile_id = self._joystick_assignments.get(joystick_guid, "default")
        return self._profiles.get(profile_id)
    
    def _save_joystick_assignments(self) -> None:
        """Save joystick assignments to disk."""
        try:
            with open(self._mappings_file, 'w', encoding='utf-8') as f:
                json.dump(self._joystick_assignments, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error("Failed to save joystick assignments", extra={
                "error": str(e)
            })
    
    def create_custom_profile(self, 
                            name: str,
                            description: str,
                            base_profile_id: str = "default",
                            joystick_guid: Optional[str] = None) -> str:
        """
        Create a new custom profile based on an existing profile.
        
        Args:
            name: Name for the new profile
            description: Description of the profile
            base_profile_id: ID of profile to base the new one on
            joystick_guid: Optional GUID of specific joystick this is for
            
        Returns:
            ID of the created profile
        """
        if base_profile_id not in self._profiles:
            raise ConfigurationError(f"Base profile '{base_profile_id}' not found")
        
        base_profile = self._profiles[base_profile_id]
        
        # Create new profile ID
        import uuid
        profile_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        # Create custom profile
        custom_profile = JoystickProfile(
            name=name,
            description=description,
            profile_type=InputProfile.CUSTOM,
            axis_mappings=base_profile.axis_mappings.copy(),
            button_mappings=base_profile.button_mappings.copy(),
            deadzone=base_profile.deadzone,
            sensitivity=base_profile.sensitivity,
            invert_y=base_profile.invert_y,
            created_for_guid=joystick_guid
        )
        
        # Save the profile
        self.save_profile(profile_id, custom_profile)
        
        # Auto-assign to joystick if provided
        if joystick_guid:
            self.assign_profile_to_joystick(joystick_guid, profile_id)
        
        self.logger.info("Custom profile created", extra={
            "profile_id": profile_id,
            "name": name,
            "base_profile": base_profile_id,
            "joystick_guid": joystick_guid
        })
        
        return profile_id
    
    def validate_profile(self, profile: JoystickProfile) -> List[str]:
        """
        Validate a profile configuration.
        
        Args:
            profile: Profile to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate deadzone
        if not 0.0 <= profile.deadzone <= 1.0:
            errors.append(f"Deadzone must be between 0.0 and 1.0, got {profile.deadzone}")
        
        # Validate sensitivity
        if not 0.1 <= profile.sensitivity <= 5.0:
            errors.append(f"Sensitivity must be between 0.1 and 5.0, got {profile.sensitivity}")
        
        # Validate axis mappings
        for axis_id, axis_config in profile.axis_mappings.items():
            try:
                # Check axis_type is valid
                AxisType(axis_config["axis_type"])
            except ValueError:
                errors.append(f"Invalid axis_type '{axis_config['axis_type']}' for axis {axis_id}")
            
            # Check scale is reasonable
            scale = axis_config.get("scale", 1.0)
            if not 0.1 <= scale <= 10.0:
                errors.append(f"Axis {axis_id} scale must be between 0.1 and 10.0, got {scale}")
        
        # Validate button mappings
        for button_id, button_config in profile.button_mappings.items():
            repeat_delay = button_config.get("repeat_delay", 0.5)
            repeat_rate = button_config.get("repeat_rate", 0.1)
            
            if repeat_delay < 0.1 or repeat_delay > 5.0:
                errors.append(f"Button {button_id} repeat_delay must be between 0.1 and 5.0")
            
            if repeat_rate < 0.05 or repeat_rate > 1.0:
                errors.append(f"Button {button_id} repeat_rate must be between 0.05 and 1.0")
        
        return errors
    
    def export_profile(self, profile_id: str, export_path: Path) -> bool:
        """
        Export a profile to a file.
        
        Args:
            profile_id: ID of profile to export
            export_path: Path to export to
            
        Returns:
            True if exported successfully, False otherwise
        """
        if profile_id not in self._profiles:
            return False
        
        try:
            profile = self._profiles[profile_id]
            profile_data = asdict(profile)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Profile exported", extra={
                "profile_id": profile_id,
                "export_path": str(export_path)
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to export profile", extra={
                "profile_id": profile_id,
                "error": str(e)
            })
            return False
    
    def import_profile(self, import_path: Path, profile_id: Optional[str] = None) -> Optional[str]:
        """
        Import a profile from a file.
        
        Args:
            import_path: Path to import from
            profile_id: Optional custom ID for the profile
            
        Returns:
            ID of imported profile, or None if failed
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create profile object
            profile = JoystickProfile(
                name=data["name"],
                description=data["description"],
                profile_type=InputProfile(data.get("profile_type", "custom")),
                axis_mappings=data["axis_mappings"],
                button_mappings=data["button_mappings"],
                deadzone=data.get("deadzone", 0.1),
                sensitivity=data.get("sensitivity", 1.0),
                invert_y=data.get("invert_y", False),
                version=data.get("version", "1.0"),
                created_for_guid=data.get("created_for_guid"),
                last_modified=data.get("last_modified")
            )
            
            # Validate profile
            errors = self.validate_profile(profile)
            if errors:
                self.logger.error("Profile validation failed", extra={
                    "errors": errors
                })
                return None
            
            # Generate ID if not provided
            if profile_id is None:
                import uuid
                profile_id = f"imported_{uuid.uuid4().hex[:8]}"
            
            # Save profile
            self.save_profile(profile_id, profile)
            
            self.logger.info("Profile imported", extra={
                "profile_id": profile_id,
                "import_path": str(import_path)
            })
            
            return profile_id
            
        except Exception as e:
            self.logger.error("Failed to import profile", extra={
                "import_path": str(import_path),
                "error": str(e)
            })
            return None


# Global input config manager instance
_input_config_manager: Optional[InputConfigManager] = None


def get_input_config_manager() -> InputConfigManager:
    """Get the global input configuration manager instance."""
    global _input_config_manager
    if _input_config_manager is None:
        _input_config_manager = InputConfigManager()
    return _input_config_manager


def reset_input_config_manager() -> None:
    """Reset the global input configuration manager instance."""
    global _input_config_manager
    _input_config_manager = None