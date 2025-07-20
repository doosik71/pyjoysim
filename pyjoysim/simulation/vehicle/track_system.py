"""
Track system for car simulation.

This module provides track loading, rendering, and collision detection
for racing circuits and driving scenarios.
"""

import json
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ...physics import Vector2D, PhysicsObject, create_static_wall, StandardMaterials
from ...rendering import StandardColors, Color
from ...core.logging import get_logger


@dataclass
class Checkpoint:
    """A checkpoint on the track."""
    position: Vector2D
    width: float
    angle: float  # radians
    checkpoint_id: int
    is_finish_line: bool = False
    
    def is_crossed(self, prev_pos: Vector2D, current_pos: Vector2D) -> bool:
        """Check if the checkpoint was crossed by movement from prev_pos to current_pos."""
        # Create checkpoint line
        half_width = self.width / 2
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        
        line_start = Vector2D(
            self.position.x - half_width * sin_angle,
            self.position.y + half_width * cos_angle
        )
        line_end = Vector2D(
            self.position.x + half_width * sin_angle,
            self.position.y - half_width * cos_angle
        )
        
        # Check if the movement line intersects the checkpoint line
        return self._lines_intersect(prev_pos, current_pos, line_start, line_end)
    
    def _lines_intersect(self, p1: Vector2D, p2: Vector2D, p3: Vector2D, p4: Vector2D) -> bool:
        """Check if line p1-p2 intersects line p3-p4."""
        denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
        if abs(denom) < 1e-10:
            return False  # Lines are parallel
        
        t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom
        u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1


@dataclass
class TrackWall:
    """A wall segment on the track."""
    start: Vector2D
    end: Vector2D
    thickness: float = 1.0
    material: str = "concrete"
    wall_type: str = "barrier"  # barrier, tire_wall, fence


@dataclass 
class TrackData:
    """Complete track data."""
    name: str
    description: str
    author: str = "Unknown"
    version: str = "1.0"
    
    # Track geometry
    spawn_position: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    spawn_heading: float = 0.0  # radians
    
    # Track elements
    walls: List[TrackWall] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    
    # Track properties
    surface_material: str = "asphalt"
    weather: str = "clear"  # clear, rain, snow
    time_of_day: str = "day"  # day, night, dawn, dusk
    
    # Bounds for camera and physics
    bounds_min: Vector2D = field(default_factory=lambda: Vector2D(-50, -50))
    bounds_max: Vector2D = field(default_factory=lambda: Vector2D(50, 50))
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrackData':
        """Create TrackData from dictionary."""
        track = cls(
            name=data.get("name", "Unnamed Track"),
            description=data.get("description", ""),
            author=data.get("author", "Unknown"),
            version=data.get("version", "1.0"),
            surface_material=data.get("surface_material", "asphalt"),
            weather=data.get("weather", "clear"),
            time_of_day=data.get("time_of_day", "day")
        )
        
        # Parse spawn
        if "spawn" in data:
            spawn = data["spawn"]
            track.spawn_position = Vector2D(spawn.get("x", 0), spawn.get("y", 0))
            track.spawn_heading = spawn.get("heading", 0.0)
        
        # Parse walls
        for wall_data in data.get("walls", []):
            wall = TrackWall(
                start=Vector2D(wall_data["start"]["x"], wall_data["start"]["y"]),
                end=Vector2D(wall_data["end"]["x"], wall_data["end"]["y"]),
                thickness=wall_data.get("thickness", 1.0),
                material=wall_data.get("material", "concrete"),
                wall_type=wall_data.get("type", "barrier")
            )
            track.walls.append(wall)
        
        # Parse checkpoints
        for cp_data in data.get("checkpoints", []):
            checkpoint = Checkpoint(
                position=Vector2D(cp_data["position"]["x"], cp_data["position"]["y"]),
                width=cp_data.get("width", 10.0),
                angle=cp_data.get("angle", 0.0),
                checkpoint_id=cp_data.get("id", 0),
                is_finish_line=cp_data.get("is_finish_line", False)
            )
            track.checkpoints.append(checkpoint)
        
        # Parse bounds
        if "bounds" in data:
            bounds = data["bounds"]
            track.bounds_min = Vector2D(bounds["min"]["x"], bounds["min"]["y"])
            track.bounds_max = Vector2D(bounds["max"]["x"], bounds["max"]["y"])
        
        return track
    
    def to_dict(self) -> Dict:
        """Convert TrackData to dictionary for saving."""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "surface_material": self.surface_material,
            "weather": self.weather,
            "time_of_day": self.time_of_day,
            "spawn": {
                "x": self.spawn_position.x,
                "y": self.spawn_position.y,
                "heading": self.spawn_heading
            },
            "walls": [
                {
                    "start": {"x": wall.start.x, "y": wall.start.y},
                    "end": {"x": wall.end.x, "y": wall.end.y},
                    "thickness": wall.thickness,
                    "material": wall.material,
                    "type": wall.wall_type
                }
                for wall in self.walls
            ],
            "checkpoints": [
                {
                    "position": {"x": cp.position.x, "y": cp.position.y},
                    "width": cp.width,
                    "angle": cp.angle,
                    "id": cp.checkpoint_id,
                    "is_finish_line": cp.is_finish_line
                }
                for cp in self.checkpoints
            ],
            "bounds": {
                "min": {"x": self.bounds_min.x, "y": self.bounds_min.y},
                "max": {"x": self.bounds_max.x, "y": self.bounds_max.y}
            }
        }


class TrackLoader:
    """Loads and manages track data."""
    
    def __init__(self, tracks_directory: str = "assets/tracks"):
        """Initialize track loader."""
        self.tracks_directory = Path(tracks_directory)
        self.logger = get_logger("track_loader")
        self._track_cache: Dict[str, TrackData] = {}
    
    def load_track(self, track_name: str) -> Optional[TrackData]:
        """Load a track by name."""
        if track_name in self._track_cache:
            return self._track_cache[track_name]
        
        track_file = self.tracks_directory / f"{track_name}.json"
        
        if not track_file.exists():
            self.logger.error("Track file not found", extra={"track": track_name, "file": str(track_file)})
            return None
        
        try:
            with open(track_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            track = TrackData.from_dict(data)
            self._track_cache[track_name] = track
            
            self.logger.info("Track loaded", extra={
                "track": track_name,
                "walls": len(track.walls),
                "checkpoints": len(track.checkpoints)
            })
            
            return track
            
        except Exception as e:
            self.logger.error("Failed to load track", extra={
                "track": track_name,
                "error": str(e)
            })
            return None
    
    def save_track(self, track: TrackData, filename: Optional[str] = None) -> bool:
        """Save a track to file."""
        if filename is None:
            filename = track.name.lower().replace(" ", "_")
        
        track_file = self.tracks_directory / f"{filename}.json"
        
        try:
            # Ensure directory exists
            track_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(track_file, 'w', encoding='utf-8') as f:
                json.dump(track.to_dict(), f, indent=2)
            
            self.logger.info("Track saved", extra={"track": track.name, "file": str(track_file)})
            return True
            
        except Exception as e:
            self.logger.error("Failed to save track", extra={
                "track": track.name,
                "error": str(e)
            })
            return False
    
    def list_tracks(self) -> List[str]:
        """List available track names."""
        if not self.tracks_directory.exists():
            return []
        
        tracks = []
        for track_file in self.tracks_directory.glob("*.json"):
            tracks.append(track_file.stem)
        
        return sorted(tracks)
    
    def create_basic_tracks(self) -> None:
        """Create basic default tracks."""
        self.logger.info("Creating basic tracks")
        
        # Oval track
        oval_track = self._create_oval_track()
        self.save_track(oval_track, "oval")
        
        # Figure-8 track
        figure8_track = self._create_figure8_track()
        self.save_track(figure8_track, "figure8")
        
        # City circuit
        city_track = self._create_city_circuit()
        self.save_track(city_track, "city_circuit")
    
    def _create_oval_track(self) -> TrackData:
        """Create a simple oval track."""
        track = TrackData(
            name="Oval Speedway",
            description="A simple oval track for high-speed racing",
            author="PyJoySim Team"
        )
        
        # Track dimensions
        length = 80
        width = 40
        wall_thickness = 2
        
        # Outer walls
        track.walls.extend([
            # Top straight
            TrackWall(Vector2D(-length/2, width/2), Vector2D(length/2, width/2), wall_thickness),
            # Bottom straight  
            TrackWall(Vector2D(-length/2, -width/2), Vector2D(length/2, -width/2), wall_thickness),
            # Left curve (simplified as straight segments)
            TrackWall(Vector2D(-length/2, -width/2), Vector2D(-length/2, width/2), wall_thickness),
            # Right curve
            TrackWall(Vector2D(length/2, -width/2), Vector2D(length/2, width/2), wall_thickness)
        ])
        
        # Inner walls (smaller oval)
        inner_length = length - 20
        inner_width = width - 20
        track.walls.extend([
            TrackWall(Vector2D(-inner_length/2, inner_width/2), Vector2D(inner_length/2, inner_width/2), wall_thickness),
            TrackWall(Vector2D(-inner_length/2, -inner_width/2), Vector2D(inner_length/2, -inner_width/2), wall_thickness),
            TrackWall(Vector2D(-inner_length/2, -inner_width/2), Vector2D(-inner_length/2, inner_width/2), wall_thickness),
            TrackWall(Vector2D(inner_length/2, -inner_width/2), Vector2D(inner_length/2, inner_width/2), wall_thickness)
        ])
        
        # Checkpoints
        track.checkpoints = [
            Checkpoint(Vector2D(-30, 0), 20, 0, 0, False),  # First quarter
            Checkpoint(Vector2D(0, 15), 20, math.pi/2, 1, False),  # Second quarter  
            Checkpoint(Vector2D(30, 0), 20, math.pi, 2, False),  # Third quarter
            Checkpoint(Vector2D(0, -15), 20, -math.pi/2, 3, True)  # Finish line
        ]
        
        # Spawn position
        track.spawn_position = Vector2D(-35, 0)
        track.spawn_heading = 0.0
        
        # Bounds
        track.bounds_min = Vector2D(-length/2 - 10, -width/2 - 10)
        track.bounds_max = Vector2D(length/2 + 10, width/2 + 10)
        
        return track
    
    def _create_figure8_track(self) -> TrackData:
        """Create a figure-8 track."""
        track = TrackData(
            name="Figure-8 Circuit",
            description="A challenging figure-8 track with elevation changes",
            author="PyJoySim Team"
        )
        
        # Create figure-8 walls (simplified)
        radius = 25
        center_offset = 15
        
        # Top loop
        for i in range(8):
            angle1 = i * math.pi / 4
            angle2 = (i + 1) * math.pi / 4
            
            start = Vector2D(
                center_offset + radius * math.cos(angle1),
                radius * math.sin(angle1)
            )
            end = Vector2D(
                center_offset + radius * math.cos(angle2),
                radius * math.sin(angle2)
            )
            track.walls.append(TrackWall(start, end, 2.0))
        
        # Bottom loop
        for i in range(8):
            angle1 = i * math.pi / 4
            angle2 = (i + 1) * math.pi / 4
            
            start = Vector2D(
                -center_offset + radius * math.cos(angle1),
                -radius * math.sin(angle1)
            )
            end = Vector2D(
                -center_offset + radius * math.cos(angle2),
                -radius * math.sin(angle2)
            )
            track.walls.append(TrackWall(start, end, 2.0))
        
        # Checkpoints
        track.checkpoints = [
            Checkpoint(Vector2D(center_offset + radius, 0), 12, math.pi/2, 0, False),
            Checkpoint(Vector2D(0, 0), 12, 0, 1, False),  # Crossover point
            Checkpoint(Vector2D(-center_offset - radius, 0), 12, -math.pi/2, 2, False),
            Checkpoint(Vector2D(0, 0), 12, math.pi, 3, True)  # Finish at crossover
        ]
        
        track.spawn_position = Vector2D(center_offset + radius - 5, 0)
        track.bounds_min = Vector2D(-50, -30)
        track.bounds_max = Vector2D(50, 30)
        
        return track
    
    def _create_city_circuit(self) -> TrackData:
        """Create a city street circuit."""
        track = TrackData(
            name="City Circuit",
            description="A technical street circuit with tight corners",
            author="PyJoySim Team"
        )
        
        # Street layout (rectangular with chicane)
        street_blocks = [
            # Main straight
            TrackWall(Vector2D(-40, 20), Vector2D(40, 20), 2.0),
            TrackWall(Vector2D(-40, -20), Vector2D(40, -20), 2.0),
            
            # End curves
            TrackWall(Vector2D(40, -20), Vector2D(40, 20), 2.0),
            TrackWall(Vector2D(-40, -20), Vector2D(-40, 20), 2.0),
            
            # Chicane in the middle
            TrackWall(Vector2D(-5, 20), Vector2D(-5, 10), 2.0),
            TrackWall(Vector2D(-5, 10), Vector2D(5, 10), 2.0),
            TrackWall(Vector2D(5, 10), Vector2D(5, -10), 2.0),
            TrackWall(Vector2D(5, -10), Vector2D(-5, -10), 2.0),
            TrackWall(Vector2D(-5, -10), Vector2D(-5, -20), 2.0)
        ]
        
        track.walls.extend(street_blocks)
        
        # Checkpoints
        track.checkpoints = [
            Checkpoint(Vector2D(-20, 0), 40, math.pi/2, 0, False),
            Checkpoint(Vector2D(0, 15), 10, 0, 1, False),  # Chicane entry
            Checkpoint(Vector2D(0, -15), 10, math.pi, 2, False),  # Chicane exit
            Checkpoint(Vector2D(20, 0), 40, -math.pi/2, 3, True)  # Finish
        ]
        
        track.spawn_position = Vector2D(-35, 0)
        track.bounds_min = Vector2D(-45, -25)
        track.bounds_max = Vector2D(45, 25)
        
        return track


class TrackInstance:
    """A loaded track instance with physics objects."""
    
    def __init__(self, track_data: TrackData, physics_engine):
        """Initialize track instance."""
        self.track_data = track_data
        self.physics_engine = physics_engine
        self.logger = get_logger(f"track.{track_data.name}")
        
        # Physics objects
        self.wall_objects: List[PhysicsObject] = []
        
        # Lap timing
        self.last_checkpoint_times: Dict[int, float] = {}
        self.last_checkpoint_id = -1
        self.lap_count = 0
        self.current_lap_start_time = 0.0
        self.best_lap_time = float('inf')
        
        # Create physics objects
        self._create_physics_objects()
        
        self.logger.info("Track instance created", extra={
            "track": track_data.name,
            "walls": len(self.wall_objects),
            "checkpoints": len(track_data.checkpoints)
        })
    
    def _create_physics_objects(self) -> None:
        """Create physics objects for walls."""
        for wall in self.track_data.walls:
            # Calculate wall center and dimensions
            center = Vector2D(
                (wall.start.x + wall.end.x) / 2,
                (wall.start.y + wall.end.y) / 2
            )
            
            length = wall.start.distance_to(wall.end)
            width = wall.thickness
            
            # Calculate angle
            dx = wall.end.x - wall.start.x
            dy = wall.end.y - wall.start.y
            angle = math.atan2(dy, dx)
            
            # Create wall object
            wall_obj, _ = create_static_wall(
                self.physics_engine,
                center,
                width=length,
                height=width,
                material=StandardMaterials.CONCRETE,
                name=f"Wall_{len(self.wall_objects)}"
            )
            
            # Set rotation
            wall_obj.set_angle(angle)
            
            self.wall_objects.append(wall_obj)
    
    def check_checkpoint_crossing(self, prev_pos: Vector2D, current_pos: Vector2D, current_time: float) -> Optional[int]:
        """
        Check if any checkpoint was crossed.
        
        Returns:
            Checkpoint ID if crossed, None otherwise
        """
        for checkpoint in self.track_data.checkpoints:
            if checkpoint.is_crossed(prev_pos, current_pos):
                # Update timing
                self.last_checkpoint_times[checkpoint.checkpoint_id] = current_time
                
                # Check for lap completion
                if checkpoint.is_finish_line and self.last_checkpoint_id != checkpoint.checkpoint_id:
                    if self.current_lap_start_time > 0:
                        lap_time = current_time - self.current_lap_start_time
                        self.best_lap_time = min(self.best_lap_time, lap_time)
                        self.lap_count += 1
                        
                        self.logger.info("Lap completed", extra={
                            "lap": self.lap_count,
                            "time": lap_time,
                            "best": self.best_lap_time
                        })
                    
                    self.current_lap_start_time = current_time
                
                self.last_checkpoint_id = checkpoint.checkpoint_id
                return checkpoint.checkpoint_id
        
        return None
    
    def get_spawn_position(self) -> Vector2D:
        """Get track spawn position."""
        return self.track_data.spawn_position
    
    def get_spawn_heading(self) -> float:
        """Get track spawn heading."""
        return self.track_data.spawn_heading
    
    def render(self, renderer, camera) -> None:
        """Render track elements."""
        # Render walls
        for wall_obj in self.wall_objects:
            if not wall_obj.is_active:
                continue
            
            screen_pos = camera.world_to_screen(wall_obj.position) if camera else wall_obj.position
            
            # Get wall color based on material
            wall_color = self._get_wall_color(wall_obj)
            
            # Simple rectangle rendering (could be improved with proper rotation)
            scale = camera.zoom if camera else 20
            renderer.draw_rectangle(
                screen_pos,
                5 * scale,  # Simplified size
                2 * scale,
                wall_color,
                fill=True
            )
        
        # Render checkpoints
        for checkpoint in self.track_data.checkpoints:
            if camera:
                screen_pos = camera.world_to_screen(checkpoint.position)
                scale = camera.zoom
            else:
                screen_pos = checkpoint.position
                scale = 20
            
            # Checkpoint line
            color = StandardColors.RED if checkpoint.is_finish_line else StandardColors.YELLOW
            
            half_width = checkpoint.width / 2
            cos_angle = math.cos(checkpoint.angle)
            sin_angle = math.sin(checkpoint.angle)
            
            start_offset = Vector2D(-half_width * sin_angle, half_width * cos_angle)
            end_offset = Vector2D(half_width * sin_angle, -half_width * cos_angle)
            
            if camera:
                start_screen = camera.world_to_screen(checkpoint.position + start_offset)
                end_screen = camera.world_to_screen(checkpoint.position + end_offset)
            else:
                start_screen = checkpoint.position + start_offset
                end_screen = checkpoint.position + end_offset
            
            renderer.draw_line(start_screen, end_screen, color, width=3)
    
    def _get_wall_color(self, wall_obj: PhysicsObject) -> Color:
        """Get color for wall based on material."""
        # This is a simplified version - in reality we'd store material info with the wall
        return StandardColors.DARK_GRAY
    
    def cleanup(self) -> None:
        """Clean up track physics objects."""
        for wall_obj in self.wall_objects:
            if wall_obj.is_active:
                wall_obj.destroy()
        
        self.wall_objects.clear()
        self.logger.info("Track cleaned up", extra={"track": self.track_data.name})