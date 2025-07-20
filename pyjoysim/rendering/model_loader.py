"""
3D Model loading system for PyJoySim.

This module provides functionality for loading 3D models from various formats
including OBJ files and creating primitive geometries.
"""

import os
import math
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from .renderer3d import Mesh3D, Material3D, PrimitiveType
from ..core.logging import get_logger


@dataclass
class ModelInfo:
    """Information about a loaded 3D model."""
    name: str
    file_path: Optional[str] = None
    vertex_count: int = 0
    triangle_count: int = 0
    bounds_min: Tuple[float, float, float] = (0, 0, 0)
    bounds_max: Tuple[float, float, float] = (0, 0, 0)
    has_normals: bool = False
    has_texcoords: bool = False
    has_colors: bool = False
    materials: List[str] = None
    
    def __post_init__(self):
        if self.materials is None:
            self.materials = []


class ModelLoader:
    """
    3D model loader with support for multiple formats and primitive generation.
    
    Supports:
    - OBJ file loading
    - Primitive generation (cube, sphere, cylinder, etc.)
    - Model optimization and validation
    - Material extraction
    """
    
    def __init__(self):
        """Initialize model loader."""
        self.logger = get_logger("model_loader")
        
        # Cache for loaded models
        self.model_cache: Dict[str, Mesh3D] = {}
        
        # Supported file formats
        self.supported_formats = ['.obj']
        if TRIMESH_AVAILABLE:
            self.supported_formats.extend(['.ply', '.stl', '.off', '.3mf'])
        
        self.logger.info("ModelLoader initialized", extra={
            "supported_formats": self.supported_formats,
            "trimesh_available": TRIMESH_AVAILABLE
        })
    
    def load_model(self, file_path: str, name: Optional[str] = None) -> Optional[Mesh3D]:
        """
        Load a 3D model from file.
        
        Args:
            file_path: Path to model file
            name: Optional name for the model
            
        Returns:
            Mesh3D object or None if loading failed
        """
        if not os.path.exists(file_path):
            self.logger.error("Model file not found", extra={"path": file_path})
            return None
        
        # Check if already cached
        cache_key = f"{file_path}:{name or 'default'}"
        if cache_key in self.model_cache:
            self.logger.debug("Model loaded from cache", extra={"path": file_path})
            return self.model_cache[cache_key]
        
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        if ext not in self.supported_formats:
            self.logger.error("Unsupported file format", extra={
                "path": file_path,
                "extension": ext,
                "supported": self.supported_formats
            })
            return None
        
        # Load based on format
        mesh = None
        if ext == '.obj':
            mesh = self._load_obj(file_path, name)
        elif TRIMESH_AVAILABLE:
            mesh = self._load_with_trimesh(file_path, name)
        
        if mesh:
            # Cache the loaded model
            self.model_cache[cache_key] = mesh
            
            self.logger.info("Model loaded successfully", extra={
                "path": file_path,
                "name": mesh.name,
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.indices) // 3 if mesh.indices is not None else 0
            })
        
        return mesh
    
    def _load_obj(self, file_path: str, name: Optional[str] = None) -> Optional[Mesh3D]:
        """Load OBJ file using manual parsing."""
        try:
            vertices = []
            normals = []
            texcoords = []
            faces = []
            
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':  # Vertex
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
                    elif parts[0] == 'vn':  # Normal
                        normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
                    elif parts[0] == 'vt':  # Texture coordinate
                        texcoords.append([float(parts[1]), float(parts[2])])
                    
                    elif parts[0] == 'f':  # Face
                        face_vertices = []
                        face_normals = []
                        face_texcoords = []
                        
                        for i in range(1, len(parts)):
                            vertex_data = parts[i].split('/')
                            
                            # Vertex index (required)
                            vertex_idx = int(vertex_data[0]) - 1  # OBJ indices are 1-based
                            face_vertices.append(vertex_idx)
                            
                            # Texture coordinate index (optional)
                            if len(vertex_data) > 1 and vertex_data[1]:
                                texcoord_idx = int(vertex_data[1]) - 1
                                face_texcoords.append(texcoord_idx)
                            
                            # Normal index (optional)
                            if len(vertex_data) > 2 and vertex_data[2]:
                                normal_idx = int(vertex_data[2]) - 1
                                face_normals.append(normal_idx)
                        
                        # Triangulate face if it has more than 3 vertices
                        if len(face_vertices) >= 3:
                            for i in range(1, len(face_vertices) - 1):
                                faces.append([
                                    face_vertices[0],
                                    face_vertices[i],
                                    face_vertices[i + 1]
                                ])
            
            if not vertices:
                self.logger.error("No vertices found in OBJ file", extra={"path": file_path})
                return None
            
            # Convert to numpy arrays
            vertices_array = np.array(vertices, dtype=np.float32)
            
            indices_array = None
            if faces:
                indices_list = []
                for face in faces:
                    indices_list.extend(face)
                indices_array = np.array(indices_list, dtype=np.uint32)
            
            normals_array = None
            if normals:
                normals_array = np.array(normals, dtype=np.float32)
            
            texcoords_array = None
            if texcoords:
                texcoords_array = np.array(texcoords, dtype=np.float32)
            
            model_name = name or os.path.splitext(os.path.basename(file_path))[0]
            
            return Mesh3D(
                name=model_name,
                vertices=vertices_array,
                indices=indices_array,
                normals=normals_array,
                texcoords=texcoords_array
            )
        
        except Exception as e:
            self.logger.error("Failed to load OBJ file", extra={
                "path": file_path,
                "error": str(e)
            })
            return None
    
    def _load_with_trimesh(self, file_path: str, name: Optional[str] = None) -> Optional[Mesh3D]:
        """Load model using trimesh library."""
        try:
            # Load with trimesh
            mesh = trimesh.load(file_path)
            
            # Handle case where multiple meshes are loaded
            if isinstance(mesh, trimesh.Scene):
                # Combine all meshes in the scene
                combined = trimesh.util.concatenate([
                    geom for geom in mesh.geometry.values()
                    if isinstance(geom, trimesh.Trimesh)
                ])
                mesh = combined
            
            if not isinstance(mesh, trimesh.Trimesh):
                self.logger.error("Loaded object is not a valid mesh", extra={"path": file_path})
                return None
            
            # Extract data
            vertices = mesh.vertices.astype(np.float32)
            indices = mesh.faces.flatten().astype(np.uint32)
            
            # Get normals
            normals = None
            if mesh.vertex_normals is not None:
                normals = mesh.vertex_normals.astype(np.float32)
            
            # Get texture coordinates (if available)
            texcoords = None
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                texcoords = mesh.visual.uv.astype(np.float32)
            
            model_name = name or os.path.splitext(os.path.basename(file_path))[0]
            
            return Mesh3D(
                name=model_name,
                vertices=vertices,
                indices=indices,
                normals=normals,
                texcoords=texcoords
            )
        
        except Exception as e:
            self.logger.error("Failed to load model with trimesh", extra={
                "path": file_path,
                "error": str(e)
            })
            return None
    
    def create_primitive(self, primitive_type: PrimitiveType, **kwargs) -> Mesh3D:
        """
        Create a primitive mesh.
        
        Args:
            primitive_type: Type of primitive to create
            **kwargs: Parameters specific to the primitive type
            
        Returns:
            Mesh3D object
        """
        if primitive_type == PrimitiveType.CUBE:
            return self.create_cube(**kwargs)
        elif primitive_type == PrimitiveType.SPHERE:
            return self.create_sphere(**kwargs)
        elif primitive_type == PrimitiveType.CYLINDER:
            return self.create_cylinder(**kwargs)
        elif primitive_type == PrimitiveType.CONE:
            return self.create_cone(**kwargs)
        elif primitive_type == PrimitiveType.PLANE:
            return self.create_plane(**kwargs)
        else:
            raise ValueError(f"Unsupported primitive type: {primitive_type}")
    
    def create_cube(self, 
                   size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                   name: str = "cube") -> Mesh3D:
        """
        Create a cube mesh.
        
        Args:
            size: Size of the cube (width, height, depth)
            name: Name for the mesh
            
        Returns:
            Mesh3D object
        """
        width, height, depth = size
        w, h, d = width * 0.5, height * 0.5, depth * 0.5
        
        # 24 vertices (4 per face for proper normals and UVs)
        vertices = np.array([
            # Front face (z = d)
            [-w, -h,  d], [ w, -h,  d], [ w,  h,  d], [-w,  h,  d],
            # Back face (z = -d)
            [ w, -h, -d], [-w, -h, -d], [-w,  h, -d], [ w,  h, -d],
            # Left face (x = -w)
            [-w, -h, -d], [-w, -h,  d], [-w,  h,  d], [-w,  h, -d],
            # Right face (x = w)
            [ w, -h,  d], [ w, -h, -d], [ w,  h, -d], [ w,  h,  d],
            # Top face (y = h)
            [-w,  h,  d], [ w,  h,  d], [ w,  h, -d], [-w,  h, -d],
            # Bottom face (y = -h)
            [-w, -h, -d], [ w, -h, -d], [ w, -h,  d], [-w, -h,  d],
        ], dtype=np.float32)
        
        # 36 indices (12 triangles)
        indices = np.array([
            # Front
            0, 1, 2,  2, 3, 0,
            # Back
            4, 5, 6,  6, 7, 4,
            # Left
            8, 9, 10,  10, 11, 8,
            # Right
            12, 13, 14,  14, 15, 12,
            # Top
            16, 17, 18,  18, 19, 16,
            # Bottom
            20, 21, 22,  22, 23, 20,
        ], dtype=np.uint32)
        
        # Normals for each face
        normals = np.array([
            # Front (z+)
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            # Back (z-)
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
            # Left (x-)
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
            # Right (x+)
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            # Top (y+)
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            # Bottom (y-)
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
        ], dtype=np.float32)
        
        # Texture coordinates
        texcoords = np.array([
            # Front
            [0, 0], [1, 0], [1, 1], [0, 1],
            # Back
            [0, 0], [1, 0], [1, 1], [0, 1],
            # Left
            [0, 0], [1, 0], [1, 1], [0, 1],
            # Right
            [0, 0], [1, 0], [1, 1], [0, 1],
            # Top
            [0, 0], [1, 0], [1, 1], [0, 1],
            # Bottom
            [0, 0], [1, 0], [1, 1], [0, 1],
        ], dtype=np.float32)
        
        return Mesh3D(name, vertices, indices, normals, texcoords)
    
    def create_sphere(self, 
                     radius: float = 1.0,
                     segments: int = 32,
                     rings: int = 16,
                     name: str = "sphere") -> Mesh3D:
        """
        Create a sphere mesh.
        
        Args:
            radius: Radius of the sphere
            segments: Number of horizontal segments
            rings: Number of vertical rings
            name: Name for the mesh
            
        Returns:
            Mesh3D object
        """
        vertices = []
        normals = []
        texcoords = []
        indices = []
        
        # Generate vertices
        for ring in range(rings + 1):
            theta = math.pi * ring / rings  # 0 to pi
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            for segment in range(segments + 1):
                phi = 2 * math.pi * segment / segments  # 0 to 2*pi
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                
                # Position
                x = radius * sin_theta * cos_phi
                y = radius * cos_theta
                z = radius * sin_theta * sin_phi
                vertices.append([x, y, z])
                
                # Normal (same as normalized position for a sphere)
                normals.append([sin_theta * cos_phi, cos_theta, sin_theta * sin_phi])
                
                # Texture coordinates
                u = segment / segments
                v = ring / rings
                texcoords.append([u, v])
        
        # Generate indices
        for ring in range(rings):
            for segment in range(segments):
                # Current ring
                current = ring * (segments + 1) + segment
                next_segment = ring * (segments + 1) + (segment + 1)
                
                # Next ring
                next_ring = (ring + 1) * (segments + 1) + segment
                next_ring_next = (ring + 1) * (segments + 1) + (segment + 1)
                
                # Two triangles per quad
                indices.extend([current, next_ring, next_segment])
                indices.extend([next_segment, next_ring, next_ring_next])
        
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        return Mesh3D(name, vertices, indices, normals, texcoords)
    
    def create_cylinder(self,
                       radius: float = 1.0,
                       height: float = 2.0,
                       segments: int = 32,
                       name: str = "cylinder") -> Mesh3D:
        """
        Create a cylinder mesh.
        
        Args:
            radius: Radius of the cylinder
            height: Height of the cylinder
            segments: Number of circular segments
            name: Name for the mesh
            
        Returns:
            Mesh3D object
        """
        vertices = []
        normals = []
        texcoords = []
        indices = []
        
        half_height = height * 0.5
        
        # Generate side vertices
        for level in range(2):  # Top and bottom
            y = half_height if level == 1 else -half_height
            
            for segment in range(segments + 1):
                angle = 2 * math.pi * segment / segments
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                
                vertices.append([x, y, z])
                normals.append([math.cos(angle), 0, math.sin(angle)])  # Side normal
                texcoords.append([segment / segments, level])
        
        # Generate side indices
        for segment in range(segments):
            # Bottom vertices
            bottom_current = segment
            bottom_next = segment + 1
            
            # Top vertices
            top_current = (segments + 1) + segment
            top_next = (segments + 1) + segment + 1
            
            # Two triangles per quad
            indices.extend([bottom_current, top_current, bottom_next])
            indices.extend([bottom_next, top_current, top_next])
        
        # Add caps
        vertex_offset = len(vertices)
        
        # Bottom cap center
        vertices.append([0, -half_height, 0])
        normals.append([0, -1, 0])
        texcoords.append([0.5, 0.5])
        bottom_center = vertex_offset
        
        # Top cap center
        vertices.append([0, half_height, 0])
        normals.append([0, 1, 0])
        texcoords.append([0.5, 0.5])
        top_center = vertex_offset + 1
        
        # Bottom cap vertices
        for segment in range(segments + 1):
            angle = 2 * math.pi * segment / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            vertices.append([x, -half_height, z])
            normals.append([0, -1, 0])
            texcoords.append([0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)])
        
        # Top cap vertices
        for segment in range(segments + 1):
            angle = 2 * math.pi * segment / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            vertices.append([x, half_height, z])
            normals.append([0, 1, 0])
            texcoords.append([0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)])
        
        # Bottom cap indices
        bottom_cap_start = vertex_offset + 2
        for segment in range(segments):
            indices.extend([
                bottom_center,
                bottom_cap_start + segment + 1,
                bottom_cap_start + segment
            ])
        
        # Top cap indices
        top_cap_start = bottom_cap_start + segments + 1
        for segment in range(segments):
            indices.extend([
                top_center,
                top_cap_start + segment,
                top_cap_start + segment + 1
            ])
        
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        return Mesh3D(name, vertices, indices, normals, texcoords)
    
    def create_cone(self,
                   radius: float = 1.0,
                   height: float = 2.0,
                   segments: int = 32,
                   name: str = "cone") -> Mesh3D:
        """
        Create a cone mesh.
        
        Args:
            radius: Radius of the base
            height: Height of the cone
            segments: Number of circular segments
            name: Name for the mesh
            
        Returns:
            Mesh3D object
        """
        vertices = []
        normals = []
        texcoords = []
        indices = []
        
        half_height = height * 0.5
        
        # Apex vertex
        vertices.append([0, half_height, 0])
        normals.append([0, 1, 0])  # Will be recalculated
        texcoords.append([0.5, 1.0])
        apex_index = 0
        
        # Base vertices
        for segment in range(segments + 1):
            angle = 2 * math.pi * segment / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            vertices.append([x, -half_height, z])
            
            # Calculate side normal
            # Normal points outward from the cone surface
            side_normal = np.array([math.cos(angle), radius / height, math.sin(angle)])
            side_normal = side_normal / np.linalg.norm(side_normal)
            normals.append(side_normal.tolist())
            
            texcoords.append([segment / segments, 0])
        
        # Side indices
        for segment in range(segments):
            base_current = 1 + segment
            base_next = 1 + segment + 1
            
            indices.extend([apex_index, base_current, base_next])
        
        # Base cap
        vertex_offset = len(vertices)
        
        # Base center
        vertices.append([0, -half_height, 0])
        normals.append([0, -1, 0])
        texcoords.append([0.5, 0.5])
        base_center = vertex_offset
        
        # Base cap vertices (separate for proper normals)
        for segment in range(segments + 1):
            angle = 2 * math.pi * segment / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            vertices.append([x, -half_height, z])
            normals.append([0, -1, 0])
            texcoords.append([0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)])
        
        # Base cap indices
        base_cap_start = vertex_offset + 1
        for segment in range(segments):
            indices.extend([
                base_center,
                base_cap_start + segment + 1,
                base_cap_start + segment
            ])
        
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        return Mesh3D(name, vertices, indices, normals, texcoords)
    
    def create_plane(self,
                    size: Tuple[float, float] = (1.0, 1.0),
                    subdivisions: Tuple[int, int] = (1, 1),
                    name: str = "plane") -> Mesh3D:
        """
        Create a plane mesh.
        
        Args:
            size: Size of the plane (width, depth)
            subdivisions: Number of subdivisions (width, depth)
            name: Name for the mesh
            
        Returns:
            Mesh3D object
        """
        width, depth = size
        width_subdivs, depth_subdivs = subdivisions
        
        vertices = []
        normals = []
        texcoords = []
        indices = []
        
        # Generate vertices
        for z in range(depth_subdivs + 1):
            for x in range(width_subdivs + 1):
                # Position
                pos_x = (x / width_subdivs - 0.5) * width
                pos_z = (z / depth_subdivs - 0.5) * depth
                vertices.append([pos_x, 0, pos_z])
                
                # Normal (up)
                normals.append([0, 1, 0])
                
                # Texture coordinates
                u = x / width_subdivs
                v = z / depth_subdivs
                texcoords.append([u, v])
        
        # Generate indices
        for z in range(depth_subdivs):
            for x in range(width_subdivs):
                # Current quad vertices
                bottom_left = z * (width_subdivs + 1) + x
                bottom_right = bottom_left + 1
                top_left = (z + 1) * (width_subdivs + 1) + x
                top_right = top_left + 1
                
                # Two triangles per quad
                indices.extend([bottom_left, top_left, bottom_right])
                indices.extend([bottom_right, top_left, top_right])
        
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        return Mesh3D(name, vertices, indices, normals, texcoords)
    
    def get_model_info(self, mesh: Mesh3D) -> ModelInfo:
        """
        Get information about a loaded model.
        
        Args:
            mesh: Mesh to analyze
            
        Returns:
            ModelInfo object
        """
        # Calculate bounds
        if len(mesh.vertices) > 0:
            bounds_min = tuple(np.min(mesh.vertices, axis=0))
            bounds_max = tuple(np.max(mesh.vertices, axis=0))
        else:
            bounds_min = (0, 0, 0)
            bounds_max = (0, 0, 0)
        
        return ModelInfo(
            name=mesh.name,
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.indices) // 3 if mesh.indices is not None else 0,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            has_normals=mesh.normals is not None,
            has_texcoords=mesh.texcoords is not None,
            has_colors=mesh.colors is not None
        )
    
    def optimize_mesh(self, mesh: Mesh3D) -> Mesh3D:
        """
        Optimize a mesh for rendering performance.
        
        Args:
            mesh: Mesh to optimize
            
        Returns:
            Optimized mesh
        """
        # For now, just ensure normals are present
        if mesh.normals is None:
            mesh.normals = mesh._calculate_normals()
        
        self.logger.debug("Mesh optimized", extra={
            "name": mesh.name,
            "vertices": len(mesh.vertices)
        })
        
        return mesh
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.model_cache.clear()
        self.logger.debug("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache."""
        return {
            "cached_models": len(self.model_cache),
            "cache_keys": list(self.model_cache.keys())
        }