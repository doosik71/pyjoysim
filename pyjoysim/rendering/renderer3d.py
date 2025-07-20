"""
3D Renderer using ModernGL for PyJoySim.

This module provides 3D rendering capabilities including:
- OpenGL-based 3D rendering
- Shader management
- Mesh rendering
- Lighting and materials
- Texture support
"""

import math
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import struct

import numpy as np

try:
    import moderngl as mgl
    import moderngl_window as mglw
    from PIL import Image
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from .camera3d import Camera3D, CameraMode
from ..physics.physics3d import Vector3D, Quaternion, PhysicsObject3D
from ..core.logging import get_logger


class PrimitiveType(Enum):
    """3D primitive types."""
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    PLANE = "plane"
    CONE = "cone"


@dataclass
class Material3D:
    """3D material properties for rendering."""
    name: str = "default"
    
    # Colors
    ambient: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    diffuse: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    specular: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Properties
    shininess: float = 32.0
    transparency: float = 1.0  # 1.0 = opaque, 0.0 = transparent
    
    # Textures (texture IDs)
    diffuse_texture: Optional[int] = None
    normal_texture: Optional[int] = None
    specular_texture: Optional[int] = None


@dataclass
class Light3D:
    """3D light source."""
    name: str
    light_type: str  # "directional", "point", "spot"
    
    # Position/Direction
    position: Vector3D = None
    direction: Vector3D = None
    
    # Colors
    ambient: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    diffuse: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    specular: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Attenuation (for point/spot lights)
    constant: float = 1.0
    linear: float = 0.09
    quadratic: float = 0.032
    
    # Spotlight properties
    cutoff: float = math.cos(math.radians(12.5))      # Inner cutoff
    outer_cutoff: float = math.cos(math.radians(15.0)) # Outer cutoff
    
    # State
    enabled: bool = True
    
    def __post_init__(self):
        if self.position is None:
            self.position = Vector3D(0, 10, 0)
        if self.direction is None:
            self.direction = Vector3D(0, -1, 0)


@dataclass
class Mesh3D:
    """3D mesh data."""
    name: str
    vertices: np.ndarray       # Vertex positions [x, y, z]
    indices: np.ndarray        # Triangle indices
    normals: Optional[np.ndarray] = None      # Vertex normals
    texcoords: Optional[np.ndarray] = None    # Texture coordinates
    colors: Optional[np.ndarray] = None       # Vertex colors
    
    # OpenGL resources (set by renderer)
    vao: Optional[Any] = None
    vbo: Optional[Any] = None
    ebo: Optional[Any] = None
    
    def __post_init__(self):
        """Calculate normals if not provided."""
        if self.normals is None:
            self.normals = self._calculate_normals()
    
    def _calculate_normals(self) -> np.ndarray:
        """Calculate vertex normals from face normals."""
        if self.indices is None:
            return np.zeros_like(self.vertices)
        
        # Initialize normals array
        normals = np.zeros_like(self.vertices)
        
        # Calculate face normals and accumulate to vertices
        for i in range(0, len(self.indices), 3):
            if i + 2 >= len(self.indices):
                break
            
            # Get triangle vertices
            v0 = self.vertices[self.indices[i]]
            v1 = self.vertices[self.indices[i + 1]]
            v2 = self.vertices[self.indices[i + 2]]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Normalize
            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal = face_normal / norm
            
            # Accumulate to vertices
            normals[self.indices[i]] += face_normal
            normals[self.indices[i + 1]] += face_normal
            normals[self.indices[i + 2]] += face_normal
        
        # Normalize vertex normals
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] = normals[i] / norm
        
        return normals


@dataclass
class RenderObject3D:
    """3D renderable object."""
    name: str
    mesh: Mesh3D
    material: Material3D
    
    # Transform
    position: Vector3D = None
    rotation: Quaternion = None
    scale: Vector3D = None
    
    # State
    visible: bool = True
    cast_shadows: bool = True
    receive_shadows: bool = True
    
    def __post_init__(self):
        if self.position is None:
            self.position = Vector3D()
        if self.rotation is None:
            self.rotation = Quaternion()
        if self.scale is None:
            self.scale = Vector3D(1, 1, 1)
    
    def get_model_matrix(self) -> np.ndarray:
        """Get the model transformation matrix."""
        # Create transformation matrices
        translation = np.eye(4)
        translation[0, 3] = self.position.x
        translation[1, 3] = self.position.y
        translation[2, 3] = self.position.z
        
        # Convert quaternion to rotation matrix
        rotation_matrix = self._quaternion_to_matrix(self.rotation)
        
        # Scale matrix
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = self.scale.x
        scale_matrix[1, 1] = self.scale.y
        scale_matrix[2, 2] = self.scale.z
        
        # Combine: T * R * S
        return translation @ rotation_matrix @ scale_matrix
    
    def _quaternion_to_matrix(self, q: Quaternion) -> np.ndarray:
        """Convert quaternion to 4x4 rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        
        # First row
        r00 = 1 - 2 * (y*y + z*z)
        r01 = 2 * (x*y - z*w)
        r02 = 2 * (x*z + y*w)
        
        # Second row
        r10 = 2 * (x*y + z*w)
        r11 = 1 - 2 * (x*x + z*z)
        r12 = 2 * (y*z - x*w)
        
        # Third row
        r20 = 2 * (x*z - y*w)
        r21 = 2 * (y*z + x*w)
        r22 = 1 - 2 * (x*x + y*y)
        
        return np.array([
            [r00, r01, r02, 0],
            [r10, r11, r12, 0],
            [r20, r21, r22, 0],
            [0,   0,   0,   1]
        ])


class Renderer3D:
    """
    3D renderer using ModernGL.
    
    Provides OpenGL-based 3D rendering with support for:
    - Mesh rendering
    - Lighting and shading
    - Texture mapping
    - Shadow mapping
    - Material systems
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize 3D renderer.
        
        Args:
            width: Viewport width
            height: Viewport height
        """
        if not MODERNGL_AVAILABLE:
            raise RuntimeError(
                "ModernGL is required for 3D rendering. "
                "Install with: pip install moderngl moderngl-window"
            )
        
        self.logger = get_logger("renderer_3d")
        
        # Viewport
        self.width = width
        self.height = height
        
        # ModernGL context (will be set by window)
        self.ctx: Optional[Any] = None
        
        # Rendering objects
        self.render_objects: Dict[str, RenderObject3D] = {}
        self.lights: Dict[str, Light3D] = {}
        self.materials: Dict[str, Material3D] = {}
        self.textures: Dict[str, Any] = {}
        
        # Primitive meshes cache
        self.primitive_meshes: Dict[PrimitiveType, Mesh3D] = {}
        
        # Shaders
        self.shaders: Dict[str, Any] = {}
        
        # Default materials and lights
        self._create_default_materials()
        self._create_default_lights()
        
        # Rendering settings
        self.background_color = (0.2, 0.3, 0.3, 1.0)
        self.wireframe_mode = False
        self.enable_depth_test = True
        self.enable_face_culling = True
        
        self.logger.info("Renderer3D initialized", extra={
            "width": width,
            "height": height
        })
    
    def initialize_context(self, ctx: Any) -> None:
        """
        Initialize ModernGL context and resources.
        
        Args:
            ctx: ModernGL context
        """
        self.ctx = ctx
        
        # Enable depth testing
        if self.enable_depth_test and hasattr(self.ctx, 'DEPTH_TEST'):
            self.ctx.enable(self.ctx.DEPTH_TEST)
        
        # Enable face culling
        if self.enable_face_culling and hasattr(self.ctx, 'CULL_FACE'):
            self.ctx.enable(self.ctx.CULL_FACE)
        
        # Create shaders
        self._create_shaders()
        
        # Create primitive meshes
        self._create_primitive_meshes()
        
        self.logger.debug("ModernGL context initialized")
    
    def _create_default_materials(self) -> None:
        """Create default materials."""
        # Default material
        self.materials["default"] = Material3D(
            name="default",
            diffuse=(0.8, 0.8, 0.8)
        )
        
        # Debug materials
        self.materials["red"] = Material3D(
            name="red",
            diffuse=(0.8, 0.2, 0.2)
        )
        
        self.materials["green"] = Material3D(
            name="green",
            diffuse=(0.2, 0.8, 0.2)
        )
        
        self.materials["blue"] = Material3D(
            name="blue",
            diffuse=(0.2, 0.2, 0.8)
        )
    
    def _create_default_lights(self) -> None:
        """Create default lighting setup."""
        # Main directional light (sun)
        self.lights["sun"] = Light3D(
            name="sun",
            light_type="directional",
            direction=Vector3D(-0.3, -1.0, -0.3),
            diffuse=(1.0, 1.0, 0.9),
            ambient=(0.2, 0.2, 0.2)
        )
        
        # Fill light
        self.lights["fill"] = Light3D(
            name="fill",
            light_type="directional",
            direction=Vector3D(0.5, -0.5, 0.5),
            diffuse=(0.3, 0.3, 0.4),
            ambient=(0.0, 0.0, 0.0)
        )
    
    def _create_shaders(self) -> None:
        """Create shader programs."""
        if not self.ctx:
            return
        
        # Basic vertex shader with shadow mapping support
        vertex_shader = """
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec2 in_texcoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normal_matrix;
        uniform mat4 light_space_matrix;
        
        out vec3 frag_pos;
        out vec3 frag_normal;
        out vec2 frag_texcoord;
        out vec4 frag_pos_light_space;
        
        void main() {
            vec4 world_pos = model * vec4(in_position, 1.0);
            frag_pos = world_pos.xyz;
            frag_normal = normalize(normal_matrix * in_normal);
            frag_texcoord = in_texcoord;
            frag_pos_light_space = light_space_matrix * world_pos;
            
            gl_Position = projection * view * world_pos;
        }
        """
        
        # Basic fragment shader with Phong lighting and shadow mapping
        fragment_shader = """
        #version 330 core
        
        in vec3 frag_pos;
        in vec3 frag_normal;
        in vec2 frag_texcoord;
        in vec4 frag_pos_light_space;
        
        out vec4 frag_color;
        
        // Material properties
        uniform vec3 material_ambient;
        uniform vec3 material_diffuse;
        uniform vec3 material_specular;
        uniform float material_shininess;
        uniform float material_transparency;
        
        // Lighting
        uniform vec3 light_direction;
        uniform vec3 light_ambient;
        uniform vec3 light_diffuse;
        uniform vec3 light_specular;
        
        uniform vec3 view_pos;
        
        // Shadow mapping
        uniform sampler2D shadow_map;
        uniform bool enable_shadows;
        
        float ShadowCalculation(vec4 fragPosLightSpace) {
            if (!enable_shadows) return 0.0;
            
            // Perspective divide
            vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
            // Transform to [0,1] range
            projCoords = projCoords * 0.5 + 0.5;
            
            // Keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
            if(projCoords.z > 1.0)
                return 0.0;
            
            // Get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
            float closestDepth = texture(shadow_map, projCoords.xy).r; 
            // Get depth of current fragment from light's perspective
            float currentDepth = projCoords.z;
            
            // Check whether current frag pos is in shadow
            float bias = 0.005;
            return currentDepth - bias > closestDepth ? 1.0 : 0.0;
        }
        
        void main() {
            // Normalize normal
            vec3 normal = normalize(frag_normal);
            
            // Ambient
            vec3 ambient = light_ambient * material_ambient;
            
            // Diffuse
            vec3 light_dir = normalize(-light_direction);
            float diff = max(dot(normal, light_dir), 0.0);
            vec3 diffuse = light_diffuse * (diff * material_diffuse);
            
            // Specular
            vec3 view_dir = normalize(view_pos - frag_pos);
            vec3 reflect_dir = reflect(-light_dir, normal);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material_shininess);
            vec3 specular = light_specular * (spec * material_specular);
            
            // Shadow calculation
            float shadow = ShadowCalculation(frag_pos_light_space);
            
            // Apply shadows only to diffuse and specular components
            vec3 result = ambient + (1.0 - shadow) * (diffuse + specular);
            frag_color = vec4(result, material_transparency);
        }
        """
        
        try:
            self.shaders["basic"] = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            self.logger.debug("Basic shader created successfully")
        except Exception as e:
            self.logger.error("Failed to create basic shader", extra={"error": str(e)})
    
    def _create_primitive_meshes(self) -> None:
        """Create primitive mesh geometries."""
        # Cube
        self.primitive_meshes[PrimitiveType.CUBE] = self._create_cube_mesh()
        
        # Sphere
        self.primitive_meshes[PrimitiveType.SPHERE] = self._create_sphere_mesh()
        
        # Plane
        self.primitive_meshes[PrimitiveType.PLANE] = self._create_plane_mesh()
        
        self.logger.debug("Primitive meshes created")
    
    def _create_cube_mesh(self) -> Mesh3D:
        """Create a unit cube mesh."""
        # Cube vertices (8 corners)
        vertices = np.array([
            # Front face
            [-0.5, -0.5,  0.5],  # 0
            [ 0.5, -0.5,  0.5],  # 1
            [ 0.5,  0.5,  0.5],  # 2
            [-0.5,  0.5,  0.5],  # 3
            # Back face
            [-0.5, -0.5, -0.5],  # 4
            [ 0.5, -0.5, -0.5],  # 5
            [ 0.5,  0.5, -0.5],  # 6
            [-0.5,  0.5, -0.5],  # 7
        ], dtype=np.float32)
        
        # Cube indices (12 triangles, 36 vertices)
        indices = np.array([
            # Front
            0, 1, 2,  2, 3, 0,
            # Back
            4, 7, 6,  6, 5, 4,
            # Left
            4, 0, 3,  3, 7, 4,
            # Right
            1, 5, 6,  6, 2, 1,
            # Top
            3, 2, 6,  6, 7, 3,
            # Bottom
            4, 5, 1,  1, 0, 4,
        ], dtype=np.uint32)
        
        # Texture coordinates
        texcoords = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1],  # Front
            [1, 0], [0, 0], [0, 1], [1, 1],  # Back
        ], dtype=np.float32)
        
        return Mesh3D("cube", vertices, indices, texcoords=texcoords)
    
    def _create_sphere_mesh(self, radius: float = 0.5, rings: int = 16, sectors: int = 32) -> Mesh3D:
        """Create a sphere mesh."""
        vertices = []
        texcoords = []
        indices = []
        
        # Generate vertices
        for i in range(rings + 1):
            lat = math.pi * (-0.5 + float(i) / rings)
            y = radius * math.sin(lat)
            radius_at_lat = radius * math.cos(lat)
            
            for j in range(sectors + 1):
                lon = 2 * math.pi * float(j) / sectors
                x = radius_at_lat * math.cos(lon)
                z = radius_at_lat * math.sin(lon)
                
                vertices.append([x, y, z])
                texcoords.append([float(j) / sectors, float(i) / rings])
        
        # Generate indices
        for i in range(rings):
            for j in range(sectors):
                first = i * (sectors + 1) + j
                second = first + sectors + 1
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        vertices = np.array(vertices, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        return Mesh3D("sphere", vertices, indices, texcoords=texcoords)
    
    def _create_plane_mesh(self, size: float = 1.0) -> Mesh3D:
        """Create a plane mesh."""
        half_size = size * 0.5
        
        vertices = np.array([
            [-half_size, 0,  half_size],  # Top-left
            [ half_size, 0,  half_size],  # Top-right
            [ half_size, 0, -half_size],  # Bottom-right
            [-half_size, 0, -half_size],  # Bottom-left
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,  2, 3, 0
        ], dtype=np.uint32)
        
        texcoords = np.array([
            [0, 1], [1, 1], [1, 0], [0, 0]
        ], dtype=np.float32)
        
        # Normal pointing up
        normals = np.array([
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]
        ], dtype=np.float32)
        
        return Mesh3D("plane", vertices, indices, normals, texcoords)
    
    def create_mesh_from_physics_object(self, physics_obj: PhysicsObject3D) -> Mesh3D:
        """
        Create a mesh from a physics object.
        
        Args:
            physics_obj: Physics object to create mesh for
            
        Returns:
            Mesh3D object
        """
        shape = physics_obj.shape
        
        if shape.shape_type.value == "box":
            mesh = self.primitive_meshes[PrimitiveType.CUBE].copy()
            # Scale vertices by shape dimensions
            mesh.vertices = mesh.vertices * np.array([
                shape.dimensions.x,
                shape.dimensions.y, 
                shape.dimensions.z
            ])
            mesh.name = f"{physics_obj.name}_mesh"
            return mesh
        
        elif shape.shape_type.value == "sphere":
            mesh = self.primitive_meshes[PrimitiveType.SPHERE].copy()
            # Scale by radius
            mesh.vertices = mesh.vertices * shape.dimensions.x
            mesh.name = f"{physics_obj.name}_mesh"
            return mesh
        
        elif shape.shape_type.value == "plane":
            mesh = self.primitive_meshes[PrimitiveType.PLANE].copy()
            mesh.name = f"{physics_obj.name}_mesh"
            return mesh
        
        else:
            # Default to cube for unsupported shapes
            self.logger.warning("Unsupported shape type, using cube", extra={
                "shape_type": shape.shape_type.value
            })
            mesh = self.primitive_meshes[PrimitiveType.CUBE].copy()
            mesh.name = f"{physics_obj.name}_mesh"
            return mesh
    
    def add_render_object(self, render_obj: RenderObject3D) -> None:
        """
        Add a render object to the scene.
        
        Args:
            render_obj: Render object to add
        """
        if not self.ctx:
            self.logger.error("Context not initialized")
            return
        
        # Create OpenGL resources for mesh
        self._create_mesh_buffers(render_obj.mesh)
        
        # Store object
        self.render_objects[render_obj.name] = render_obj
        
        self.logger.debug("Render object added", extra={"name": render_obj.name})
    
    def _create_mesh_buffers(self, mesh: Mesh3D) -> None:
        """Create OpenGL buffers for a mesh."""
        if not self.ctx or mesh.vao is not None:
            return
        
        # Prepare vertex data
        vertex_data = []
        
        for i in range(len(mesh.vertices)):
            # Position
            vertex_data.extend(mesh.vertices[i])
            
            # Normal
            if mesh.normals is not None and i < len(mesh.normals):
                vertex_data.extend(mesh.normals[i])
            else:
                vertex_data.extend([0.0, 1.0, 0.0])  # Default up normal
            
            # Texture coordinates
            if mesh.texcoords is not None and i < len(mesh.texcoords):
                vertex_data.extend(mesh.texcoords[i])
            else:
                vertex_data.extend([0.0, 0.0])  # Default texture coordinates
        
        vertex_data = np.array(vertex_data, dtype=np.float32)
        
        # Create buffers
        mesh.vbo = self.ctx.buffer(vertex_data.tobytes())
        
        if mesh.indices is not None:
            mesh.ebo = self.ctx.buffer(mesh.indices.tobytes())
        
        # Create VAO
        mesh.vao = self.ctx.vertex_array(
            self.shaders["basic"],
            [
                (mesh.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord')
            ],
            mesh.ebo
        )
    
    def remove_render_object(self, name: str) -> None:
        """
        Remove a render object from the scene.
        
        Args:
            name: Name of object to remove
        """
        if name in self.render_objects:
            # Clean up OpenGL resources
            obj = self.render_objects[name]
            if obj.mesh.vao:
                obj.mesh.vao.release()
            if obj.mesh.vbo:
                obj.mesh.vbo.release()
            if obj.mesh.ebo:
                obj.mesh.ebo.release()
            
            del self.render_objects[name]
            self.logger.debug("Render object removed", extra={"name": name})
    
    def render(self, camera: Camera3D) -> None:
        """
        Render the 3D scene.
        
        Args:
            camera: Camera for rendering
        """
        if not self.ctx:
            self.logger.error("Context not initialized")
            return
        
        # Clear buffers
        self.ctx.clear(*self.background_color)
        
        # Set viewport
        self.ctx.viewport = (0, 0, self.width, self.height)
        
        # Get camera matrices
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        
        # Set wireframe mode
        if self.wireframe_mode:
            self.ctx.wireframe = True
        else:
            self.ctx.wireframe = False
        
        # Render all objects
        for render_obj in self.render_objects.values():
            if not render_obj.visible or not render_obj.mesh.vao:
                continue
            
            self._render_object(render_obj, view_matrix, proj_matrix, camera.position)
    
    def _render_object(self, 
                      render_obj: RenderObject3D, 
                      view_matrix: np.ndarray,
                      proj_matrix: np.ndarray,
                      camera_pos: np.ndarray) -> None:
        """Render a single object."""
        shader = self.shaders["basic"]
        shader.use()
        
        # Set matrices
        model_matrix = render_obj.get_model_matrix()
        normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T
        
        shader['model'].write(model_matrix.astype(np.float32).tobytes())
        shader['view'].write(view_matrix.astype(np.float32).tobytes())
        shader['projection'].write(proj_matrix.astype(np.float32).tobytes())
        shader['normal_matrix'].write(normal_matrix.astype(np.float32).tobytes())
        
        # Set material properties
        material = render_obj.material
        shader['material_ambient'].value = material.ambient
        shader['material_diffuse'].value = material.diffuse
        shader['material_specular'].value = material.specular
        shader['material_shininess'].value = material.shininess
        shader['material_transparency'].value = material.transparency
        
        # Set lighting (use first directional light)
        main_light = next((light for light in self.lights.values() 
                         if light.enabled and light.light_type == "directional"), None)
        
        if main_light:
            shader['light_direction'].value = main_light.direction.to_tuple()
            shader['light_ambient'].value = main_light.ambient
            shader['light_diffuse'].value = main_light.diffuse
            shader['light_specular'].value = main_light.specular
        
        # Set camera position
        shader['view_pos'].value = tuple(camera_pos)
        
        # Render mesh
        render_obj.mesh.vao.render()
    
    def resize(self, width: int, height: int) -> None:
        """
        Resize the viewport.
        
        Args:
            width: New width
            height: New height
        """
        self.width = width
        self.height = height
        
        if self.ctx:
            self.ctx.viewport = (0, 0, width, height)
    
    def add_light(self, light: Light3D) -> None:
        """Add a light to the scene."""
        self.lights[light.name] = light
    
    def remove_light(self, name: str) -> None:
        """Remove a light from the scene."""
        if name in self.lights:
            del self.lights[name]
    
    def add_material(self, material: Material3D) -> None:
        """Add a material to the library."""
        self.materials[material.name] = material
    
    def get_material(self, name: str) -> Optional[Material3D]:
        """Get a material by name."""
        return self.materials.get(name)
    
    def load_texture(self, name: str, file_path: str) -> bool:
        """
        Load a texture from file.
        
        Args:
            name: Texture name
            file_path: Path to texture file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ctx:
            self.logger.error("Context not initialized")
            return False
        
        try:
            # Load image
            img = Image.open(file_path)
            img = img.convert('RGB')
            img_data = np.array(img)
            
            # Create texture
            texture = self.ctx.texture((img.width, img.height), 3, img_data.tobytes())
            if hasattr(texture, 'filter'):
                texture.filter = (getattr(self.ctx, 'LINEAR_MIPMAP_LINEAR', 0x2703), 
                                  getattr(self.ctx, 'LINEAR', 0x2601))
            texture.build_mipmaps()
            
            self.textures[name] = texture
            
            self.logger.debug("Texture loaded", extra={
                "name": name,
                "path": file_path,
                "size": f"{img.width}x{img.height}"
            })
            
            return True
        
        except Exception as e:
            self.logger.error("Failed to load texture", extra={
                "name": name,
                "path": file_path,
                "error": str(e)
            })
            return False
    
    def cleanup(self) -> None:
        """Clean up renderer resources."""
        # Clean up render objects
        for name in list(self.render_objects.keys()):
            self.remove_render_object(name)
        
        # Clean up textures
        for texture in self.textures.values():
            texture.release()
        self.textures.clear()
        
        # Clean up shaders
        for shader in self.shaders.values():
            shader.release()
        self.shaders.clear()
        
        self.logger.info("Renderer3D cleaned up")


# Convenience functions
def create_render_object_from_physics(physics_obj: PhysicsObject3D, 
                                    renderer: Renderer3D,
                                    material_name: str = "default") -> RenderObject3D:
    """
    Create a render object from a physics object.
    
    Args:
        physics_obj: Physics object
        renderer: Renderer instance
        material_name: Material name to use
        
    Returns:
        RenderObject3D
    """
    mesh = renderer.create_mesh_from_physics_object(physics_obj)
    material = renderer.get_material(material_name) or renderer.materials["default"]
    
    return RenderObject3D(
        name=physics_obj.name,
        mesh=mesh,
        material=material,
        position=physics_obj.position,
        rotation=physics_obj.rotation
    )