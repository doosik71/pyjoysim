#!/usr/bin/env python3
"""
Basic 3D system test without external dependencies.
"""

import sys
import os
import numpy as np

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pyjoysim.physics.physics3d import (
    Physics3D, PhysicsMode, Vector3D, Quaternion,
    create_box_3d, create_sphere_3d
)
from pyjoysim.rendering.camera3d import Camera3D, CameraMode
from pyjoysim.rendering.model_loader import ModelLoader, PrimitiveType

def test_physics_3d():
    """Test basic 3D physics functionality."""
    print("Testing 3D Physics...")
    
    try:
        # Create physics world
        physics = Physics3D(PhysicsMode.HEADLESS)
        
        # Create test objects
        box = create_box_3d("test_box", Vector3D(1, 1, 1), Vector3D(0, 5, 0), 1.0)
        sphere = create_sphere_3d("test_sphere", 0.5, Vector3D(2, 5, 0), 1.0)
        
        physics.add_object(box)
        physics.add_object(sphere)
        
        # Run simulation
        for i in range(60):  # 1 second at 60 FPS
            physics.step(1.0 / 60.0)
        
        # Check that objects have fallen
        print(f"Box final position: {box.position.to_tuple()}")
        print(f"Sphere final position: {sphere.position.to_tuple()}")
        
        assert box.position.y < 5.0, "Box should have fallen"
        assert sphere.position.y < 5.0, "Sphere should have fallen"
        
        physics.cleanup()
        print("[PASS] 3D Physics test passed")
        
    except Exception as e:
        print(f"[FAIL] 3D Physics test failed: {e}")
        return False
    
    return True

def test_camera_3d():
    """Test 3D camera functionality."""
    print("Testing 3D Camera...")
    
    try:
        # Create camera
        camera = Camera3D(800, 600)
        
        # Test basic operations
        camera.set_position(np.array([5, 5, 5]))
        camera.look_at(np.array([0, 0, 0]))
        
        # Test camera modes
        camera.set_mode(CameraMode.FREE)
        assert camera.mode == CameraMode.FREE
        
        # Test transformations
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        
        assert view_matrix.shape == (4, 4), "View matrix should be 4x4"
        assert proj_matrix.shape == (4, 4), "Projection matrix should be 4x4"
        
        # Test world to screen conversion
        screen_pos = camera.world_to_screen(np.array([0, 0, 0]))
        assert len(screen_pos) == 3, "Screen position should have 3 components"
        
        print("[PASS] 3D Camera test passed")
        
    except Exception as e:
        print(f"[FAIL] 3D Camera test failed: {e}")
        return False
    
    return True

def test_model_loader():
    """Test model loading functionality."""
    print("Testing Model Loader...")
    
    try:
        loader = ModelLoader()
        
        # Test primitive creation
        cube = loader.create_primitive(PrimitiveType.CUBE)
        sphere = loader.create_primitive(PrimitiveType.SPHERE)
        cylinder = loader.create_primitive(PrimitiveType.CYLINDER)
        
        # Verify mesh properties
        assert cube.vertices is not None, "Cube should have vertices"
        assert cube.indices is not None, "Cube should have indices"
        assert cube.normals is not None, "Cube should have normals"
        
        assert sphere.vertices is not None, "Sphere should have vertices"
        assert len(sphere.vertices) > 0, "Sphere should have vertices"
        
        # Test model info
        cube_info = loader.get_model_info(cube)
        assert cube_info.vertex_count > 0, "Cube should have vertices"
        assert cube_info.triangle_count > 0, "Cube should have triangles"
        
        print(f"Cube: {cube_info.vertex_count} vertices, {cube_info.triangle_count} triangles")
        print(f"Sphere: {len(sphere.vertices)} vertices")
        
        print("[PASS] Model Loader test passed")
        
    except Exception as e:
        print(f"[FAIL] Model Loader test failed: {e}")
        return False
    
    return True

def test_vector_math():
    """Test vector and quaternion math."""
    print("Testing Vector Math...")
    
    try:
        # Test Vector3D
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        
        v3 = v1 + v2
        assert v3.x == 5 and v3.y == 7 and v3.z == 9
        
        v4 = v1 * 2
        assert v4.x == 2 and v4.y == 4 and v4.z == 6
        
        mag = v1.magnitude()
        assert abs(mag - np.sqrt(14)) < 1e-6
        
        # Test Quaternion
        q = Quaternion()
        assert q.w == 1.0  # Identity quaternion
        
        euler = Vector3D(0, np.pi/2, 0)  # 90 degree rotation around Y
        q2 = Quaternion.from_euler(euler)
        euler_back = q2.to_euler()
        
        # Should be close to original
        assert abs(euler_back.y - np.pi/2) < 1e-6
        
        print("[PASS] Vector Math test passed")
        
    except Exception as e:
        print(f"[FAIL] Vector Math test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running PyJoySim 3D System Tests")
    print("=" * 40)
    
    tests = [
        test_vector_math,
        test_model_loader,
        test_camera_3d,
        test_physics_3d,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("SUCCESS: All tests passed! 3D system is working correctly.")
        return 0
    else:
        print("FAILURE: Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())