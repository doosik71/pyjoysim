# PyJoySim Project Overview

## 1. Project Goal

PyJoySim is a Python-based simulation platform that utilizes joysticks for intuitive control. It aims to provide a versatile tool for educational and research purposes, covering a wide range of simulations from vehicle dynamics to robotics and space exploration.

## 2. Core Features

- **Joystick Integration**: Advanced support for various joysticks, including hot-plugging, custom key mapping, and multi-device management.
- **Modular Simulation Framework**: A flexible architecture that allows for easy addition of new simulations.
- **Physics Engine**: A abstracted physics layer supporting both 2D (Pymunk) and 3D (PyBullet) simulations.
- **Rendering System**: A rendering engine capable of handling 2D and 3D graphics with features like custom shaders, lighting, and post-processing effects.
- **User-Friendly UI**: An intuitive interface for selecting and controlling simulations, complete with real-time performance monitoring.

## 3. Development Roadmap

The project is divided into four main phases:

### Phase 1: Foundation (Completed)

- **Status**: ✅ Completed
- **Key Achievements**:
    - Core joystick input system.
    - 2D physics and rendering engines.
    - A stable and extensible simulation framework.
    - Basic demos like a bouncing ball.

### Phase 2: Basic Simulations (Completed)

- **Status**: ✅ Completed
- **Key Achievements**:
    - **2D Car Simulation**: Realistic vehicle dynamics, track system, and joystick controls.
    - **Robot Arm Simulation**: 3-DOF robotic arm with inverse kinematics and PID control.
    - **Integrated UI**: A comprehensive user interface for simulation selection, control, and settings management.

### Phase 3: Advanced Features (In Progress)

- **Status**: ⏳ In Progress
- **Current Focus**:
    - **3D Rendering Engine**: Integration of ModernGL/PyOpenGL for 3D environments.
    - **3D Physics**: Integration of PyBullet for 3D rigid body dynamics.
    - **Drone Simulation**: Developing a realistic quadcopter simulation with various flight modes.
- **Upcoming**:
    - Spaceship and Submarine simulations.
    - Advanced visual and physics effects (particles, shaders).
    - Foundational work for multiplayer capabilities.

### Phase 4: Completion & Optimization (Planned)

- **Status**: 📝 Planned
- **Goals**:
    - Performance optimization across all simulations.
    - Extensive bug fixing and stability improvements.
    - Comprehensive documentation for users and developers.
    - Packaging for distribution (PyPI, executables).
    - Building a community contribution system.

## 4. Key Technologies

- **Language**: Python 3.8+
- **Core Libraries**:
    - `pygame`: For 2D rendering and window management.
    - `pymunk`: For 2D physics.
    - `pybullet`: For 3D physics.
    - `moderngl`: For 3D rendering (OpenGL).
    - `numpy`, `scipy`: For numerical calculations.
- **UI**: Custom UI framework built on `pygame`.
- **Dependency Management**: `uv`

## 5. How to Contribute

We welcome contributions from the community! Please refer to the (upcoming) `CONTRIBUTING.md` for guidelines on how to:
- Report bugs or request features.
- Contribute code or documentation.
- Develop new simulation plugins.

This document provides a high-level overview of the PyJoySim project. For more detailed information, please refer to the documents in the `docs/` directory.
