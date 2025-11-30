# Game Engine Documentation

## Overview
This is a simple 3D game engine built with Pygame that demonstrates a rotating cube. The engine provides basic 3D mathematics, transformation, and rendering capabilities.

---

## Table of Contents
1. [Vector3 Class](#vector3-class)
2. [Cube Class](#cube-class)
3. [GameEngine Class](#gameengine-class)
4. [Constants](#constants)

---

## Vector3 Class

A class for handling 3D vector operations and transformations.

### Constructor

#### `__init__(self, x: float = 0, y: float = 0, z: float = 0)`
Initializes a Vector3 object with X, Y, and Z coordinates.

**Parameters:**
- `x` (float, optional): X coordinate. Defaults to 0.
- `y` (float, optional): Y coordinate. Defaults to 0.
- `z` (float, optional): Z coordinate. Defaults to 0.

**Example:**
```python
v = Vector3(1.0, 2.0, 3.0)
origin = Vector3()  # (0, 0, 0)
```

### Methods

#### `__add__(self, other: Vector3) -> Vector3`
Adds two vectors together (vector addition).

**Parameters:**
- `other` (Vector3): The vector to add to this vector.

**Returns:**
- (Vector3): A new Vector3 representing the sum.

**Example:**
```python
v1 = Vector3(1, 2, 3)
v2 = Vector3(4, 5, 6)
result = v1 + v2  # Vector3(5, 7, 9)
```

---

#### `__sub__(self, other: Vector3) -> Vector3`
Subtracts one vector from another (vector subtraction).

**Parameters:**
- `other` (Vector3): The vector to subtract from this vector.

**Returns:**
- (Vector3): A new Vector3 representing the difference.

**Example:**
```python
v1 = Vector3(5, 7, 9)
v2 = Vector3(1, 2, 3)
result = v1 - v2  # Vector3(4, 5, 6)
```

---

#### `__mul__(self, scalar: float) -> Vector3`
Multiplies a vector by a scalar value (scalar multiplication).

**Parameters:**
- `scalar` (float): The scalar value to multiply by.

**Returns:**
- (Vector3): A new Vector3 with each component multiplied by the scalar.

**Example:**
```python
v = Vector3(1, 2, 3)
result = v * 2  # Vector3(2, 4, 6)
```

---

#### `rotate_x(self, angle: float) -> Vector3`
Rotates the vector around the X axis by the specified angle.

**Parameters:**
- `angle` (float): The rotation angle in radians.

**Returns:**
- (Vector3): A new Vector3 representing the rotated vector.

**Details:**
- Uses the rotation matrix formula for rotation around the X axis.
- The Y and Z components are transformed while X remains unchanged.
- Rotation follows the right-hand rule.

**Example:**
```python
v = Vector3(0, 1, 0)
rotated = v.rotate_x(math.pi / 2)  # Rotate 90 degrees
```

---

#### `rotate_y(self, angle: float) -> Vector3`
Rotates the vector around the Y axis by the specified angle.

**Parameters:**
- `angle` (float): The rotation angle in radians.

**Returns:**
- (Vector3): A new Vector3 representing the rotated vector.

**Details:**
- Uses the rotation matrix formula for rotation around the Y axis.
- The X and Z components are transformed while Y remains unchanged.
- Rotation follows the right-hand rule.

**Example:**
```python
v = Vector3(1, 0, 0)
rotated = v.rotate_y(math.pi / 2)  # Rotate 90 degrees
```

---

#### `rotate_z(self, angle: float) -> Vector3`
Rotates the vector around the Z axis by the specified angle.

**Parameters:**
- `angle` (float): The rotation angle in radians.

**Returns:**
- (Vector3): A new Vector3 representing the rotated vector.

**Details:**
- Uses the rotation matrix formula for rotation around the Z axis.
- The X and Y components are transformed while Z remains unchanged.
- Rotation follows the right-hand rule.

**Example:**
```python
v = Vector3(1, 0, 0)
rotated = v.rotate_z(math.pi / 4)  # Rotate 45 degrees
```

---

#### `project_2d(self, distance: float = 500) -> Tuple[float, float]`
Projects a 3D vector onto a 2D screen coordinate system using perspective projection.

**Parameters:**
- `distance` (float, optional): The distance from the camera to the projection plane. Defaults to 500.

**Returns:**
- (Tuple[float, float]): A tuple of (x, y) screen coordinates.

**Details:**
- Uses perspective projection formula: `scale = distance / (distance + z)`
- X coordinate is centered at `SCREEN_WIDTH / 2`
- Y coordinate is centered at `SCREEN_HEIGHT / 2` (inverted to match screen coordinates)
- Objects farther away (higher Z) appear smaller
- Objects closer appear larger

**Example:**
```python
v = Vector3(100, 100, 300)
screen_pos = v.project_2d()  # Returns screen coordinates
```

---

## Cube Class

Represents a 3D cube object with rotation, position, and rendering capabilities.

### Constructor

#### `__init__(self, size: float = 100)`
Initializes a Cube object with a specified size.

**Parameters:**
- `size` (float, optional): The side length of the cube. Defaults to 100.

**Attributes:**
- `size` (float): The side length of the cube.
- `rotation` (Vector3): Current rotation angles (in radians) around X, Y, and Z axes.
- `position` (Vector3): The center position of the cube in 3D space.
- `vertices` (List[Vector3]): List of 8 vertices defining the cube corners.
- `edges` (List[Tuple[int, int]]): List of 12 edges as vertex index pairs.

**Example:**
```python
small_cube = Cube(size=50)
large_cube = Cube(size=200)
default_cube = Cube()  # 100x100x100
```

---

### Methods

#### `update(self, delta_time: float)`
Updates the cube's rotation based on elapsed time.

**Parameters:**
- `delta_time` (float): Time elapsed since the last update, in seconds.

**Details:**
- Rotates the cube around all three axes at different speeds for a dynamic effect.
- X-axis rotation speed: 1.0 radians per second
- Y-axis rotation speed: 1.5 radians per second
- Z-axis rotation speed: 0.5 radians per second
- All rotation angles accumulate over time.

**Example:**
```python
cube = Cube()
delta_time = 0.016  # ~60 FPS
cube.update(delta_time)
```

---

#### `get_transformed_vertices(self) -> List[Vector3]`
Calculates the vertices of the cube after applying rotations and position transformations.

**Returns:**
- (List[Vector3]): A list of 8 Vector3 objects representing transformed vertices.

**Details:**
- Applies rotations in order: X, then Y, then Z (intrinsic rotations).
- Applies position offset to all vertices.
- Does not modify the original vertices stored in the cube.
- Used internally by the rendering system.

**Example:**
```python
cube = Cube()
cube.rotation = Vector3(0.5, 1.0, 0.2)
cube.position = Vector3(10, 20, 300)
transformed = cube.get_transformed_vertices()
```

---

#### `draw(self, surface: pygame.Surface)`
Renders the cube onto a Pygame surface.

**Parameters:**
- `surface` (pygame.Surface): The Pygame surface to draw on (typically the main screen).

**Details:**
- Gets transformed vertices using `get_transformed_vertices()`.
- Projects each 3D vertex to 2D screen coordinates.
- Draws edges as white lines connecting vertices.
- Draws vertices as small green circles.
- Uses `EDGE_COLOR` and `CUBE_COLOR` constants for colors.

**Example:**
```python
cube = Cube()
pygame.display.set_mode((800, 600))
screen = pygame.display.get_surface()
cube.draw(screen)
```

---

## GameEngine Class

The main game engine class that manages the game loop, events, updates, and rendering.

### Constructor

#### `__init__(self)`
Initializes the GameEngine with Pygame setup, window creation, and cube instantiation.

**Attributes:**
- `screen` (pygame.Surface): The main display surface (800x600 pixels).
- `clock` (pygame.time.Clock): Pygame clock for managing frame rate.
- `running` (bool): Flag indicating whether the game loop should continue.
- `cube` (Cube): The cube object being displayed.

**Details:**
- Sets up a 800x600 pixel window titled "Simple 3D Game Engine - Rotating Cube".
- Creates a cube with size 150 at position (0, 0, 300).
- The cube is positioned 300 units away from the camera for proper perspective.

**Example:**
```python
engine = GameEngine()
```

---

### Methods

#### `handle_events(self)`
Processes user input and window events.

**Details:**
- Checks for `pygame.QUIT` event (window close button).
- Checks for `pygame.KEYDOWN` event with `pygame.K_ESCAPE` key.
- Sets `self.running` to `False` when either quit condition is met.
- Allows users to exit the application gracefully.

**Example:**
```python
engine = GameEngine()
engine.handle_events()
```

---

#### `update(self, delta_time: float)`
Updates the game state.

**Parameters:**
- `delta_time` (float): Time elapsed since the last update, in seconds.

**Details:**
- Calls the `update()` method on the cube to animate its rotation.
- Can be extended to update other game objects or logic.

**Example:**
```python
engine = GameEngine()
delta_time = 0.016  # ~60 FPS
engine.update(delta_time)
```

---

#### `render(self)`
Renders the current frame to the screen.

**Details:**
- Clears the screen with `BACKGROUND_COLOR` (dark blue).
- Draws the cube using `cube.draw()`.
- Renders information text showing engine description.
- Renders FPS counter showing current frames per second.
- Updates the display using `pygame.display.flip()`.

**Example:**
```python
engine = GameEngine()
engine.render()
```

---

#### `run(self)`
The main game loop that orchestrates events, updates, and rendering.

**Details:**
- Runs continuously while `self.running` is `True`.
- Limits the frame rate to 60 FPS.
- Calculates `delta_time` by converting milliseconds to seconds.
- Calls `handle_events()`, `update()`, and `render()` in sequence each frame.
- Gracefully shuts down Pygame when the loop exits.

**Flow:**
1. Check if running
2. Cap frame rate at 60 FPS
3. Calculate delta time
4. Handle user input
5. Update game state
6. Render frame
7. Repeat until `self.running` is `False`

**Example:**
```python
if __name__ == "__main__":
    engine = GameEngine()
    engine.run()
```

---

## Constants

Global configuration constants used throughout the engine.

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `SCREEN_WIDTH` | int | 800 | Width of the display window in pixels. |
| `SCREEN_HEIGHT` | int | 600 | Height of the display window in pixels. |
| `FPS` | int | 60 | Target frames per second for the game loop. |
| `BACKGROUND_COLOR` | tuple | (20, 20, 40) | RGB color for the background (dark blue). |
| `CUBE_COLOR` | tuple | (0, 255, 100) | RGB color for cube vertices (bright green). |
| `EDGE_COLOR` | tuple | (255, 255, 255) | RGB color for cube edges (white). |

---

## Usage Example

```python
import pygame
import math
from main import GameEngine, Cube, Vector3

# Initialize and run the game engine
if __name__ == "__main__":
    engine = GameEngine()
    engine.run()
```

## Getting Started

### Prerequisites
- Python 3.6+
- Pygame library (`pip install pygame`)

### Running the Engine
```bash
python main.py
```

### Controls
- **ESC Key**: Exit the application
- **Window Close Button**: Exit the application

### Output
- A window displaying a rotating 3D cube
- FPS counter in the top-left corner
- Engine description text

---

## Technical Details

### Coordinate System
- **X-axis**: Horizontal (left-right)
- **Y-axis**: Vertical (up-down)
- **Z-axis**: Depth (towards-away from camera)

### Perspective Projection
The engine uses a simple perspective projection to convert 3D coordinates to 2D screen coordinates:
- Objects farther away (higher Z) appear smaller
- Objects closer appear larger
- The projection distance is 500 units by default

### Rotation
- All rotations use the standard 3D rotation matrices
- Rotations are applied in order: X, then Y, then Z
- Angles are specified in radians

### Rendering
- The cube is rendered as edges (lines) and vertices (points)
- No face culling or hidden surface removal is performed
- All vertices and edges are always visible

---

## Future Enhancements

Possible improvements to the game engine:

1. **Multiple Objects**: Support for rendering multiple 3D objects
2. **Lighting**: Basic lighting and shading
3. **Texturing**: Apply textures to cube faces
4. **User Input**: Keyboard/mouse controls for camera movement
5. **Face Culling**: Optimize rendering by hiding back-facing polygons
6. **Collision Detection**: Add basic collision system
7. **Animation System**: Support for keyframe animations
8. **File Format Support**: Load 3D models from common formats (OBJ, STL, etc.)

---

## License

This game engine is provided as-is for educational purposes.
