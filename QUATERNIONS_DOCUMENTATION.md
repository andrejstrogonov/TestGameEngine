# Quaternion-Based Game Engine Documentation

## Overview

This documentation covers a complete 3D game engine built with Pygame that uses quaternion mathematics for 3D rotations. Quaternions provide a robust and elegant solution for representing and manipulating 3D rotations without the limitations of Euler angles, such as gimbal lock.

---

## Table of Contents

1. [Quaternion Fundamentals](#quaternion-fundamentals)
2. [Vector3 Class](#vector3-class)
3. [Quaternion Class](#quaternion-class)
4. [QuaternionCube Class](#quaternioncube-class)
5. [QuaternionGameEngine Class](#quaterniongameengine-class)
6. [Constants](#constants)
7. [Mathematical Concepts](#mathematical-concepts)
8. [Usage Examples](#usage-examples)

---

## Quaternion Fundamentals

### What is a Quaternion?

A quaternion is a mathematical entity consisting of four components:

```
q = w + xi + yj + zk
```

Or in component form:
```
q = (w, x, y, z)
```

Where:
- **w** is the scalar (real) part
- **(x, y, z)** is the vector (imaginary) part
- **i, j, k** are imaginary units with the properties: i² = j² = k² = ijk = -1

### Why Use Quaternions for 3D Rotations?

**Advantages:**
1. **No Gimbal Lock**: Unlike Euler angles, quaternions never suffer from gimbal lock
2. **Smooth Interpolation**: Quaternions can be smoothly interpolated (SLERP) between rotations
3. **Efficient**: Requires fewer operations than rotation matrices
4. **Numerically Stable**: Less prone to numerical errors when properly normalized
5. **Composable**: Multiple rotations can be easily combined via multiplication

**Disadvantages:**
1. Less intuitive than Euler angles
2. Four components to store (vs. three for Euler angles)
3. Requires normalization to maintain unit quaternion property

### Unit Quaternions

A **unit quaternion** (also called a versor) has magnitude 1:
```
|q| = √(w² + x² + y² + z²) = 1
```

Unit quaternions represent rotations in 3D space. Non-unit quaternions do not represent valid rotations.

---

## Vector3 Class

A 3D vector class with support for vector operations and 3D-to-2D projection.

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
- `other` (Vector3): The vector to add.

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
- `other` (Vector3): The vector to subtract.

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
- (Vector3): A new Vector3 with each component scaled.

**Example:**
```python
v = Vector3(1, 2, 3)
result = v * 2  # Vector3(2, 4, 6)
```

---

#### `__truediv__(self, scalar: float) -> Vector3`

Divides a vector by a scalar value (scalar division).

**Parameters:**
- `scalar` (float): The scalar value to divide by (must not be zero).

**Returns:**
- (Vector3): A new Vector3 with each component divided.

**Raises:**
- ValueError: If scalar is 0.

**Example:**
```python
v = Vector3(2, 4, 6)
result = v / 2  # Vector3(1, 2, 3)
```

---

#### `dot(self, other: Vector3) -> float`

Calculates the dot product (scalar product) of two vectors.

**Parameters:**
- `other` (Vector3): The other vector.

**Returns:**
- (float): The dot product value.

**Formula:**
```
a · b = a.x * b.x + a.y * b.y + a.z * b.z
```

**Properties:**
- Returns 0 if vectors are perpendicular
- Returns positive value if angle < 90°
- Returns negative value if angle > 90°

**Example:**
```python
v1 = Vector3(1, 0, 0)
v2 = Vector3(1, 0, 0)
result = v1.dot(v2)  # 1.0 (same direction)
```

---

#### `cross(self, other: Vector3) -> Vector3`

Calculates the cross product of two vectors.

**Parameters:**
- `other` (Vector3): The other vector.

**Returns:**
- (Vector3): A new Vector3 perpendicular to both input vectors.

**Formula:**
```
a × b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
```

**Properties:**
- Result is perpendicular to both input vectors
- Magnitude equals the area of the parallelogram formed by the vectors
- Order matters: a × b ≠ b × a (anti-commutative)

**Example:**
```python
x_axis = Vector3(1, 0, 0)
y_axis = Vector3(0, 1, 0)
z_axis = x_axis.cross(y_axis)  # Vector3(0, 0, 1)
```

---

#### `magnitude(self) -> float`

Calculates the length (magnitude) of the vector.

**Returns:**
- (float): The magnitude of the vector.

**Formula:**
```
|v| = √(x² + y² + z²)
```

**Example:**
```python
v = Vector3(3, 4, 0)
length = v.magnitude()  # 5.0
```

---

#### `normalize(self) -> Vector3`

Returns a normalized (unit) vector in the same direction.

**Returns:**
- (Vector3): A new Vector3 with magnitude 1 (unless original vector was zero).

**Details:**
- Returns zero vector if magnitude is 0 to avoid division by zero.
- Useful for direction vectors.

**Formula:**
```
v_normalized = v / |v|
```

**Example:**
```python
v = Vector3(3, 4, 0)
unit_v = v.normalize()  # Vector3(0.6, 0.8, 0)
```

---

#### `project_2d(self, distance: float = 500) -> Tuple[float, float]`

Projects a 3D vector onto a 2D screen coordinate system using perspective projection.

**Parameters:**
- `distance` (float, optional): Distance from camera to projection plane. Defaults to 500.

**Returns:**
- (Tuple[float, float]): Screen coordinates (x, y).

**Details:**
- Uses perspective projection formula: `scale = distance / (distance + z)`
- X coordinate is centered at `SCREEN_WIDTH / 2`
- Y coordinate is centered at `SCREEN_HEIGHT / 2` (inverted for screen coordinates)
- Objects farther away appear smaller
- Objects closer appear larger

**Example:**
```python
v = Vector3(100, 100, 300)
screen_pos = v.project_2d()  # Returns (550, 350) approximately
```

---

#### `__repr__(self) -> str`

Returns a string representation of the vector.

**Example:**
```python
v = Vector3(1.5, 2.3, 3.7)
print(v)  # Vector3(1.50, 2.30, 3.70)
```

---

## Quaternion Class

Complete quaternion mathematics implementation for 3D rotations.

### Constructor

#### `__init__(self, w: float = 1, x: float = 0, y: float = 0, z: float = 0)`

Initializes a Quaternion with components (w, x, y, z).

**Parameters:**
- `w` (float, optional): Scalar (real) part. Defaults to 1.
- `x` (float, optional): X component of vector part. Defaults to 0.
- `y` (float, optional): Y component of vector part. Defaults to 0.
- `z` (float, optional): Z component of vector part. Defaults to 0.

**Details:**
- Default values create the identity quaternion (1, 0, 0, 0)
- Represents "no rotation"

**Example:**
```python
identity = Quaternion()  # (1, 0, 0, 0)
custom_quat = Quaternion(0.7071, 0.7071, 0, 0)
```

---

### Methods

#### `__mul__(self, other: Quaternion) -> Quaternion`

Multiplies two quaternions (quaternion multiplication).

**Parameters:**
- `other` (Quaternion): The quaternion to multiply with.

**Returns:**
- (Quaternion): The product quaternion.

**Important Properties:**
- **Non-commutative**: q1 * q2 ≠ q2 * q1
- **Associative**: (q1 * q2) * q3 = q1 * (q2 * q3)
- Used to compose rotations

**Formula:**
```
q1 * q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
           w1*x2 + x1*w2 + y1*z2 - z1*y2,
           w1*y2 - x1*z2 + y1*w2 + z1*x2,
           w1*z2 + x1*y2 - y1*x2 + z1*w2)
```

**Example:**
```python
# Rotation around X axis by 90 degrees
q1 = Quaternion.from_axis_angle(Vector3(1, 0, 0), math.pi / 2)
# Rotation around Y axis by 45 degrees
q2 = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 4)
# Combined rotation
combined = q1 * q2
```

---

#### `magnitude(self) -> float`

Calculates the magnitude (norm) of the quaternion.

**Returns:**
- (float): The magnitude of the quaternion.

**Formula:**
```
|q| = √(w² + x² + y² + z²)
```

**Details:**
- Unit quaternions (used for rotations) have magnitude 1.
- Non-unit quaternions do not represent valid rotations.

**Example:**
```python
identity = Quaternion()
mag = identity.magnitude()  # 1.0

q = Quaternion(2, 0, 0, 0)
mag = q.magnitude()  # 2.0
```

---

#### `normalize(self) -> Quaternion`

Returns a normalized unit quaternion.

**Returns:**
- (Quaternion): A new Quaternion with magnitude 1.

**Details:**
- Essential for maintaining rotation validity.
- Returns identity quaternion if magnitude is 0.

**Example:**
```python
q = Quaternion(2, 0, 0, 0)
normalized = q.normalize()  # Quaternion(1, 0, 0, 0)
```

---

#### `conjugate(self) -> Quaternion`

Returns the conjugate of the quaternion.

**Returns:**
- (Quaternion): The conjugate quaternion with negated vector part.

**Formula:**
```
q* = w - xi - yj - zk = (w, -x, -y, -z)
```

**Properties:**
- For unit quaternions: q* = q⁻¹ (conjugate equals inverse)
- Used in rotation of vectors: q * v * q*

**Example:**
```python
q = Quaternion(0.7071, 0.7071, 0, 0)
q_conj = q.conjugate()  # Quaternion(0.7071, -0.7071, 0, 0)
```

---

#### `inverse(self) -> Quaternion`

Returns the inverse (reciprocal) of the quaternion.

**Returns:**
- (Quaternion): The inverse quaternion q⁻¹.

**Formula:**
```
q⁻¹ = q* / |q|² = (w, -x, -y, -z) / (w² + x² + y² + z²)
```

**Properties:**
- For unit quaternions: q⁻¹ = q*
- q * q⁻¹ = (1, 0, 0, 0) (identity quaternion)
- Used to reverse a rotation

**Example:**
```python
q = Quaternion.from_axis_angle(Vector3(1, 0, 0), math.pi / 2)
q_inv = q.inverse()  # Opposite rotation

# Reverse the rotation
rotated_back = q_inv.rotate_vector(rotated_vector)
```

---

#### `from_axis_angle(axis: Vector3, angle: float) -> Quaternion` (Static Method)

Creates a quaternion representing a rotation around an axis by a given angle.

**Parameters:**
- `axis` (Vector3): The rotation axis (will be normalized internally).
- `angle` (float): The rotation angle in radians.

**Returns:**
- (Quaternion): A unit quaternion representing the rotation.

**Formula:**
```
q = (cos(θ/2), sin(θ/2)*axis.x, sin(θ/2)*axis.y, sin(θ/2)*axis.z)
where θ is the angle and axis is normalized
```

**Details:**
- The most intuitive way to create rotation quaternions
- Axis is automatically normalized
- Angle is typically in radians (use math.radians() to convert from degrees)

**Example:**
```python
# 90-degree rotation around Z axis
angle = math.pi / 2  # 90 degrees
axis = Vector3(0, 0, 1)  # Z axis
q = Quaternion.from_axis_angle(axis, angle)

# 45-degree rotation around Y axis
q2 = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 4)
```

---

#### `rotate_vector(self, v: Vector3) -> Vector3`

Rotates a vector using this quaternion.

**Parameters:**
- `v` (Vector3): The vector to rotate.

**Returns:**
- (Vector3): The rotated vector.

**Formula:**
```
v' = q * v * q⁻¹
where v is treated as a quaternion (0, v.x, v.y, v.z)
```

**Details:**
- This is the standard way to rotate vectors with quaternions
- Works correctly only if the quaternion is normalized
- Preserves vector magnitude

**Example:**
```python
# Create a 90-degree rotation around Z axis
q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)

# Rotate a vector
v = Vector3(1, 0, 0)
rotated = q.rotate_vector(v)  # Approximately (0, 1, 0)

# Rotate it again
rotated_again = q.rotate_vector(rotated)  # Approximately (-1, 0, 0)
```

---

#### `to_euler_angles(self) -> Tuple[float, float, float]`

Converts the quaternion to Euler angles (roll, pitch, yaw).

**Returns:**
- (Tuple[float, float, float]): A tuple of (roll, pitch, yaw) in radians.

**Details:**
- **Roll**: Rotation around X axis
- **Pitch**: Rotation around Y axis
- **Yaw**: Rotation around Z axis
- Angles are in radians (use math.degrees() to convert to degrees)
- Useful for displaying rotation state to users

**Example:**
```python
q = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 4)
roll, pitch, yaw = q.to_euler_angles()

# Convert to degrees for display
roll_deg = math.degrees(roll)
pitch_deg = math.degrees(pitch)
yaw_deg = math.degrees(yaw)

print(f"Rotation: Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°, Yaw={yaw_deg:.1f}°")
```

---

#### `slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion` (Static Method)

Performs spherical linear interpolation (SLERP) between two quaternions.

**Parameters:**
- `q1` (Quaternion): The starting quaternion.
- `q2` (Quaternion): The ending quaternion.
- `t` (float): Interpolation parameter in range [0, 1].
  - t=0 returns q1
  - t=1 returns q2
  - 0 < t < 1 returns interpolated quaternion

**Returns:**
- (Quaternion): The interpolated unit quaternion.

**Details:**
- Provides smooth rotation interpolation along the shortest path
- Handles the shorter path automatically
- Falls back to linear interpolation if quaternions are very close
- Essential for smooth animation transitions

**Formula:**
```
SLERP(q1, q2, t) = (sin((1-t)θ)/sin(θ)) * q1 + (sin(tθ)/sin(θ)) * q2
where θ is the angle between q1 and q2
```

**Example:**
```python
# Starting and ending rotations
q_start = Quaternion(1, 0, 0, 0)  # Identity (no rotation)
q_end = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 2)  # 90° around Y

# Interpolate between them
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    q_interpolated = Quaternion.slerp(q_start, q_end, t)
    # Use q_interpolated for smooth animation
```

**Use Cases:**
- Animation keyframing
- Smooth camera rotations
- Character animation
- Transition between different orientations

---

#### `__repr__(self) -> str`

Returns a string representation of the quaternion.

**Example:**
```python
q = Quaternion(0.7071, 0.7071, 0, 0)
print(q)  # Quaternion(w=0.71, x=0.71, y=0.00, z=0.00)
```

---

## QuaternionCube Class

A 3D cube object that uses quaternion-based rotations for smooth, gimbal-lock-free animation.

### Constructor

#### `__init__(self, size: float = 100)`

Initializes a QuaternionCube with a specified size.

**Parameters:**
- `size` (float, optional): The side length of the cube. Defaults to 100.

**Attributes:**
- `size` (float): The side length of the cube.
- `rotation` (Quaternion): Current rotation as a unit quaternion (identity by default).
- `position` (Vector3): The center position of the cube in 3D space.
- `angular_velocity` (Quaternion): Quaternion representing rotation per frame (not directly used).
- `vertices` (List[Vector3]): List of 8 vertices defining the cube corners.
- `edges` (List[Tuple[int, int]]): List of 12 edges as vertex index pairs.
- `quat_x`, `quat_y`, `quat_z` (Quaternion): Pre-computed rotation quaternions for smooth animation.

**Example:**
```python
small_cube = QuaternionCube(size=50)
large_cube = QuaternionCube(size=200)
default_cube = QuaternionCube()  # 100x100x100
```

---

### Methods

#### `_setup_angular_velocities(self)` (Private Method)

Sets up the pre-computed quaternions for rotation axes and speeds.

**Details:**
- Called automatically during initialization
- Creates quaternions for rotation around X, Y, and Z axes
- Rotation speeds:
  - X axis: 0.02 radians per frame
  - Y axis: 0.03 radians per frame (1.5x faster)
  - Z axis: 0.01 radians per frame (0.5x faster)

---

#### `update(self, delta_time: float)`

Updates the cube's rotation for one frame using quaternion multiplication.

**Parameters:**
- `delta_time` (float): Time elapsed since the last update (in seconds, though not currently used).

**Details:**
- Applies rotations around all three axes by multiplying quaternions
- Normalizes the resulting quaternion to prevent numerical drift
- Quaternion multiplication automatically combines rotations
- The order of multiplication determines the order of rotations applied

**Example:**
```python
cube = QuaternionCube()
delta_time = 0.016  # ~60 FPS
cube.update(delta_time)
```

---

#### `get_transformed_vertices(self) -> List[Vector3]`

Calculates the cube's vertices after applying quaternion rotation and position.

**Returns:**
- (List[Vector3]): A list of 8 Vector3 objects representing the transformed vertices.

**Details:**
- Uses quaternion rotation: v' = q.rotate_vector(v)
- Applies position offset to all vertices
- Does not modify the original vertices
- Called internally by the rendering system

**Example:**
```python
cube = QuaternionCube()
cube.rotation = Quaternion.from_axis_angle(Vector3(1, 1, 0), math.pi / 4)
cube.position = Vector3(10, 20, 300)
transformed = cube.get_transformed_vertices()
```

---

#### `draw(self, surface: pygame.Surface)`

Renders the cube onto a Pygame surface.

**Parameters:**
- `surface` (pygame.Surface): The Pygame surface to draw on (typically the main screen).

**Details:**
- Gets transformed vertices
- Projects each 3D vertex to 2D screen coordinates
- Draws edges as white lines connecting vertices (2 pixels wide)
- Draws vertices as small green circles (4 pixels radius)
- Uses `EDGE_COLOR` and `CUBE_COLOR` constants

**Example:**
```python
cube = QuaternionCube()
pygame.display.set_mode((800, 600))
screen = pygame.display.get_surface()
cube.draw(screen)
```

---

## QuaternionGameEngine Class

The main game engine class that manages the game loop, events, updates, and rendering.

### Constructor

#### `__init__(self)`

Initializes the QuaternionGameEngine with Pygame setup and cube creation.

**Attributes:**
- `screen` (pygame.Surface): The main display surface (800x600 pixels).
- `clock` (pygame.time.Clock): Pygame clock for managing frame rate and FPS.
- `running` (bool): Flag indicating whether the game loop should continue.
- `cube` (QuaternionCube): The cube object being displayed and animated.

**Details:**
- Sets window title: "Quaternion-Based 3D Game Engine - Rotating Cube"
- Creates a cube with size 150
- Positions cube at (0, 0, 300) - 300 units away from camera for proper perspective
- Sets target FPS to 60

**Example:**
```python
engine = QuaternionGameEngine()
```

---

### Methods

#### `handle_events(self)`

Processes user input and window events.

**Details:**
- **QUIT event**: Window close button → sets `running` to False
- **ESC key**: Exits the application gracefully
- **SPACE key**: Resets cube rotation to identity quaternion

**Example:**
```python
engine = QuaternionGameEngine()
engine.handle_events()
```

---

#### `update(self, delta_time: float)`

Updates the game state for one frame.

**Parameters:**
- `delta_time` (float): Time elapsed since the last update (in seconds).

**Details:**
- Calls `cube.update(delta_time)` to advance animation
- Can be extended for additional game logic

**Example:**
```python
engine = QuaternionGameEngine()
delta_time = 0.016  # ~60 FPS
engine.update(delta_time)
```

---

#### `render(self)`

Renders the current frame to the screen.

**Details:**
- Clears screen with `BACKGROUND_COLOR` (dark blue)
- Draws the cube
- Renders information text layer:
  - Engine description and controls
  - Current FPS counter
  - Euler angle representation of current rotation (in degrees)
- Updates display using `pygame.display.flip()`

**Example:**
```python
engine = QuaternionGameEngine()
engine.render()
```

---

#### `run(self)`

The main game loop that orchestrates the game flow.

**Game Loop Flow:**
1. Calculate frame time using clock (capped at 60 FPS)
2. Handle user input and events
3. Update game state
4. Render frame
5. Repeat until `running` is False

**Details:**
- Runs continuously while `self.running` is True
- Frame rate is limited to 60 FPS via `clock.tick(FPS)`
- Calculates `delta_time` by converting milliseconds to seconds
- Gracefully shuts down Pygame when loop exits

**Example:**
```python
if __name__ == "__main__":
    engine = QuaternionGameEngine()
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
| `BACKGROUND_COLOR` | tuple | (20, 20, 40) | RGB color for background (dark blue). |
| `CUBE_COLOR` | tuple | (0, 255, 100) | RGB color for cube vertices (bright green). |
| `EDGE_COLOR` | tuple | (255, 255, 255) | RGB color for cube edges (white). |

---

## Mathematical Concepts

### Gimbal Lock Problem

**What is Gimbal Lock?**

Gimbal lock occurs when using Euler angles (roll, pitch, yaw) to represent rotations. When the pitch angle reaches ±90°, the roll and yaw axes become aligned, causing a loss of one degree of freedom.

**How Quaternions Solve It:**

Quaternions represent rotations as a single rotation around an arbitrary axis, avoiding the alignment issue that causes gimbal lock. This makes them ideal for 3D graphics and animation.

### Quaternion Rotation Formula

To rotate a vector **v** using quaternion **q**:

```
v' = q * v * q⁻¹
```

Where:
- **q** is the rotation quaternion (unit quaternion)
- **v** is treated as quaternion (0, v.x, v.y, v.z)
- **q⁻¹** is the inverse of q

For unit quaternions: **q⁻¹ = q*** (inverse equals conjugate)

### Quaternion Multiplication

Quaternion multiplication is non-commutative. For q₁ = (w₁, x₁, y₁, z₁) and q₂ = (w₂, x₂, y₂, z₂):

```
q₁ * q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂,
           w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂,
           w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂,
           w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)
```

This formula represents the composition of two rotations.

### Spherical Linear Interpolation (SLERP)

SLERP smoothly interpolates between two rotations along the shortest path on the unit hypersphere:

```
SLERP(q₁, q₂, t) = (sin((1-t)θ) / sin(θ)) * q₁ + (sin(tθ) / sin(θ)) * q₂
```

Where θ is the angle between the quaternions.

**Properties:**
- Constant angular velocity throughout interpolation
- Smooth rotation without acceleration artifacts
- Essential for animation and camera control

---

## Usage Examples

### Example 1: Basic Rotation

```python
from Quaternions import Vector3, Quaternion
import math

# Create a rotation of 45 degrees around the Y axis
axis = Vector3(0, 1, 0)
angle = math.pi / 4  # 45 degrees in radians
q = Quaternion.from_axis_angle(axis, angle)

# Rotate a vector
v = Vector3(1, 0, 0)
rotated = q.rotate_vector(v)
print(f"Rotated vector: {rotated}")
```

### Example 2: Combining Rotations

```python
from Quaternions import Vector3, Quaternion
import math

# Create two rotations
q1 = Quaternion.from_axis_angle(Vector3(1, 0, 0), math.pi / 2)  # 90° around X
q2 = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 2)  # 90° around Y

# Combine them
combined = q1 * q2

# Rotate a vector with the combined rotation
v = Vector3(1, 0, 0)
result = combined.rotate_vector(v)
print(f"Result: {result}")
```

### Example 3: Smooth Animation with SLERP

```python
from Quaternions import Quaternion, Vector3
import math

# Starting and ending orientations
q_start = Quaternion(1, 0, 0, 0)  # Identity
q_end = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi)  # 180° around Y

# Interpolate over 10 frames
for frame in range(11):
    t = frame / 10.0  # 0 to 1
    q_current = Quaternion.slerp(q_start, q_end, t)
    print(f"Frame {frame}: {q_current}")
```

### Example 4: Running the Game Engine

```python
if __name__ == "__main__":
    from Quaternions import QuaternionGameEngine
    
    engine = QuaternionGameEngine()
    engine.run()
```

### Example 5: Creating a Custom Rotating Cube

```python
from Quaternions import QuaternionCube, Vector3, Quaternion
import math

# Create a cube
cube = QuaternionCube(size=100)

# Set its position
cube.position = Vector3(0, 0, 500)

# Apply an initial rotation
initial_rotation = Quaternion.from_axis_angle(Vector3(1, 1, 1), math.pi / 6)
cube.rotation = initial_rotation

# Update it each frame
for frame in range(60):
    cube.update(0.016)  # 60 FPS
    # Render cube here
```

---

## Best Practices

### 1. Always Normalize Quaternions

After multiple operations, numerical errors can accumulate. Normalize regularly:

```python
q = q * some_rotation
q = q.normalize()
```

### 2. Use Unit Quaternions for Rotations

Ensure quaternions used for rotations are unit quaternions (magnitude = 1):

```python
q = Quaternion.from_axis_angle(axis, angle)  # Already normalized
q = q.normalize()  # Extra safety
```

### 3. Use SLERP for Animation

For smooth transitions between rotations, always use SLERP:

```python
q_current = Quaternion.slerp(q_start, q_end, t)  # Smooth and correct
```

### 4. Normalize Axes Before Creating Rotations

The `from_axis_angle` method normalizes automatically, but it's good practice:

```python
axis = Vector3(1, 1, 0)
axis = axis.normalize()
q = Quaternion.from_axis_angle(axis, angle)
```

### 5. Avoid Euler Angles in Real-time

Only convert to Euler angles for display purposes:

```python
# For internal calculations: use quaternions
q = q * rotation_delta
q = q.normalize()

# For display only:
roll, pitch, yaw = q.to_euler_angles()
print(f"Display: {math.degrees(roll)}°, {math.degrees(pitch)}°, {math.degrees(yaw)}°")
```

---

## Performance Considerations

### Memory Usage

- **Quaternion**: 4 floats (16 bytes on 32-bit systems)
- **Rotation Matrix**: 9 floats (36 bytes on 32-bit systems)
- **Euler Angles**: 3 floats (12 bytes on 32-bit systems)

Quaternions are a middle ground between efficiency and stability.

### Computation Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Quaternion Multiplication | O(1) | 16 multiplications, 12 additions |
| Vector Rotation | O(1) | Efficient compared to matrix multiplication |
| SLERP | O(1) | Slightly more expensive than linear interpolation |
| Normalization | O(1) | Necessary for numerical stability |
| Euler Conversion | O(1) | Use sparingly (mainly for display) |

---

## Troubleshooting

### Issue: Cube Spinning Erratically

**Cause**: Quaternion not normalized.

**Solution**:
```python
cube.rotation = cube.rotation.normalize()
```

### Issue: Rotation Going the "Long Way"

**Cause**: SLERP path not optimized.

**Solution**: The implementation already handles this by checking dot product sign.

### Issue: Animation Not Smooth

**Cause**: Insufficient frame rate or not using SLERP.

**Solution**:
```python
# Use SLERP for smooth interpolation
q_current = Quaternion.slerp(q_start, q_end, t)
```

---

## Future Enhancements

Possible improvements to the quaternion engine:

1. **Inverse Kinematics**: Use quaternions for IK solving
2. **Quaternion Splines**: Implement squad (spherical quadrangle) for smoother curves
3. **Motion Capture Integration**: Support for mocap data using quaternions
4. **Physics Simulation**: Use quaternions for rigid body dynamics
5. **Pose Blending**: Smooth blending between multiple poses
6. **Skeletal Animation**: Full skeletal animation support
7. **GPU Acceleration**: Move quaternion math to GPU shaders
8. **Optimization**: Cache frequently used values

---

## References

### Mathematical Resources

- **Unit Quaternions**: Understanding the basics
- **Rotation Matrices vs Quaternions**: Comparative analysis
- **SLERP Algorithm**: Spherical linear interpolation theory
- **Gimbal Lock**: Visual explanation and solutions

### Related Concepts

- 3D Graphics Transformations
- Euler Angles and Their Limitations
- Rotation Matrices
- Axis-Angle Representation
- Rotation Composition

---

## License

This quaternion-based game engine is provided as-is for educational purposes.

## Author Notes

This implementation prioritizes clarity and correctness over extreme performance optimization. It serves as an excellent reference for understanding quaternion mathematics and their application to 3D graphics programming.

For production systems, consider specialized quaternion libraries like:
- PyQuaternion
- Numpy with quaternion extensions
- Specialized game engines (Unity, Unreal Engine)
