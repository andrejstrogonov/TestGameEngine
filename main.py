import math
from typing import Tuple

import numpy as np
import pygame

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available, falling back to CPU (NumPy)")
    cp = np  # Fallback to NumPy if CuPy is not available

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
BACKGROUND_COLOR = (20, 20, 40)
PLANE_COLOR = (0, 255, 100)
EDGE_COLOR = (255, 255, 255)
AIRPLANE_MODEL_PATH = "airplane/11805_airplane_v2_L2.obj"
AIRPLANE_MTL_PATH = "airplane/11805_airplane_v2_L2.mtl"
AIRPLANE_TEXTURES_DIR = "airplane/"

# Zoom constants
MIN_ZOOM = 100
MAX_ZOOM = 2000
ZOOM_SPEED = 50


class Vector3:
    """3D Vector class for basic 3D math"""
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other: 'Vector3') -> float:
        """Calculate dot product"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        """Calculate cross product"""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        """Calculate vector magnitude (length)"""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def normalize(self) -> 'Vector3':
        """Return normalized vector (unit vector)"""
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return self / mag
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    @staticmethod
    def from_numpy(arr) -> 'Vector3':
        """Create from numpy/cupy array"""
        if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def project_2d(self, distance: float = 500) -> Tuple[float, float]:
        """Project 3D point to 2D screen coordinates"""
        scale = distance / (distance + self.z)
        x = self.x * scale + SCREEN_WIDTH / 2
        y = -self.y * scale + SCREEN_HEIGHT / 2
        return (x, y)
    
    def __repr__(self) -> str:
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Quaternion:
    """Quaternion class for 3D rotations using quaternion mathematics"""
    def __init__(self, w: float = 1, x: float = 0, y: float = 0, z: float = 0):
        """
        Initialize a quaternion with components (w, x, y, z).
        w is the scalar part, (x, y, z) is the vector part.
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Multiply two quaternions (quaternion multiplication)"""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)
    
    def magnitude(self) -> float:
        """Calculate quaternion magnitude"""
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def normalize(self) -> 'Quaternion':
        """Return normalized quaternion (unit quaternion)"""
        mag = self.magnitude()
        if mag == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(
            self.w / mag,
            self.x / mag,
            self.y / mag,
            self.z / mag
        )
    
    def conjugate(self) -> 'Quaternion':
        """Return the conjugate of the quaternion"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        """Return the inverse of the quaternion"""
        mag_sq = self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2
        if mag_sq == 0:
            return Quaternion(1, 0, 0, 0)
        conj = self.conjugate()
        return Quaternion(
            conj.w / mag_sq,
            conj.x / mag_sq,
            conj.y / mag_sq,
            conj.z / mag_sq
        )
    
    @staticmethod
    def from_axis_angle(axis: Vector3, angle: float) -> 'Quaternion':
        """
        Create a quaternion from an axis and angle.
        
        Args:
            axis: The rotation axis (should be normalized)
            angle: The rotation angle in radians
        
        Returns:
            A quaternion representing the rotation
        """
        axis = axis.normalize()
        half_angle = angle / 2
        sin_half = math.sin(half_angle)
        
        return Quaternion(
            math.cos(half_angle),
            axis.x * sin_half,
            axis.y * sin_half,
            axis.z * sin_half
        )
    
    def rotate_vector(self, v: Vector3) -> Vector3:
        """
        Rotate a vector using this quaternion.
        
        Args:
            v: The vector to rotate
        
        Returns:
            The rotated vector
        """
        # Convert vector to quaternion (w=0)
        v_quat = Quaternion(0, v.x, v.y, v.z)
        
        # Perform rotation: q * v * q^-1
        result = self * v_quat * self.inverse()
        
        return Vector3(result.x, result.y, result.z)
    
    def to_euler_angles(self) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
        
        Returns:
            Tuple of (roll, pitch, yaw) angles in radians
        """
        # Roll (rotation around X-axis)
        sin_roll = 2 * (self.w * self.x + self.y * self.z)
        cos_roll = 1 - 2 * (self.x ** 2 + self.y ** 2)
        roll = math.atan2(sin_roll, cos_roll)
        
        # Pitch (rotation around Y-axis)
        sin_pitch = 2 * (self.w * self.y - self.z * self.x)
        sin_pitch = max(-1, min(1, sin_pitch))  # Clamp to [-1, 1]
        pitch = math.asin(sin_pitch)
        
        # Yaw (rotation around Z-axis)
        sin_yaw = 2 * (self.w * self.z + self.x * self.y)
        cos_yaw = 1 - 2 * (self.y ** 2 + self.z ** 2)
        yaw = math.atan2(sin_yaw, cos_yaw)
        
        return (roll, pitch, yaw)
    
    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.2f}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class GPUQuaternionRotator:
    """GPU-accelerated quaternion rotation using CuPy"""
    def __init__(self, quaternion: Quaternion):
        self.q = quaternion
        self.q_conj = quaternion.conjugate()
        self.q_inv = quaternion.inverse()
    
    def rotate_batch_gpu(self, vertices_gpu) -> 'cp.ndarray':
        """
        Rotate a batch of vertices on GPU using quaternion.
        
        Args:
            vertices_gpu: GPU array of shape (N, 3) containing vertices
        
        Returns:
            GPU array of rotated vertices
        """
        # Extract quaternion components
        w, x, y, z = self.q.w, self.q.x, self.q.y, self.q.z
        w_inv, x_inv, y_inv, z_inv = self.q_inv.w, self.q_inv.x, self.q_inv.y, self.q_inv.z
        
        # Extract vertex components
        vx = vertices_gpu[:, 0]
        vy = vertices_gpu[:, 1]
        vz = vertices_gpu[:, 2]
        
        # First multiplication: q * v (where v is treated as quaternion with w=0)
        # q * v = (w*vx + x*0 - y*vz + z*vy, x*vx + w*vy + y*0 - z*vz, ...)
        # Simplified: q * v
        qv_w = -x * vx - y * vy - z * vz
        qv_x = w * vx + y * vz - z * vy
        qv_y = w * vy + z * vx - x * vz
        qv_z = w * vz + x * vy - y * vx
        
        # Second multiplication: (q*v) * q^-1
        # Result components
        rx = qv_w * x_inv + qv_x * w_inv + qv_y * z_inv - qv_z * y_inv
        ry = qv_w * y_inv - qv_x * z_inv + qv_y * w_inv + qv_z * x_inv
        rz = qv_w * z_inv + qv_x * y_inv - qv_y * x_inv + qv_z * w_inv
        
        # Stack results
        rotated = cp.stack([rx, ry, rz], axis=1)
        return rotated


class Airplane:
    """3D Airplane model with GPU acceleration using CuPy"""
    def __init__(self, obj_path: str):
        self.rotation = Quaternion(1, 0, 0, 0)  # Identity quaternion
        self.position = Vector3(0, 0, 0)
        self.scale = 1.0
        
        # Load OBJ model
        self.vertices = []
        self.vertices_np = None  # NumPy array
        self.vertices_gpu = None  # GPU array (CuPy)
        self.faces = []
        self.load_obj(obj_path)
        
        # Calculate bounding box for scaling
        self.calculate_bounds()
        
        # Cache for transformed vertices
        self.transformed_vertices_cache = None
        self.last_rotation = None
        self.last_position = None
        
        # GPU rotator
        self.gpu_rotator = None
        
        # Performance tracking
        self.use_gpu = GPU_AVAILABLE and self.vertices_gpu is not None
    
    def load_obj(self, obj_path: str):
        """Load vertices and faces from OBJ file"""
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    # Parse vertex positions
                    if parts[0] == 'v':
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        self.vertices.append(Vector3(x, y, z))
                    
                    # Parse faces (triangles or quads)
                    elif parts[0] == 'f':
                        face = []
                        for i in range(1, len(parts)):
                            # Handle vertex indices (ignore texture and normal indices)
                            vertex_idx = int(parts[i].split('/')[0]) - 1  # OBJ indices are 1-based
                            face.append(vertex_idx)
                        
                        # Convert quads to triangles
                        if len(face) == 4:
                            self.faces.append((face[0], face[1], face[2]))
                            self.faces.append((face[0], face[2], face[3]))
                        elif len(face) == 3:
                            self.faces.append(tuple(face))
            
            # Convert vertices to NumPy tensor
            if self.vertices:
                self.vertices_np = np.array([[v.x, v.y, v.z] for v in self.vertices], dtype=np.float32)
                
                # Transfer to GPU if available
                if GPU_AVAILABLE:
                    self.vertices_gpu = cp.asarray(self.vertices_np)
            
            print(f"Loaded airplane model: {len(self.vertices)} vertices, {len(self.faces)} faces")
            if GPU_AVAILABLE:
                print(f"GPU acceleration: ENABLED")
            else:
                print(f"GPU acceleration: DISABLED (CuPy not available)")
        except FileNotFoundError:
            print(f"Error: Could not find OBJ file at {obj_path}")
            # Create a simple fallback cube
            self.create_fallback_model()
    
    def create_fallback_model(self):
        """Create a simple cube as fallback if OBJ loading fails"""
        size = 50
        self.vertices = [
            Vector3(-size, -size, -size),
            Vector3(size, -size, -size),
            Vector3(size, size, -size),
            Vector3(-size, size, -size),
            Vector3(-size, -size, size),
            Vector3(size, -size, size),
            Vector3(size, size, size),
            Vector3(-size, size, size),
        ]
        
        self.faces = [
            # Front
            (0, 1, 2), (0, 2, 3),
            # Back
            (4, 6, 5), (4, 7, 6),
            # Left
            (0, 4, 7), (0, 7, 3),
            # Right
            (1, 5, 6), (1, 6, 2),
            # Top
            (3, 7, 6), (3, 6, 2),
            # Bottom
            (0, 5, 1), (0, 4, 5),
        ]
        
        # Convert to NumPy tensor
        self.vertices_np = np.array([[v.x, v.y, v.z] for v in self.vertices], dtype=np.float32)
        
        # Transfer to GPU if available
        if GPU_AVAILABLE:
            self.vertices_gpu = cp.asarray(self.vertices_np)
    
    def calculate_bounds(self):
        """Calculate bounding box and scale model appropriately"""
        if self.vertices_np is None or len(self.vertices_np) == 0:
            return
        
        # Use NumPy for efficient min/max calculations
        min_coords = np.min(self.vertices_np, axis=0)
        max_coords = np.max(self.vertices_np, axis=0)
        
        # Calculate dimensions
        dimensions = max_coords - min_coords
        max_dim = np.max(dimensions)
        
        # Scale to fit in a reasonable size (200 units)
        if max_dim > 0:
            self.scale = 200.0 / max_dim
        
        # Center the model
        self.center_offset = Vector3(
            float((min_coords[0] + max_coords[0]) / 2),
            float((min_coords[1] + max_coords[1]) / 2),
            float((min_coords[2] + max_coords[2]) / 2)
        )
    
    def update(self, mouse_delta: Tuple[float, float]):
        """Update airplane rotation based on mouse movement using quaternions"""
        sensitivity = 0.005
        
        if mouse_delta[0] != 0:
            # Rotation around Y axis (horizontal mouse movement)
            angle_y = mouse_delta[0] * sensitivity
            axis_y = Vector3(0, 1, 0)
            quat_y = Quaternion.from_axis_angle(axis_y, angle_y)
            self.rotation = quat_y * self.rotation
        
        if mouse_delta[1] != 0:
            # Rotation around X axis (vertical mouse movement)
            angle_x = mouse_delta[1] * sensitivity
            axis_x = Vector3(1, 0, 0)
            quat_x = Quaternion.from_axis_angle(axis_x, angle_x)
            self.rotation = quat_x * self.rotation
        
        # Normalize to prevent numerical drift
        self.rotation = self.rotation.normalize()
        
        # Update GPU rotator
        if self.use_gpu:
            self.gpu_rotator = GPUQuaternionRotator(self.rotation)
        
        # Invalidate cache
        self.transformed_vertices_cache = None
    
    def get_transformed_vertices_gpu(self) -> np.ndarray:
        """Get vertices after applying transformations using GPU acceleration"""
        if self.vertices_gpu is None:
            return np.array([], dtype=np.float32)
        
        # Check cache validity
        if (self.transformed_vertices_cache is not None and 
            self.last_rotation == (self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z) and
            self.last_position == (self.position.x, self.position.y, self.position.z)):
            return self.transformed_vertices_cache
        
        # Center the model on GPU
        center_offset_gpu = cp.asarray(self.center_offset.to_numpy())
        centered_gpu = self.vertices_gpu - center_offset_gpu
        
        # Apply scale on GPU
        scaled_gpu = centered_gpu * self.scale
        
        # Apply rotation on GPU using batch operation
        if self.gpu_rotator is None:
            self.gpu_rotator = GPUQuaternionRotator(self.rotation)
        
        rotated_gpu = self.gpu_rotator.rotate_batch_gpu(scaled_gpu)
        
        # Apply position on GPU
        position_gpu = cp.asarray(self.position.to_numpy())
        transformed_gpu = rotated_gpu + position_gpu
        
        # Transfer back to CPU for rendering
        transformed = cp.asnumpy(transformed_gpu)
        
        # Cache the result
        self.transformed_vertices_cache = transformed
        self.last_rotation = (self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z)
        self.last_position = (self.position.x, self.position.y, self.position.z)
        
        return transformed
    
    def get_transformed_vertices_cpu(self) -> np.ndarray:
        """Get vertices after applying transformations using CPU (fallback)"""
        if self.vertices_np is None:
            return np.array([], dtype=np.float32)
        
        # Check cache validity
        if (self.transformed_vertices_cache is not None and 
            self.last_rotation == (self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z) and
            self.last_position == (self.position.x, self.position.y, self.position.z)):
            return self.transformed_vertices_cache
        
        # Center the model using NumPy
        center_offset_np = self.center_offset.to_numpy()
        centered = self.vertices_np - center_offset_np
        
        # Apply scale
        scaled = centered * self.scale
        
        # Apply rotation using quaternion (convert to Vector3 for rotation)
        rotated = np.array([
            self.rotation.rotate_vector(Vector3(v[0], v[1], v[2])).to_numpy()
            for v in scaled
        ], dtype=np.float32)
        
        # Apply position
        position_np = self.position.to_numpy()
        transformed = rotated + position_np
        
        # Cache the result
        self.transformed_vertices_cache = transformed
        self.last_rotation = (self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z)
        self.last_position = (self.position.x, self.position.y, self.position.z)
        
        return transformed
    
    def get_transformed_vertices(self) -> np.ndarray:
        """Get transformed vertices using GPU if available, otherwise CPU"""
        if self.use_gpu:
            return self.get_transformed_vertices_gpu()
        else:
            return self.get_transformed_vertices_cpu()
    
    def draw(self, surface: pygame.Surface):
        """Draw the airplane model"""
        transformed_vertices_np = self.get_transformed_vertices()
        
        if len(transformed_vertices_np) == 0:
            return
        
        # Project vertices to 2D
        projected_vertices = []
        for v in transformed_vertices_np:
            vec3 = Vector3(float(v[0]), float(v[1]), float(v[2]))
            projected_vertices.append(vec3.project_2d())
        
        # Draw faces as wireframe
        for face in self.faces:
            if all(0 <= idx < len(projected_vertices) for idx in face):
                points = [projected_vertices[idx] for idx in face]
                
                # Draw triangle edges
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    pygame.draw.line(surface, EDGE_COLOR, start, end, 1)
                
                # Draw filled triangle with semi-transparent color
                try:
                    pygame.draw.polygon(surface, PLANE_COLOR, points)
                except:
                    pass  # Skip if polygon is degenerate


class GameEngine:
    """3D game engine with GPU acceleration"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        gpu_status = "GPU Enabled" if GPU_AVAILABLE else "CPU Only"
        pygame.display.set_caption(f"3D Airplane Engine - {gpu_status} (Scroll to Zoom)")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Load airplane model
        self.airplane = Airplane(AIRPLANE_MODEL_PATH)
        self.zoom_distance = 500  # Initial zoom distance
        self.airplane.position = Vector3(0, 0, self.zoom_distance)
        
        # Mouse tracking
        self.prev_mouse_pos = pygame.mouse.get_pos()
        self.mouse_delta = (0, 0)
        
        # Performance metrics
        self.frame_times = []
        self.max_frame_samples = 60
    
    def handle_events(self):
        """Handle user input and window events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.MOUSEWHEEL:
                # Handle mouse wheel zoom
                if event.y > 0:  # Scroll up - zoom in
                    self.zoom_distance = max(MIN_ZOOM, self.zoom_distance - ZOOM_SPEED)
                elif event.y < 0:  # Scroll down - zoom out
                    self.zoom_distance = min(MAX_ZOOM, self.zoom_distance + ZOOM_SPEED)
                
                # Update airplane position
                self.airplane.position = Vector3(0, 0, self.zoom_distance)
    
    def update(self, delta_time: float):
        """Update game state"""
        # Get current mouse position
        current_mouse_pos = pygame.mouse.get_pos()
        
        # Calculate mouse delta
        self.mouse_delta = (
            current_mouse_pos[0] - self.prev_mouse_pos[0],
            current_mouse_pos[1] - self.prev_mouse_pos[1]
        )
        
        # Update airplane rotation based on mouse movement
        self.airplane.update(self.mouse_delta)
        
        # Update previous mouse position
        self.prev_mouse_pos = current_mouse_pos
    
    def render(self):
        """Render the scene"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw airplane
        self.airplane.draw(self.screen)
        
        # Draw info text
        font = pygame.font.Font(None, 22)
        
        # Title and controls
        info_text = font.render("3D Airplane Engine - Move mouse to rotate, Scroll to zoom", True, (200, 200, 200))
        self.screen.blit(info_text, (10, 10))
        
        # FPS
        fps_text = font.render(f"FPS: {int(self.clock.get_fps())}", True, (200, 200, 200))
        self.screen.blit(fps_text, (10, 35))
        
        # GPU Status
        gpu_status = "GPU: ENABLED" if self.airplane.use_gpu else "GPU: DISABLED (CPU)"
        gpu_color = (0, 255, 100) if self.airplane.use_gpu else (255, 165, 0)
        gpu_text = font.render(gpu_status, True, gpu_color)
        self.screen.blit(gpu_text, (10, 60))
        
        # Zoom distance
        zoom_text = font.render(f"Zoom Distance: {self.zoom_distance:.0f}", True, (200, 200, 200))
        self.screen.blit(zoom_text, (10, 85))
        
        # Quaternion info
        quat_text = font.render(
            f"Quaternion: w={self.airplane.rotation.w:.3f} x={self.airplane.rotation.x:.3f} y={self.airplane.rotation.y:.3f} z={self.airplane.rotation.z:.3f}",
            True, (200, 200, 200)
        )
        self.screen.blit(quat_text, (10, 110))
        
        # Euler angles
        euler = self.airplane.rotation.to_euler_angles()
        euler_text = font.render(
            f"Euler (deg): Roll={math.degrees(euler[0]):.1f} Pitch={math.degrees(euler[1]):.1f} Yaw={math.degrees(euler[2]):.1f}",
            True, (200, 200, 200)
        )
        self.screen.blit(euler_text, (10, 135))
        
        # Model info
        model_text = font.render(
            f"Model: {len(self.airplane.vertices)} vertices, {len(self.airplane.faces)} faces",
            True, (200, 200, 200)
        )
        self.screen.blit(model_text, (10, 160))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        while self.running:
            delta_time = self.clock.tick(FPS) / 1000.0  # Convert to seconds
            
            self.handle_events()
            self.update(delta_time)
            self.render()
        
        pygame.quit()


if __name__ == "__main__":
    engine = GameEngine()
    engine.run()
