import pygame
import math
from typing import List, Tuple

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
BACKGROUND_COLOR = (20, 20, 40)
CUBE_COLOR = (0, 255, 100)
EDGE_COLOR = (255, 255, 255)


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
    
    @staticmethod
    def slerp(q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """
        Spherical linear interpolation between two quaternions.
        
        Args:
            q1: First quaternion
            q2: Second quaternion
            t: Interpolation parameter (0 to 1)
        
        Returns:
            Interpolated quaternion
        """
        # Normalize quaternions
        q1 = q1.normalize()
        q2 = q2.normalize()
        
        # Calculate dot product
        dot_product = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        
        # If dot product is negative, negate one quaternion to take shorter path
        if dot_product < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot_product = -dot_product
        
        # Clamp dot product
        dot_product = max(-1, min(1, dot_product))
        
        # Calculate angle between quaternions
        theta = math.acos(dot_product)
        sin_theta = math.sin(theta)
        
        if sin_theta < 0.001:
            # Quaternions are very close, use linear interpolation
            w = q1.w + t * (q2.w - q1.w)
            x = q1.x + t * (q2.x - q1.x)
            y = q1.y + t * (q2.y - q1.y)
            z = q1.z + t * (q2.z - q1.z)
            return Quaternion(w, x, y, z).normalize()
        
        # Calculate interpolation coefficients
        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        
        # Interpolate
        return Quaternion(
            w1 * q1.w + w2 * q2.w,
            w1 * q1.x + w2 * q2.x,
            w1 * q1.y + w2 * q2.y,
            w1 * q1.z + w2 * q2.z
        )
    
    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.2f}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class QuaternionCube:
    """3D Cube object using quaternion-based rotations"""
    def __init__(self, size: float = 100):
        self.size = size
        self.rotation = Quaternion(1, 0, 0, 0)  # Identity quaternion
        self.position = Vector3(0, 0, 0)
        self.angular_velocity = Quaternion(1, 0, 0, 0)  # Rotation per frame
        
        # Define cube vertices (8 corners)
        half_size = size / 2
        self.vertices = [
            Vector3(-half_size, -half_size, -half_size),
            Vector3(half_size, -half_size, -half_size),
            Vector3(half_size, half_size, -half_size),
            Vector3(-half_size, half_size, -half_size),
            Vector3(-half_size, -half_size, half_size),
            Vector3(half_size, -half_size, half_size),
            Vector3(half_size, half_size, half_size),
            Vector3(-half_size, half_size, half_size),
        ]
        
        # Define cube edges (connections between vertices)
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
        ]
        
        # Set up angular velocities for smooth rotation
        self._setup_angular_velocities()
    
    def _setup_angular_velocities(self):
        """Setup rotation axes and speeds"""
        # Create rotation axes
        axis_x = Vector3(1, 0, 0)
        axis_y = Vector3(0, 1, 0)
        axis_z = Vector3(0, 0, 1)
        
        # Create quaternions for rotation around each axis
        # These represent small rotations per frame
        angle_per_frame = 0.02  # radians
        
        self.quat_x = Quaternion.from_axis_angle(axis_x, angle_per_frame)
        self.quat_y = Quaternion.from_axis_angle(axis_y, angle_per_frame * 1.5)
        self.quat_z = Quaternion.from_axis_angle(axis_z, angle_per_frame * 0.5)
    
    def update(self, delta_time: float):
        """Update cube rotation using quaternions"""
        # Apply rotations
        self.rotation = self.quat_x * self.rotation
        self.rotation = self.quat_y * self.rotation
        self.rotation = self.quat_z * self.rotation
        
        # Normalize to prevent numerical drift
        self.rotation = self.rotation.normalize()
    
    def get_transformed_vertices(self) -> List[Vector3]:
        """Get vertices after applying quaternion rotation and position"""
        transformed = []
        for vertex in self.vertices:
            # Apply rotation using quaternion
            rotated = self.rotation.rotate_vector(vertex)
            
            # Apply position
            transformed_vertex = rotated + self.position
            transformed.append(transformed_vertex)
        
        return transformed
    
    def draw(self, surface: pygame.Surface):
        """Draw the cube"""
        transformed_vertices = self.get_transformed_vertices()
        
        # Project vertices to 2D
        projected_vertices = [v.project_2d() for v in transformed_vertices]
        
        # Draw edges
        for edge in self.edges:
            start = projected_vertices[edge[0]]
            end = projected_vertices[edge[1]]
            pygame.draw.line(surface, EDGE_COLOR, start, end, 2)
        
        # Draw vertices as small circles
        for vertex in projected_vertices:
            pygame.draw.circle(surface, CUBE_COLOR, (int(vertex[0]), int(vertex[1])), 4)


class QuaternionGameEngine:
    """Game engine using quaternion-based rotations"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Quaternion-Based 3D Game Engine - Rotating Cube")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create cube
        self.cube = QuaternionCube(size=150)
        self.cube.position = Vector3(0, 0, 300)
    
    def handle_events(self):
        """Handle user input and window events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Reset rotation
                    self.cube.rotation = Quaternion(1, 0, 0, 0)
    
    def update(self, delta_time: float):
        """Update game state"""
        self.cube.update(delta_time)
    
    def render(self):
        """Render the scene"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw cube
        self.cube.draw(self.screen)
        
        # Draw info text
        font = pygame.font.Font(None, 24)
        info_text = font.render("Quaternion-Based 3D Engine - Press ESC to exit, SPACE to reset", True, (200, 200, 200))
        self.screen.blit(info_text, (10, 10))
        
        fps_text = font.render(f"FPS: {int(self.clock.get_fps())}", True, (200, 200, 200))
        self.screen.blit(fps_text, (10, 40))
        
        # Display quaternion info
        euler = self.cube.rotation.to_euler_angles()
        euler_text = font.render(
            f"Rotation (deg): X={math.degrees(euler[0]):.1f} Y={math.degrees(euler[1]):.1f} Z={math.degrees(euler[2]):.1f}",
            True,
            (200, 200, 200)
        )
        self.screen.blit(euler_text, (10, 70))
        
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
    engine = QuaternionGameEngine()
    engine.run()
