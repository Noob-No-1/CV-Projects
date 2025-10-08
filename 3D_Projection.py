import math
from typing import List, Tuple

import numpy as np
import pygame


# ----------------------------- Math utilities ----------------------------- #

def degrees_to_radians(deg: float) -> float:
    return deg * math.pi / 180.0


def build_rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """
    Create a right-handed rotation matrix R (world -> camera) using intrinsic
    Tait-Bryan angles (roll around Z, pitch around X, yaw around Y), applied in
    the order: R = Rz(roll) @ Rx(pitch) @ Ry(yaw).
    """
    rx = degrees_to_radians(pitch_deg)
    ry = degrees_to_radians(yaw_deg)
    rz = degrees_to_radians(roll_deg)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rx), -math.sin(rx)],
            [0.0, math.sin(rx), math.cos(rx)],
        ]
    )
    Ry = np.array(
        [
            [math.cos(ry), 0.0, math.sin(ry)],
            [0.0, 1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)],
        ]
    )
    Rz = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0.0],
            [math.sin(rz), math.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return Rz @ Rx @ Ry


def build_intrinsic_matrix(f: float, ox: float, oy: float) -> np.ndarray:
    return np.array([[f, 0.0, ox], [0.0, f, oy], [0.0, 0.0, 1.0]])


def build_extrinsic_matrix(R: np.ndarray, camera_center_world: np.ndarray) -> np.ndarray:
    """
    Build [R|t] with t = -R @ C. C is the camera center in world coordinates.
    The transform maps world coordinates to the camera frame: Xc = R Xw + t.
    """
    t = (-R @ camera_center_world.reshape(3, 1)).reshape(3)
    Rt = np.hstack([R, t.reshape(3, 1)])
    return Rt


def project_points(K: np.ndarray, Rt: np.ndarray, points_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project homogeneous world points (N x 4) to pixel coordinates (N x 2).
    Returns (uv, depths) where depths are the Z values in camera frame.
    """
    assert points_h.shape[1] == 4
    P = K @ Rt  # 3x4
    X = P @ points_h.T  # 3xN
    x = X[:2, :] / X[2:3, :]
    uv = x.T

    # Depths from camera frame for visibility testing
    Xc = Rt @ points_h.T  # 3xN (without K)
    depths = Xc[2, :]
    return uv, depths


# ------------------------------ Scene geometry ---------------------------- #

def cube_vertices(size: float = 2.0) -> np.ndarray:
    s = size / 2.0
    verts = np.array(
        [
            [-s, -s, -s, 1.0],
            [s, -s, -s, 1.0],
            [s, s, -s, 1.0],
            [-s, s, -s, 1.0],
            [-s, -s, s, 1.0],
            [s, -s, s, 1.0],
            [s, s, s, 1.0],
            [-s, s, s, 1.0],
        ],
        dtype=float,
    )
    return verts


def cube_edges() -> List[Tuple[int, int]]:
    # 12 edges of a cube
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]


def cube_faces() -> List[Tuple[List[int], Tuple[float, float, float]]]:
    """
    Returns list of (face_vertex_indices, face_normal) for each cube face.
    Face normals point outward from the cube center.
    """
    return [
        # Bottom face (y = -1)
        ([0, 1, 2, 3], (0.0, -1.0, 0.0)),
        # Top face (y = +1)
        ([4, 7, 6, 5], (0.0, 1.0, 0.0)),
        # Front face (z = -1)
        ([0, 4, 5, 1], (0.0, 0.0, -1.0)),
        # Back face (z = +1)
        ([2, 6, 7, 3], (0.0, 0.0, 1.0)),
        # Left face (x = -1)
        ([0, 3, 7, 4], (-1.0, 0.0, 0.0)),
        # Right face (x = +1)
        ([1, 5, 6, 2], (1.0, 0.0, 0.0)),
    ]


def transform_points(points_h: np.ndarray, S: float, T: Tuple[float, float, float]) -> np.ndarray:
    """Apply uniform scale and translation in world coordinates to homogeneous points."""
    S_mat = np.diag([S, S, S, 1.0])
    T_mat = np.eye(4)
    T_mat[:3, 3] = np.array(T)
    return (T_mat @ (S_mat @ points_h.T)).T


# --------------------------------- Drawing -------------------------------- #

def draw_lines(surface: pygame.Surface, pts2d: np.ndarray, depths: np.ndarray, edges: List[Tuple[int, int]], color: Tuple[int, int, int]):
    h, w = surface.get_height(), surface.get_width()
    for i, j in edges:
        if depths[i] > 0 and depths[j] > 0:
            xi, yi = pts2d[i]
            xj, yj = pts2d[j]
            # Only draw if inside extended screen bounds to avoid huge lines
            if -w <= xi <= 2 * w and -h <= yi <= 2 * h and -w <= xj <= 2 * w and -h <= yj <= 2 * h:
                pygame.draw.line(surface, color, (xi, yi), (xj, yj), 2)


def is_face_visible(face_normal: Tuple[float, float, float], camera_forward: np.ndarray) -> bool:
    """Check if a face is visible using back-face culling."""
    normal = np.array(face_normal)
    # A face is visible if its normal points toward the camera
    # (dot product with camera forward direction is negative)
    return np.dot(normal, camera_forward) < 0


def transform_face_normals(faces: List[Tuple[List[int], Tuple[float, float, float]]], 
                          R: np.ndarray) -> List[Tuple[List[int], np.ndarray]]:
    """Transform face normals from object space to world space."""
    transformed_faces = []
    for face_verts, normal in faces:
        # Transform normal from object space to world space
        # Note: For rotation matrices, R^(-1) = R^T
        world_normal = R.T @ np.array(normal)
        transformed_faces.append((face_verts, world_normal))
    return transformed_faces


def draw_filled_cube(surface: pygame.Surface, pts2d: np.ndarray, depths: np.ndarray, 
                    transformed_faces: List[Tuple[List[int], np.ndarray]], 
                    camera_forward: np.ndarray, base_color: Tuple[int, int, int]):
    """Draw a filled cube with face visibility and depth sorting."""
    h, w = surface.get_height(), surface.get_width()
    
    # Collect visible faces with their depths
    visible_faces = []
    for face_verts, world_normal in transformed_faces:
        if is_face_visible(world_normal, camera_forward):
            # Calculate average depth of face vertices
            face_depths = [depths[i] for i in face_verts if depths[i] > 0]
            if face_depths:
                avg_depth = sum(face_depths) / len(face_depths)
                visible_faces.append((face_verts, avg_depth))
    
    # Sort faces by depth (back to front for proper rendering)
    visible_faces.sort(key=lambda x: x[1], reverse=True)
    
    # Draw each visible face
    for face_verts, _ in visible_faces:
        # Get 2D points for this face
        face_points = []
        for vert_idx in face_verts:
            if depths[vert_idx] > 0:
                x, y = pts2d[vert_idx]
                # Clamp to screen bounds to avoid rendering issues
                x = max(0, min(w-1, x))
                y = max(0, min(h-1, y))
                face_points.append((int(x), int(y)))
        
        # Only draw if we have at least 3 valid points
        if len(face_points) >= 3:
            # Create a slightly darker shade for depth
            depth_factor = 0.8
            face_color = tuple(int(c * depth_factor) for c in base_color)
            pygame.draw.polygon(surface, face_color, face_points)
            
            # Draw wireframe outline for better visibility
            if len(face_points) >= 3:
                pygame.draw.polygon(surface, (255, 255, 255), face_points, 2)


def draw_text(surface: pygame.Surface, text: str, pos: Tuple[int, int], color=(255, 255, 255)):
    font = pygame.font.SysFont("monospace", 16, bold=True)
    for i, line in enumerate(text.split("\n")):
        img = font.render(line, True, color)
        surface.blit(img, (pos[0], pos[1] + i * 18))


# ---------------------------------- Main ---------------------------------- #

def main():
    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Perspective Projection - Cube")
    clock = pygame.time.Clock()

    # Intrinsics (square pixels, principal point as given)
    f = 800.0
    ox, oy = 320.0, 240.0
    K = build_intrinsic_matrix(f, ox, oy)

    # Camera pose (world -> camera). Start at origin looking down +Z.
    camera_center = np.array([0.0, 0.0, 0.0])
    pitch, yaw, roll = 0.0, 0.0, 0.0

    # Place a stack of cubes marching away from the camera
    base_cube = cube_vertices(size=2.0)
    edges = cube_edges()
    faces = cube_faces()

    colors = [
        (220, 60, 60),
        (60, 220, 60),
        (60, 120, 255),
        (250, 220, 60),
        (120, 60, 255),
        (60, 240, 240),
    ]

    running = True
    speed = 2.0  # units per second
    rot_speed = 60.0  # degrees per second
    fov_speed = 200.0  # focal change per second

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()

        # Camera local axes based movement
        R = build_rotation_matrix(pitch, yaw, roll)
        forward = R.T @ np.array([0.0, 0.0, 1.0])
        right = R.T @ np.array([1.0, 0.0, 0.0])
        up = R.T @ np.array([0.0, 1.0, 0.0])

        # WASD move (one axis at a time preference)
        if keys[pygame.K_w]:
            camera_center += forward * speed * dt
        if keys[pygame.K_s]:
            camera_center -= forward * speed * dt
        if keys[pygame.K_a]:
            camera_center -= right * speed * dt
        if keys[pygame.K_d]:
            camera_center += right * speed * dt
        if keys[pygame.K_q]:
            camera_center += up * speed * dt
        if keys[pygame.K_e]:
            camera_center -= up * speed * dt

        # Arrow keys: pitch/yaw
        if keys[pygame.K_UP]:
            pitch += rot_speed * dt
        if keys[pygame.K_DOWN]:
            pitch -= rot_speed * dt
        if keys[pygame.K_LEFT]:
            yaw += rot_speed * dt
        if keys[pygame.K_RIGHT]:
            yaw -= rot_speed * dt

        # Z/X: roll
        if keys[pygame.K_z]:
            roll -= rot_speed * dt
        if keys[pygame.K_x]:
            roll += rot_speed * dt

        # F/G: change focal length (FOV)
        if keys[pygame.K_f]:
            f += fov_speed * dt
            K = build_intrinsic_matrix(f, ox, oy)
        if keys[pygame.K_g]:
            f = max(10.0, f - fov_speed * dt)
            K = build_intrinsic_matrix(f, ox, oy)

        # Clear screen
        screen.fill((0, 0, 0))

        # Build extrinsics and project
        R = build_rotation_matrix(pitch, yaw, roll)
        Rt = build_extrinsic_matrix(R, camera_center)

        # Draw several cubes receding into the scene
        for idx, z_offset in enumerate([5.0, 7.0, 9.0, 11.0, 13.0, 15.0]):
            scale = 1.0 / (1.0 + 0.15 * idx)
            cube_w = transform_points(base_cube, S=scale, T=(0.0, 0.0, z_offset))
            uv, depths = project_points(K, Rt, cube_w)
            
            # Draw wireframe edges
            draw_lines(screen, uv, depths, edges, colors[idx % len(colors)])

        # HUD
        help_text = (
            "WASD: Move  |  QE: Up/Down  |  Arrows: Pitch/Yaw  |  Z/X: Roll\n"
            "F/G: Increase/Decrease Focal  |  ESC: Exit\n"
            f"focal={f:.1f}  pos=({camera_center[0]:.2f}, {camera_center[1]:.2f}, {camera_center[2]:.2f})  "
            f"rot(p,y,r)=({pitch:.1f}, {yaw:.1f}, {roll:.1f})"
        )
        draw_text(screen, help_text, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()


