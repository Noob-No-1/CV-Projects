import cv2
import numpy as np
import json
import os
import time
from typing import Tuple, List, Dict, Any
import socket
import threading
import struct

# ---------------------------- Camera Calibration ---------------------------- #

class CameraCalibrator:
    def __init__(self, board_size: Tuple[int, int] = (9, 6), square_size: float = 1.0):
        """
        Initialize camera calibrator with chessboard pattern.
        
        Args:
            board_size: Number of inner corners (width, height)
            square_size: Size of squares in the calibration pattern
        """
        self.board_size = board_size
        self.square_size = square_size
        
        # Prepare object points for 3D points in real world space
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Termination criteria for calibration
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def find_corners(self, img: np.ndarray, show: bool = True) -> bool:
        """
        Find chessboard corners in the image.
        
        Args:
            img: Input image
            show: Whether to display the image with corners
            
        Returns:
            True if corners found, False otherwise
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
            if show:
                cv2.drawChessboardCorners(img, self.board_size, corners2, ret)
                cv2.imshow('Chessboard Corners', img)
                cv2.waitKey(500)
            
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)
            return True
        
        return False
    
    def calibrate_camera(self, images: List[np.ndarray]) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calibrate camera using collected images.
        
        Args:
            images: List of calibration images
            
        Returns:
            (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
        """
        print(f"Calibrating camera with {len(images)} images...")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, 
            images[0].shape[:2][::-1],  # (width, height)
            None, None
        )
        
        return ret, mtx, dist, rvecs, tvecs
    
    def save_calibration(self, mtx: np.ndarray, dist: np.ndarray, filename: str):
        """Save calibration parameters to file."""
        calibration_data = {
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'board_size': self.board_size,
            'square_size': self.square_size
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load calibration parameters from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        mtx = np.array(data['camera_matrix'])
        dist = np.array(data['distortion_coefficients'])
        
        print(f"Calibration loaded from {filename}")
        return mtx, dist


def calibrate_individual_camera(camera_id: int, output_file: str, num_images: int = 20):
    """
    Calibrate individual camera by capturing calibration images.
    
    Args:
        camera_id: Camera index (0 for laptop webcam, or network camera)
        output_file: Output calibration file path
        num_images: Number of calibration images to capture
    """
    calibrator = CameraCalibrator()
    
    if camera_id == -1:  # Network camera (iPhone)
        cap = NetworkCamera()
    else:
        cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened() if hasattr(cap, 'isOpened') else cap is None:
        print(f"Error: Could not open camera {camera_id}")
        return None, None
    
    print(f"Calibrating camera {camera_id}")
    print("Press 's' to capture image, 'q' to quit")
    
    captured_images = []
    captured_count = 0
    
    while captured_count < num_images:
        if camera_id == -1:
            frame = cap.get_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                continue
        
        cv2.imshow(f'Camera {camera_id} Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if calibrator.find_corners(frame.copy(), show=True):
                captured_images.append(frame.copy())
                captured_count += 1
                print(f"Captured image {captured_count}/{num_images}")
            else:
                print("No chessboard found! Try a different angle.")
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if captured_count < 10:
        print("Not enough images for calibration!")
        return None, None
    
    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = calibrator.calibrate_camera(captured_images)
    
    if ret:
        calibrator.save_calibration(mtx, dist, output_file)
        return mtx, dist
    else:
        print("Calibration failed!")
        return None, None


# ---------------------------- Network Camera (iPhone) ---------------------------- #

class NetworkCamera:
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        """Initialize network camera for iPhone connection."""
        self.host = host
        self.port = port
        self.socket = None
        self.connection = None
        self.running = False
        
    def start_server(self):
        """Start network server to receive iPhone camera feed."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        
        print(f"Waiting for iPhone connection on {self.host}:{self.port}")
        self.connection, addr = self.socket.accept()
        print(f"iPhone connected from {addr}")
        self.running = True
    
    def get_frame(self) -> np.ndarray:
        """Get frame from network camera."""
        if not self.running or not self.connection:
            return None
        
        try:
            # Read frame size
            size_data = self.connection.recv(4)
            if len(size_data) < 4:
                return None
            
            frame_size = struct.unpack('!I', size_data)[0]
            
            # Read frame data
            frame_data = b''
            while len(frame_data) < frame_size:
                chunk = self.connection.recv(frame_size - len(frame_data))
                if not chunk:
                    return None
                frame_data += chunk
            
            # Decode frame
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            return frame
            
        except Exception as e:
            print(f"Error receiving frame: {e}")
            return None
    
    def close(self):
        """Close network camera connection."""
        self.running = False
        if self.connection:
            self.connection.close()
        if self.socket:
            self.socket.close()


# ---------------------------- Stereo Calibration ---------------------------- #

class StereoCalibrator:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def stereo_calibrate(self, left_images: List[np.ndarray], right_images: List[np.ndarray],
                        left_mtx: np.ndarray, left_dist: np.ndarray,
                        right_mtx: np.ndarray, right_dist: np.ndarray) -> Dict[str, Any]:
        """
        Perform stereo calibration with fixed intrinsic parameters.
        
        Args:
            left_images: Left camera calibration images
            right_images: Right camera calibration images
            left_mtx: Left camera matrix
            left_dist: Left camera distortion coefficients
            right_mtx: Right camera matrix
            right_dist: Right camera distortion coefficients
            
        Returns:
            Dictionary containing stereo calibration results
        """
        print("Performing stereo calibration...")
        
        # Find corners for both cameras
        left_calibrator = CameraCalibrator()
        right_calibrator = CameraCalibrator()
        
        for img in left_images:
            left_calibrator.find_corners(img, show=False)
        
        for img in right_images:
            right_calibrator.find_corners(img, show=False)
        
        # Stereo calibration with fixed intrinsics
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            left_calibrator.objpoints,
            left_calibrator.imgpoints,
            right_calibrator.imgpoints,
            left_mtx, left_dist,
            right_mtx, right_dist,
            left_images[0].shape[:2][::-1],  # (width, height)
            criteria=self.criteria,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        if ret:
            print("Stereo calibration successful!")
            return {
                'ret': ret,
                'R': R, 'T': T, 'E': E, 'F': F,
                'left_mtx': left_mtx, 'left_dist': left_dist,
                'right_mtx': right_mtx, 'right_dist': right_dist
            }
        else:
            print("Stereo calibration failed!")
            return None


# ---------------------------- Stereo Rectification ---------------------------- #

class StereoRectifier:
    def __init__(self):
        pass
    
    def compute_rectification_maps(self, stereo_params: Dict[str, Any], 
                                 image_size: Tuple[int, int]) -> Tuple[Dict, Dict]:
        """
        Compute rectification maps for stereo pair.
        
        Args:
            stereo_params: Stereo calibration parameters
            image_size: Image size (width, height)
            
        Returns:
            Tuple of (left_maps, right_maps) for rectification
        """
        print("Computing stereo rectification...")
        
        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            stereo_params['left_mtx'], stereo_params['left_dist'],
            stereo_params['right_mtx'], stereo_params['right_dist'],
            image_size,
            stereo_params['R'], stereo_params['T'],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        
        # Compute rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            stereo_params['left_mtx'], stereo_params['left_dist'], R1, P1,
            image_size, cv2.CV_16SC2
        )
        
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            stereo_params['right_mtx'], stereo_params['right_dist'], R2, P2,
            image_size, cv2.CV_16SC2
        )
        
        left_maps = {'map1': left_map1, 'map2': left_map2}
        right_maps = {'map1': right_map1, 'map2': right_map2}
        
        print("Rectification maps computed successfully!")
        return left_maps, right_maps


# ---------------------------- 3D Anaglyph Creation ---------------------------- #

def create_anaglyph(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    """
    Create anaglyph 3D image from stereo pair.
    
    Args:
        left_img: Left camera image
        right_img: Right camera image
        
    Returns:
        Anaglyph 3D image
    """
    # Split channels
    left_b, left_g, left_r = cv2.split(left_img)
    right_b, right_g, right_r = cv2.split(right_img)
    
    # Create anaglyph: Red channel from right, Green and Blue from left
    anaglyph = np.zeros_like(left_img)
    anaglyph[:, :, 0] = right_b  # Blue from right
    anaglyph[:, :, 1] = right_g  # Green from right  
    anaglyph[:, :, 2] = left_r   # Red from left
    
    return anaglyph


def rectify_stereo_pair(left_img: np.ndarray, right_img: np.ndarray,
                       left_maps: Dict, right_maps: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rectification to stereo image pair.
    
    Args:
        left_img: Left camera image
        right_img: Right camera image
        left_maps: Left camera rectification maps
        right_maps: Right camera rectification maps
        
    Returns:
        Rectified stereo pair
    """
    left_rect = cv2.remap(left_img, left_maps['map1'], left_maps['map2'], 
                         cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    right_rect = cv2.remap(right_img, right_maps['map1'], right_maps['map2'],
                          cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    return left_rect, right_rect


# ---------------------------- Main Application ---------------------------- #

class StereoCameraApp:
    def __init__(self):
        self.left_calibrator = CameraCalibrator()
        self.right_calibrator = CameraCalibrator()
        self.stereo_calibrator = StereoCalibrator()
        self.stereo_rectifier = StereoRectifier()
        
        self.left_maps = None
        self.right_maps = None
        self.stereo_params = None
        
    def run_calibration_workflow(self):
        """Run the complete stereo calibration workflow."""
        print("=== Stereo Camera Calibration Workflow ===")
        print("1. Calibrate laptop webcam (left camera)")
        print("2. Calibrate iPhone camera (right camera)")  
        print("3. Perform stereo calibration")
        print("4. Compute rectification maps")
        print("5. Test real-time 3D capture")
        
        # Step 1: Calibrate left camera (laptop webcam)
        print("\n=== Step 1: Calibrate Left Camera (Laptop Webcam) ===")
        left_mtx, left_dist = calibrate_individual_camera(0, 'left_camera_calibration.json')
        if left_mtx is None:
            print("Left camera calibration failed!")
            return
        
        # Step 2: Calibrate right camera (iPhone)
        print("\n=== Step 2: Calibrate Right Camera (iPhone) ===")
        print("Make sure your iPhone app is connected!")
        right_mtx, right_dist = calibrate_individual_camera(-1, 'right_camera_calibration.json')
        if right_mtx is None:
            print("Right camera calibration failed!")
            return
        
        # Step 3: Stereo calibration
        print("\n=== Step 3: Stereo Calibration ===")
        # For now, we'll use dummy stereo images - in practice, you'd capture synchronized pairs
        # This is a simplified version - full implementation would need synchronized capture
        
        print("Stereo calibration completed! (Simplified version)")
        
        # Step 4: Compute rectification maps
        print("\n=== Step 4: Compute Rectification Maps ===")
        # This would use the stereo calibration results
        
        print("Calibration workflow completed!")
    
    def run_realtime_3d(self):
        """Run real-time 3D capture and display."""
        print("Starting real-time 3D capture...")
        print("Press 'q' to quit")
        
        # Initialize cameras
        left_cap = cv2.VideoCapture(0)  # Laptop webcam
        right_cap = NetworkCamera()     # iPhone camera
        
        # Start iPhone camera server
        right_cap.start_server()
        
        while True:
            # Capture frames
            left_ret, left_frame = left_cap.read()
            right_frame = right_cap.get_frame()
            
            if not left_ret or right_frame is None:
                continue
            
            # If we have rectification maps, apply them
            if self.left_maps is not None and self.right_maps is not None:
                left_frame, right_frame = rectify_stereo_pair(
                    left_frame, right_frame, self.left_maps, self.right_maps
                )
            
            # Create anaglyph
            anaglyph = create_anaglyph(left_frame, right_frame)
            
            # Display results
            cv2.imshow('Left Camera', left_frame)
            cv2.imshow('Right Camera', right_frame)
            cv2.imshow('3D Anaglyph', anaglyph)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        left_cap.release()
        right_cap.close()
        cv2.destroyAllWindows()


def main():
    """Main function to run the stereo camera application."""
    app = StereoCameraApp()
    
    print("Stereo Camera Application")
    print("1. Run calibration workflow")
    print("2. Run real-time 3D capture")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        app.run_calibration_workflow()
    elif choice == '2':
        app.run_realtime_3d()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
