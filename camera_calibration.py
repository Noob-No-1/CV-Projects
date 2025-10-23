"""
Camera Calibration System using Zhang's Method
Robotics Project - Complete Implementation

This module implements a comprehensive camera calibration pipeline including:
- Image loading and preprocessing
- Corner detection with sub-pixel accuracy
- 3D-2D correspondence building
- Integration with partner's compute_intrinsic_matrix function
- Reprojection error analysis
- Undistortion and visualization
- Comprehensive reporting and demo capabilities

Author: [Your Name]
Partner: Yanran (implements compute_intrinsic_matrix)
"""

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
import json

# Import partner's function
try:
    from calibration_math import compute_intrinsic_matrix
    PARTNER_FUNCTION_AVAILABLE = True
except ImportError:
    print("Warning: calibration_math module not found. Using OpenCV fallback.")
    PARTNER_FUNCTION_AVAILABLE = False


class CameraCalibrator:
    """
    Main camera calibration pipeline using Zhang's method.
    
    This class orchestrates the complete calibration process from image loading
    through final calibration and error analysis.
    """
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (6, 9), 
                 square_size: float = 0.025, image_folder: str = "online_datasets/",
                 output_folder: str = "results/"):
        """
        Initialize the camera calibrator.
        
        Args:
            checkerboard_size: Number of internal corners (width, height)
            square_size: Size of squares in meters
            image_folder: Path to folder containing calibration images
            output_folder: Path to folder for output results
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.image_folder = image_folder
        self.output_folder = output_folder
        
        # Create output directories
        self._create_output_directories()
        
        # Calibration data storage
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.image_size = None
        self.images_loaded = []
        self.corner_detection_results = []
        
        # Calibration results
        self.K = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.rms_error = None
        self.reprojection_errors = []
        
        # Sub-pixel corner detection criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print("=" * 70)
        print("CAMERA CALIBRATION USING ZHANG'S METHOD")
        print("=" * 70)
        print(f"Checkerboard size: {checkerboard_size}")
        print(f"Square size: {square_size}m")
        print(f"Image folder: {image_folder}")
        print(f"Output folder: {output_folder}")
        print("=" * 70)
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_folder,
            os.path.join(self.output_folder, "corner_detections"),
            os.path.join(self.output_folder, "undistorted_samples")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_images(self) -> int:
        """
        Load all calibration images from the specified folder.
        
        Returns:
            Number of images successfully loaded
        """
        print(f"\nLoading images from {self.image_folder}...")
        
        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for extension in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.image_folder, extension)))
            image_paths.extend(glob.glob(os.path.join(self.image_folder, extension.upper())))
        
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.image_folder}")
        
        # Sort paths for consistent ordering
        image_paths.sort()
        
        # Load images
        self.images_loaded = []
        for i, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    self.images_loaded.append({
                        'path': image_path,
                        'image': image,
                        'filename': os.path.basename(image_path)
                    })
                    
                    # Set image size from first image
                    if self.image_size is None:
                        self.image_size = (image.shape[1], image.shape[0])  # (width, height)
                        
            except Exception as e:
                print(f"Warning: Could not load {image_path}: {e}")
        
        print(f"Found {len(self.images_loaded)} images")
        return len(self.images_loaded)
    
    def create_object_points(self) -> np.ndarray:
        """
        Create 3D object points for the checkerboard pattern.
        
        Returns:
            3D coordinates of checkerboard corners
        """
        # Create 3D points for checkerboard corners
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Scale by square size
        
        return objp
    
    def detect_corners(self, image: np.ndarray, image_path: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect and refine checkerboard corners in a single image.
        
        Args:
            image: Input image
            image_path: Path to image for error reporting
            
        Returns:
            (success, refined_corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Refine corner positions to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners_refined
        else:
            return False, None
    
    def process_all_images(self) -> Dict[str, int]:
        """
        Detect corners in all loaded images.
        
        Returns:
            Dictionary with detection statistics
        """
        print(f"\nDetecting corners in {len(self.images_loaded)} images...")
        print("-" * 70)
        
        successful_detections = 0
        total_images = len(self.images_loaded)
        
        # Create object points (same for all images)
        objp = self.create_object_points()
        
        for i, img_data in enumerate(self.images_loaded):
            image = img_data['image']
            image_path = img_data['path']
            filename = img_data['filename']
            
            success, corners = self.detect_corners(image, image_path)
            
            if success:
                # Store successful detection
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                successful_detections += 1
                
                # Visualize and save corner detection
                self._visualize_corners(image, corners, i, filename)
                
                print(f"  ✓ [{i+1:2d}/{total_images}] {filename}")
            else:
                print(f"  ✗ [{i+1:2d}/{total_images}] {filename} - NO CORNERS DETECTED")
            
            # Store result for analysis
            self.corner_detection_results.append({
                'filename': filename,
                'success': success,
                'image_index': i
            })
        
        print("-" * 70)
        print(f"Corner detection: {successful_detections}/{total_images} successful")
        
        if successful_detections < 10:
            raise ValueError(f"Insufficient successful corner detections: {successful_detections}/10 minimum required")
        
        return {
            'total': total_images,
            'successful': successful_detections,
            'success_rate': successful_detections / total_images
        }
    
    def _visualize_corners(self, image: np.ndarray, corners: np.ndarray, 
                          image_index: int, filename: str):
        """Visualize and save corner detection results."""
        # Draw corners on image
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, self.checkerboard_size, corners, True)
        
        # Save visualization
        output_path = os.path.join(self.output_folder, "corner_detections", 
                                 f"corners_{image_index:02d}_{filename}")
        cv2.imwrite(output_path, img_with_corners)
    
    def calibrate(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform camera calibration using partner's function or OpenCV fallback.
        
        Returns:
            Dictionary containing calibration results
        """
        print(f"\nCalibrating camera with {len(self.objpoints)} successful images...")
        
        if PARTNER_FUNCTION_AVAILABLE:
            print("Using partner's compute_intrinsic_matrix function...")
            try:
                self.K, self.dist, self.rvecs, self.tvecs, self.rms_error = compute_intrinsic_matrix(
                    self.objpoints, self.imgpoints, self.image_size
                )
                print("✓ Partner's function completed successfully")
            except Exception as e:
                print(f"Error in partner's function: {e}")
                print("Falling back to OpenCV calibration...")
                self._calibrate_opencv()
        else:
            print("Using OpenCV calibration (partner function not available)...")
            self._calibrate_opencv()
        
        # Compute detailed reprojection errors
        self._compute_reprojection_errors()
        
        return {
            'K': self.K,
            'dist': self.dist,
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
            'rms_error': self.rms_error,
            'mean_reprojection_error': np.mean(self.reprojection_errors),
            'max_reprojection_error': np.max(self.reprojection_errors)
        }
    
    def _calibrate_opencv(self):
        """Fallback calibration using OpenCV."""
        ret, self.K, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )
        self.rms_error = self._compute_rms_error()
    
    def _compute_rms_error(self) -> float:
        """Compute RMS reprojection error."""
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], 
                                            self.K, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            total_points += 1
        
        return total_error / total_points
    
    def _compute_reprojection_errors(self):
        """Compute detailed reprojection errors for each image."""
        self.reprojection_errors = []
        
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], 
                                            self.K, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            self.reprojection_errors.append(error)
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from an image.
        
        Args:
            image: Input distorted image
            
        Returns:
            Undistorted image
        """
        return cv2.undistort(image, self.K, self.dist)
    
    def save_calibration(self, filepath: str):
        """
        Save calibration results to .npz file.
        
        Args:
            filepath: Path to save calibration data
        """
        np.savez(filepath,
                 K=self.K,
                 dist=self.dist,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs,
                 rms_error=self.rms_error,
                 image_size=self.image_size,
                 checkerboard_size=self.checkerboard_size,
                 square_size=self.square_size)
        print(f"✓ Calibration saved to {filepath}")
    
    def print_calibration_results(self):
        """Print comprehensive calibration results."""
        print("\n" + "=" * 70)
        print("CALIBRATION RESULTS")
        print("=" * 70)
        
        # Intrinsic matrix
        print(f"\nIntrinsic Matrix K:")
        print(f"[[{self.K[0,0]:8.2f} {self.K[0,1]:8.2f} {self.K[0,2]:8.2f}]")
        print(f" [{self.K[1,0]:8.2f} {self.K[1,1]:8.2f} {self.K[1,2]:8.2f}]")
        print(f" [{self.K[2,0]:8.2f} {self.K[2,1]:8.2f} {self.K[2,2]:8.2f}]]")
        
        # Focal lengths
        print(f"\nFocal lengths:")
        print(f"  fx = {self.K[0,0]:8.2f} pixels")
        print(f"  fy = {self.K[1,1]:8.2f} pixels")
        
        # Principal point
        print(f"\nPrincipal point:")
        print(f"  cx = {self.K[0,2]:8.2f} pixels")
        print(f"  cy = {self.K[1,2]:8.2f} pixels")
        
        # Distortion coefficients
        print(f"\nDistortion coefficients:")
        print(f"  k1 = {self.dist[0,0]:+8.6f}")
        print(f"  k2 = {self.dist[0,1]:+8.6f}")
        print(f"  p1 = {self.dist[0,2]:+8.6f}")
        print(f"  p2 = {self.dist[0,3]:+8.6f}")
        if len(self.dist[0]) > 4:
            print(f"  k3 = {self.dist[0,4]:+8.6f}")
        
        # Reprojection error
        mean_error = np.mean(self.reprojection_errors)
        max_error = np.max(self.reprojection_errors)
        
        print(f"\nReprojection Error:")
        print(f"  Mean = {mean_error:.4f} pixels")
        print(f"  Max  = {max_error:.4f} pixels")
        
        # Quality assessment
        if mean_error < 0.5:
            quality = "✓✓✓ Excellent! (< 0.5 pixels)"
        elif mean_error < 1.0:
            quality = "✓✓ Good (< 1.0 pixels)"
        elif mean_error < 2.0:
            quality = "✓ Acceptable (< 2.0 pixels)"
        else:
            quality = "⚠ Poor (>= 2.0 pixels)"
        
        print(f"  {quality}")
        
        print(f"\nImages Used: {len(self.objpoints)}")
        print("=" * 70)


def visualize_results(calibrator: CameraCalibrator, output_folder: str):
    """
    Create comprehensive visualization of calibration results.
    
    Args:
        calibrator: Calibrated CameraCalibrator instance
        output_folder: Path to save visualization
    """
    print(f"\nCreating comprehensive visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Camera Calibration Results', fontsize=16, fontweight='bold')
    
    # Subplot 1: Per-image reprojection error
    ax1 = plt.subplot(2, 3, 1)
    image_indices = range(len(calibrator.reprojection_errors))
    bars = ax1.bar(image_indices, calibrator.reprojection_errors, 
                   color=['green' if e < 1.0 else 'orange' if e < 2.0 else 'red' for e in calibrator.reprojection_errors])
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Reprojection Error (pixels)')
    ax1.set_title('Per-Image Reprojection Error')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2 pixel threshold')
    ax1.legend()
    
    # Subplot 2: Error distribution histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(calibrator.reprojection_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Reprojection Error (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=np.mean(calibrator.reprojection_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(calibrator.reprojection_errors):.3f}')
    ax2.legend()
    
    # Subplot 3: Calibration summary
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    # Create summary text
    summary_text = f"""Calibration Summary
    
Images Used: {len(calibrator.objpoints)}
Checkerboard: {calibrator.checkerboard_size}
Square Size: {calibrator.square_size}m

Intrinsic Matrix:
fx = {calibrator.K[0,0]:.1f}
fy = {calibrator.K[1,1]:.1f}
cx = {calibrator.K[0,2]:.1f}
cy = {calibrator.K[1,2]:.1f}

Distortion:
k1 = {calibrator.dist[0,0]:.4f}
k2 = {calibrator.dist[0,1]:.4f}
p1 = {calibrator.dist[0,2]:.4f}
p2 = {calibrator.dist[0,3]:.4f}

RMS Error: {calibrator.rms_error:.4f} px
Mean Error: {np.mean(calibrator.reprojection_errors):.4f} px
Max Error: {np.max(calibrator.reprojection_errors):.4f} px"""
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Subplot 4: Distortion field visualization
    ax4 = plt.subplot(2, 3, 4)
    _visualize_distortion_field(ax4, calibrator.K, calibrator.dist, calibrator.image_size)
    
    # Subplot 5-6: Before/after undistortion comparison
    if len(calibrator.images_loaded) > 0:
        # Use first successful image for comparison
        sample_image = calibrator.images_loaded[0]['image']
        undistorted = calibrator.undistort_image(sample_image)
        
        # Before
        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
        ax5.set_title('Original (Distorted)')
        ax5.axis('off')
        
        # After
        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        ax6.set_title('Undistorted')
        ax6.axis('off')
    
    plt.tight_layout()
    
    # Save high-DPI figure
    output_path = os.path.join(output_folder, "calibration_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


def _visualize_distortion_field(ax, K, dist, image_size):
    """Visualize lens distortion field."""
    h, w = image_size[1], image_size[0]
    
    # Create grid of points with proper spacing
    step = 40  # Larger step to avoid reshape issues
    y, x = np.mgrid[0:h:step, 0:w:step]
    points = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
    
    # Undistort points
    points_undist = cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    
    # Calculate displacement vectors
    dx = points_undist[:, 0] - points[:, 0]
    dy = points_undist[:, 1] - points[:, 1]
    
    # Reshape for quiver plot - ensure dimensions match
    grid_h, grid_w = x.shape
    dx_grid = dx.reshape(grid_h, grid_w)
    dy_grid = dy.reshape(grid_h, grid_w)
    
    # Plot distortion field
    ax.quiver(x, y, dx_grid, dy_grid, angles='xy', scale_units='xy', scale=1, alpha=0.6)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_title('Distortion Field')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.invert_yaxis()  # Image coordinates


def save_report(calibrator: CameraCalibrator, output_folder: str):
    """
    Generate comprehensive text report.
    
    Args:
        calibrator: Calibrated CameraCalibrator instance
        output_folder: Path to save report
    """
    print(f"\nGenerating comprehensive report...")
    
    report_path = os.path.join(output_folder, "calibration_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("CAMERA CALIBRATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Calibration Method: Zhang's Method\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Checkerboard size: {calibrator.checkerboard_size}\n")
        f.write(f"Square size: {calibrator.square_size}m\n")
        f.write(f"Image size: {calibrator.image_size}\n")
        f.write(f"Images processed: {len(calibrator.images_loaded)}\n")
        f.write(f"Successful detections: {len(calibrator.objpoints)}\n\n")
        
        # Calibration results
        f.write("CALIBRATION RESULTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Intrinsic Matrix K:\n")
        f.write(f"{calibrator.K}\n\n")
        
        f.write(f"Distortion coefficients:\n")
        f.write(f"{calibrator.dist}\n\n")
        
        f.write(f"RMS reprojection error: {calibrator.rms_error:.6f} pixels\n")
        f.write(f"Mean reprojection error: {np.mean(calibrator.reprojection_errors):.6f} pixels\n")
        f.write(f"Max reprojection error: {np.max(calibrator.reprojection_errors):.6f} pixels\n\n")
        
        # Per-image errors
        f.write("PER-IMAGE REPROJECTION ERRORS\n")
        f.write("-" * 30 + "\n")
        for i, (result, error) in enumerate(zip(calibrator.corner_detection_results, calibrator.reprojection_errors)):
            if result['success']:
                f.write(f"Image {i:2d}: {result['filename']:20s} - {error:.4f} pixels\n")
        
        # Quality assessment
        mean_error = np.mean(calibrator.reprojection_errors)
        f.write(f"\nQUALITY ASSESSMENT\n")
        f.write("-" * 20 + "\n")
        if mean_error < 0.5:
            f.write("Quality: EXCELLENT (< 0.5 pixels)\n")
        elif mean_error < 1.0:
            f.write("Quality: GOOD (< 1.0 pixels)\n")
        elif mean_error < 2.0:
            f.write("Quality: ACCEPTABLE (< 2.0 pixels)\n")
        else:
            f.write("Quality: POOR (>= 2.0 pixels)\n")
        
        f.write(f"Success rate: {len(calibrator.objpoints)/len(calibrator.images_loaded)*100:.1f}%\n")
    
    print(f"✓ Report saved to {report_path}")


def create_undistorted_samples(calibrator: CameraCalibrator, output_folder: str, num_samples: int = 3):
    """
    Create before/after undistortion comparison samples.
    
    Args:
        calibrator: Calibrated CameraCalibrator instance
        output_folder: Path to save samples
        num_samples: Number of sample images to process
    """
    print(f"\nCreating undistorted samples...")
    
    successful_images = [img for i, img in enumerate(calibrator.images_loaded) 
                        if calibrator.corner_detection_results[i]['success']]
    
    num_samples = min(num_samples, len(successful_images))
    
    for i in range(num_samples):
        img_data = successful_images[i]
        original = img_data['image']
        undistorted = calibrator.undistort_image(original)
        
        # Create side-by-side comparison
        comparison = np.hstack([original, undistorted])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (original.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save comparison
        output_path = os.path.join(output_folder, "undistorted_samples", 
                                 f"comparison_{i:02d}_{img_data['filename']}")
        cv2.imwrite(output_path, comparison)
    
    print(f"✓ Created {num_samples} undistorted samples")


def main():
    """
    Main calibration pipeline orchestration.
    
    This function coordinates the complete calibration workflow:
    1. Initialize calibrator
    2. Load and process images
    3. Detect corners
    4. Perform calibration
    5. Analyze errors
    6. Visualize results
    7. Save everything
    """
    try:
        # Initialize calibrator
        calibrator = CameraCalibrator()
        
        # Step 1: Load images
        num_images = calibrator.load_images()
        if num_images == 0:
            raise ValueError("No images found in the specified folder")
        
        # Step 2: Process images and detect corners
        detection_stats = calibrator.process_all_images()
        
        # Step 3: Perform calibration
        calibration_results = calibrator.calibrate()
        
        # Step 4: Print results
        calibrator.print_calibration_results()
        
        # Step 5: Create visualizations
        visualize_results(calibrator, calibrator.output_folder)
        
        # Step 6: Generate report
        save_report(calibrator, calibrator.output_folder)
        
        # Step 7: Create undistorted samples
        create_undistorted_samples(calibrator, calibrator.output_folder)
        
        # Step 8: Save calibration data
        calibration_path = os.path.join(calibrator.output_folder, "calibration.npz")
        calibrator.save_calibration(calibration_path)
        
        print(f"\n✅ CALIBRATION COMPLETE!")
        print(f"Results saved to: {calibrator.output_folder}")
        
        return calibrator
        
    except Exception as e:
        print(f"\n❌ CALIBRATION FAILED: {e}")
        print("\nTroubleshooting suggestions:")
        print("- Check that images/ folder contains calibration images")
        print("- Ensure images show clear checkerboard patterns")
        print("- Try different lighting conditions")
        print("- Verify checkerboard size parameters")
        raise


if __name__ == "__main__":
    main()
