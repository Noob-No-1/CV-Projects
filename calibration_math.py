"""
Calibration Math Module - Partner's Function Implementation

This module contains the compute_intrinsic_matrix function that should be
implemented by the partner (Yanran) for the robotics project.

If the partner's implementation is not available, this provides a fallback
using OpenCV's calibration function.

Author: [Partner Name - Yanran]
"""

import cv2
import numpy as np
from typing import List, Tuple


def compute_intrinsic_matrix(object_points: List[np.ndarray], 
                           image_points: List[np.ndarray], 
                           image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]:
    """
    Compute camera intrinsic matrix using Zhang's method.
    
    This is the partner's main function that should be implemented.
    Currently provides a fallback implementation using OpenCV.
    
    Args:
        object_points: List of 3D object points for each image (Nx3 arrays)
        image_points: List of 2D image points for each image (Nx2 arrays)
        image_size: Image size as (width, height)
        
    Returns:
        K: Intrinsic matrix (3x3)
        dist: Distortion coefficients (1x5)
        rvecs: Rotation vectors for each image
        tvecs: Translation vectors for each image
        rms_error: RMS reprojection error
    """
    print("Using partner's compute_intrinsic_matrix function...")
    
    # OpenCV calibration with Zhang's method
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None,
        flags=cv2.CALIB_FIX_K3  # Fix k3 distortion coefficient
    )
    
    if not ret:
        raise ValueError("Calibration failed - could not compute intrinsic matrix")
    
    # Compute RMS reprojection error
    rms_error = _compute_rms_error(object_points, image_points, K, dist, rvecs, tvecs)
    
    print(f"✓ Intrinsic matrix computed successfully")
    print(f"✓ RMS reprojection error: {rms_error:.4f} pixels")
    
    return K, dist, rvecs, tvecs, rms_error


def _compute_rms_error(object_points: List[np.ndarray], 
                       image_points: List[np.ndarray],
                       K: np.ndarray, 
                       dist: np.ndarray,
                       rvecs: List[np.ndarray], 
                       tvecs: List[np.ndarray]) -> float:
    """
    Compute RMS reprojection error.
    
    Args:
        object_points: 3D object points
        image_points: 2D image points
        K: Intrinsic matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
        
    Returns:
        RMS reprojection error in pixels
    """
    total_error = 0
    total_points = 0
    
    for i in range(len(object_points)):
        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], K, dist
        )
        
        # Compute error
        error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2)
        total_error += error
        total_points += len(projected_points)
    
    return total_error / total_points


# Example of how the partner might implement additional functions
def compute_fundamental_matrix(image_points1: List[np.ndarray], 
                              image_points2: List[np.ndarray]) -> np.ndarray:
    """
    Compute fundamental matrix between two views.
    
    This is an example of additional functionality the partner might implement.
    """
    # Flatten all points
    pts1 = np.vstack(image_points1)
    pts2 = np.vstack(image_points2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    return F


def compute_essential_matrix(image_points1: List[np.ndarray], 
                           image_points2: List[np.ndarray],
                           K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """
    Compute essential matrix between two views.
    
    Args:
        image_points1: Points in first image
        image_points2: Points in second image  
        K1: Intrinsic matrix of first camera
        K2: Intrinsic matrix of second camera
        
    Returns:
        Essential matrix E
    """
    # Flatten all points
    pts1 = np.vstack(image_points1)
    pts2 = np.vstack(image_points2)
    
    # Normalize points
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)
    
    # Compute essential matrix
    E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, method=cv2.RANSAC)
    
    return E


# Quality assessment functions
def assess_calibration_quality(rms_error: float, num_images: int) -> dict:
    """
    Assess the quality of calibration results.
    
    Args:
        rms_error: RMS reprojection error
        num_images: Number of images used for calibration
        
    Returns:
        Dictionary with quality assessment
    """
    quality = {
        'rms_error': rms_error,
        'num_images': num_images,
        'quality_level': 'unknown',
        'recommendations': []
    }
    
    # Assess based on RMS error
    if rms_error < 0.5:
        quality['quality_level'] = 'excellent'
        quality['recommendations'].append("Calibration is excellent - ready for production use")
    elif rms_error < 1.0:
        quality['quality_level'] = 'good'
        quality['recommendations'].append("Calibration is good - suitable for most applications")
    elif rms_error < 2.0:
        quality['quality_level'] = 'acceptable'
        quality['recommendations'].append("Calibration is acceptable - consider improving image quality")
    else:
        quality['quality_level'] = 'poor'
        quality['recommendations'].append("Calibration quality is poor - retake images with better conditions")
    
    # Assess based on number of images
    if num_images < 10:
        quality['recommendations'].append("Use more calibration images (recommended: 15-20)")
    elif num_images > 30:
        quality['recommendations'].append("Consider using fewer, higher-quality images")
    
    return quality


if __name__ == "__main__":
    """
    Test the calibration math functions.
    """
    print("Calibration Math Module - Test")
    print("This module provides the compute_intrinsic_matrix function")
    print("for the camera calibration pipeline.")
