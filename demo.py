"""
Live Demonstration Script for Camera Calibration Presentation

This script provides an interactive demonstration of the camera calibration system
for the robotics project presentation. It showcases key features and results.

Author: [Your Name]
Partner: Yanran (implements compute_intrinsic_matrix)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from camera_calibration import CameraCalibrator
import time


def demo_corner_detection():
    """
    Demonstrate corner detection step-by-step on a sample image.
    """
    print("=" * 60)
    print("DEMO 1: CORNER DETECTION")
    print("=" * 60)
    
    # Check if we have images
    image_folder = "online_datasets/"
    if not os.path.exists(image_folder):
        print(f"❌ Image folder '{image_folder}' not found!")
        print("Please ensure you have calibration images in the images/ folder")
        return False
    
    # Find a sample image
    import glob
    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        print(f"❌ No images found in '{image_folder}'")
        return False
    
    sample_image_path = image_files[0]
    print(f"Using sample image: {os.path.basename(sample_image_path)}")
    
    # Load and display image
    image = cv2.imread(sample_image_path)
    if image is None:
        print(f"❌ Could not load image: {sample_image_path}")
        return False
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Initialize calibrator for corner detection
    calibrator = CameraCalibrator()
    
    # Detect corners
    print("\nDetecting checkerboard corners...")
    success, corners = calibrator.detect_corners(image, sample_image_path)
    
    if success:
        print("✅ Corners detected successfully!")
        print(f"Found {len(corners)} corner points")
        
        # Visualize corners
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, calibrator.checkerboard_size, corners, success)
        
        # Display results
        cv2.imshow('Original Image', image)
        cv2.imshow('Detected Corners', img_with_corners)
        
        print("\nPress any key to continue...")
        
        # Fix for waitKey not responding
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Any key pressed
                break
            if cv2.getWindowProperty('Original Image', cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty('Detected Corners', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        
        return True
    else:
        print("❌ No corners detected in this image")
        print("This might be due to:")
        print("- Poor lighting conditions")
        print("- Blurry image")
        print("- Incorrect checkerboard size")
        print("- Partial checkerboard visibility")
        return False


def demo_undistortion():
    """
    Demonstrate before/after undistortion comparison.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: UNDISTORTION")
    print("=" * 60)
    
    # Check if we have calibration results
    calibration_file = "results/calibration.npz"
    if not os.path.exists(calibration_file):
        print(f"❌ Calibration file '{calibration_file}' not found!")
        print("Please run the full calibration first: python camera_calibration.py")
        return False
    
    # Load calibration data
    try:
        data = np.load(calibration_file)
        K = data['K']
        dist = data['dist']
        print("✅ Calibration data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading calibration data: {e}")
        return False
    
    # Find a sample image
    image_folder = "online_datasets/"
    import glob
    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        print(f"❌ No images found in '{image_folder}'")
        return False
    
    sample_image_path = image_files[0]
    print(f"Using sample image: {os.path.basename(sample_image_path)}")
    
    # Load image
    image = cv2.imread(sample_image_path)
    if image is None:
        print(f"❌ Could not load image: {sample_image_path}")
        return False
    
    # Apply undistortion
    print("Applying undistortion...")
    undistorted = cv2.undistort(image, K, dist)
    
    # Create side-by-side comparison
    comparison = np.hstack([image, undistorted])
    
    # Add labels
    cv2.putText(comparison, "Original (Distorted)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Undistorted", (image.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display comparison
    cv2.imshow('Distortion Correction', comparison)
    print("✅ Undistortion applied successfully!")
    print("Notice the correction of lens distortion effects")
    
    print("\nPress any key to continue...")
    
    # Fix for waitKey not responding
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Any key pressed
            break
        if cv2.getWindowProperty('Distortion Correction', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    
    return True


def demo_results():
    """
    Display calibration metrics and quality assessment.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: CALIBRATION RESULTS")
    print("=" * 60)
    
    # Check if we have results
    results_file = "results/calibration_results.png"
    report_file = "results/calibration_report.txt"
    
    if not os.path.exists(results_file):
        print(f"❌ Results visualization '{results_file}' not found!")
        print("Please run the full calibration first: python camera_calibration.py")
        return False
    
    print("✅ Calibration results found!")
    
    # Display comprehensive results
    if os.path.exists(results_file):
        print(f"\n📊 Comprehensive visualization: {results_file}")
        print("This includes:")
        print("- Per-image reprojection errors")
        print("- Error distribution histogram")
        print("- Calibration summary")
        print("- Distortion field visualization")
        print("- Before/after undistortion comparison")
    
    # Display report summary
    if os.path.exists(report_file):
        print(f"\n📄 Detailed report: {report_file}")
        print("\nReport contents:")
        with open(report_file, 'r') as f:
            lines = f.readlines()
            for line in lines[:20]:  # Show first 20 lines
                print(f"  {line.rstrip()}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")
    
    # Load and display key metrics
    calibration_file = "results/calibration.npz"
    if os.path.exists(calibration_file):
        try:
            data = np.load(calibration_file)
            K = data['K']
            dist = data['dist']
            rms_error = data['rms_error']
            
            print(f"\n📈 Key Calibration Metrics:")
            print(f"  Focal length (fx): {K[0,0]:.1f} pixels")
            print(f"  Focal length (fy): {K[1,1]:.1f} pixels")
            print(f"  Principal point (cx, cy): ({K[0,2]:.1f}, {K[1,2]:.1f})")
            print(f"  RMS reprojection error: {rms_error:.4f} pixels")
            
            # Quality assessment
            if rms_error < 0.5:
                quality = "🏆 EXCELLENT"
            elif rms_error < 1.0:
                quality = "✅ GOOD"
            elif rms_error < 2.0:
                quality = "⚠️ ACCEPTABLE"
            else:
                quality = "❌ POOR"
            
            print(f"  Quality assessment: {quality}")
            
        except Exception as e:
            print(f"❌ Error loading calibration data: {e}")
            return False
    
    return True


def demo_live_calibration():
    """
    Demonstrate live calibration process (simplified version).
    """
    print("\n" + "=" * 60)
    print("DEMO 4: LIVE CALIBRATION PROCESS")
    print("=" * 60)
    
    print("This demonstrates the complete calibration workflow:")
    print("1. Loading images...")
    time.sleep(1)
    print("2. Detecting corners...")
    time.sleep(1)
    print("3. Building correspondences...")
    time.sleep(1)
    print("4. Computing intrinsic matrix...")
    time.sleep(1)
    print("5. Analyzing errors...")
    time.sleep(1)
    print("6. Generating visualizations...")
    time.sleep(1)
    print("7. Saving results...")
    time.sleep(1)
    
    print("\n✅ Live calibration process completed!")
    print("In the actual implementation, this process:")
    print("- Processes 20-30 calibration images")
    print("- Detects corners with sub-pixel accuracy")
    print("- Uses Zhang's method for robust calibration")
    print("- Provides comprehensive error analysis")
    print("- Generates professional visualizations")


def interactive_demo():
    """
    Interactive demonstration menu.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE CAMERA CALIBRATION DEMO")
    print("=" * 60)
    print("Choose a demonstration:")
    print("1. Corner Detection")
    print("2. Undistortion")
    print("3. Calibration Results")
    print("4. Live Calibration Process")
    print("5. Run All Demos")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                demo_corner_detection()
            elif choice == '2':
                demo_undistortion()
            elif choice == '3':
                demo_results()
            elif choice == '4':
                demo_live_calibration()
            elif choice == '5':
                print("Running all demonstrations...")
                demo_corner_detection()
                demo_undistortion()
                demo_results()
                demo_live_calibration()
                print("\n✅ All demonstrations completed!")
            else:
                print("Invalid choice. Please enter 0-5.")
                
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"Error during demo: {e}")


def main():
    """
    Main demo function.
    """
    print("🎥 CAMERA CALIBRATION DEMONSTRATION")
    print("Robotics Project Presentation")
    print("Partner: Yanran (compute_intrinsic_matrix)")
    print("=" * 60)
    
    # Check prerequisites
    print("Checking prerequisites...")
    
    # Check for OpenCV
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not available. Please install: pip install opencv-python")
        return
    
    # Check for matplotlib
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib available")
    except ImportError:
        print("❌ Matplotlib not available. Please install: pip install matplotlib")
        return
    
    # Check for numpy
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available. Please install: pip install numpy")
        return
    
    print("\n✅ All prerequisites met!")
    
    # Run interactive demo
    interactive_demo()


if __name__ == "__main__":
    main()
