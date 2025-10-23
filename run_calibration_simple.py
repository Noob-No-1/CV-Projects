"""
Simple Camera Calibration Runner
Runs calibration without the problematic visualization
"""

from camera_calibration import CameraCalibrator, save_report, create_undistorted_samples
import os

def main():
    """Run calibration with basic output only."""
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
        
        # Step 5: Generate report (without visualization)
        save_report(calibrator, calibrator.output_folder)
        
        # Step 6: Create undistorted samples
        create_undistorted_samples(calibrator, calibrator.output_folder)
        
        # Step 7: Save calibration data
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
