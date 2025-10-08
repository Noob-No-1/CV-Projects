# DIY Stereo Camera Setup

This project implements a stereo camera system using your laptop webcam and iPhone, following the workflow from [LearnOpenCV's stereo camera tutorial](https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/).

## Features

- **Individual Camera Calibration**: Calibrate each camera separately using chessboard patterns
- **Stereo Calibration**: Determine the relationship between the two cameras
- **Stereo Rectification**: Align stereo images for accurate depth computation
- **3D Anaglyph Generation**: Create 3D images viewable with red-cyan glasses
- **Real-time Processing**: Live 3D video capture and display

## Hardware Requirements

- Laptop with webcam
- iPhone with camera
- Chessboard calibration pattern (9x6 inner corners, print on A4 paper)
- Red-cyan 3D glasses (for viewing 3D anaglyph images)

## Software Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Print Calibration Pattern

Download and print a 9x6 chessboard pattern on A4 paper. Ensure the squares are clearly visible and the pattern is flat.

## Usage

### Step 1: Camera Calibration

1. **Calibrate Laptop Webcam (Left Camera)**:
   ```bash
   python DIY_StereoCamera.py
   # Choose option 1: Run calibration workflow
   # Follow prompts to calibrate left camera
   ```

2. **Calibrate iPhone Camera (Right Camera)**:
   - Run the iPhone camera sender script on your iPhone or a computer simulating the iPhone:
   ```bash
   python iphone_camera_sender.py --target-ip YOUR_LAPTOP_IP
   ```
   - Follow the calibration prompts in the main application

### Step 2: Stereo Calibration

The application will automatically perform stereo calibration using the individual camera parameters.

### Step 3: Real-time 3D Capture

```bash
python DIY_StereoCamera.py
# Choose option 2: Run real-time 3D capture
```

## Workflow Overview

Based on the [LearnOpenCV tutorial](https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/), the complete workflow is:

1. **Individual Camera Calibration**
   - Capture multiple images of chessboard pattern from different angles
   - Compute intrinsic parameters (camera matrix, distortion coefficients)
   - Save calibration data for each camera

2. **Stereo Calibration**
   - Use fixed intrinsic parameters from individual calibrations
   - Compute extrinsic parameters (rotation, translation between cameras)
   - Calculate Essential and Fundamental matrices

3. **Stereo Rectification**
   - Align stereo images to make corresponding points have equal Y-coordinates
   - Compute rectification maps for undistortion and alignment

4. **3D Anaglyph Creation**
   - Apply rectification maps to live video streams
   - Create anaglyph images by combining color channels
   - Display real-time 3D video

## Technical Implementation

### Camera Calibration
- Uses OpenCV's `calibrateCamera()` function
- Chessboard corner detection with sub-pixel accuracy
- Saves calibration parameters in JSON format

### Stereo Calibration
- Uses `stereoCalibrate()` with `CALIB_FIX_INTRINSIC` flag
- Computes rotation matrix R and translation vector T
- Calculates Essential matrix E and Fundamental matrix F

### Stereo Rectification
- Uses `stereoRectify()` to compute rectification transforms
- Creates undistortion maps with `initUndistortRectifyMap()`
- Ensures corresponding points have equal Y-coordinates

### 3D Anaglyph Generation
- Red channel from left camera
- Green and Blue channels from right camera
- Compatible with red-cyan 3D glasses

## Network Communication

The iPhone camera feed is transmitted via TCP socket connection:
- iPhone sends JPEG-encoded frames
- Frame size is transmitted first (4 bytes)
- Frame data follows as binary stream
- Automatic reconnection on connection loss

## Troubleshooting

### Camera Connection Issues
- Ensure both devices are on the same WiFi network
- Check firewall settings allow connections on port 8080
- Verify camera permissions are granted

### Calibration Problems
- Use high-quality chessboard pattern
- Ensure good lighting and contrast
- Capture from various angles and distances
- Need at least 10-15 good calibration images

### Poor 3D Effect
- Verify cameras are properly aligned
- Check stereo calibration accuracy
- Ensure synchronized capture timing
- Use proper red-cyan glasses

## File Structure

```
CV-Projects/
├── DIY_StereoCamera.py          # Main stereo camera application
├── iphone_camera_sender.py      # iPhone camera feed sender
├── 3D_Projection.py             # 3D projection viewer (separate project)
├── requirements.txt             # Python dependencies
└── STEREO_CAMERA_README.md      # This documentation
```

## Future Enhancements

- Native iOS app for iPhone camera control
- Disparity map computation and depth estimation
- 3D point cloud generation
- Improved synchronization between cameras
- GUI interface for easier operation

## References

- [LearnOpenCV Stereo Camera Tutorial](https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/)
- [OpenCV Camera Calibration Documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Stereo Vision Documentation](https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html)
