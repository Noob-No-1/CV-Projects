#!/usr/bin/env python3
"""
iPhone Camera Sender Script
This script captures video from iPhone camera and sends it to the laptop via WiFi.

Note: This is a simplified version. For a real iPhone app, you would need:
1. A native iOS app or web app
2. Proper camera access permissions
3. Better error handling and reconnection logic

For this demo, you can run this script on a computer with camera access
and modify the target IP to simulate iPhone functionality.
"""

import cv2
import socket
import struct
import time
import argparse
import sys

class CameraSender:
    def __init__(self, target_ip: str, port: int = 8080):
        """
        Initialize camera sender.
        
        Args:
            target_ip: IP address of the target laptop
            port: Port number for connection
        """
        self.target_ip = target_ip
        self.port = port
        self.socket = None
        self.cap = None
        self.connected = False
        
    def connect_to_laptop(self):
        """Connect to the laptop."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.target_ip, self.port))
            self.connected = True
            print(f"Connected to laptop at {self.target_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to laptop: {e}")
            return False
    
    def initialize_camera(self, camera_id: int = 0):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully")
        return True
    
    def send_frame(self, frame):
        """Send a single frame to the laptop."""
        if not self.connected or self.socket is None:
            return False
        
        try:
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, frame_data = cv2.imencode('.jpg', frame, encode_param)
            
            # Send frame size
            frame_size = len(frame_data)
            size_data = struct.pack('!I', frame_size)
            self.socket.send(size_data)
            
            # Send frame data
            self.socket.send(frame_data.tobytes())
            
            return True
            
        except Exception as e:
            print(f"Error sending frame: {e}")
            self.connected = False
            return False
    
    def run(self, camera_id: int = 0):
        """Run the camera sender."""
        print("iPhone Camera Sender")
        print(f"Target: {self.target_ip}:{self.port}")
        print("Press 'q' to quit")
        
        # Initialize camera
        if not self.initialize_camera(camera_id):
            return
        
        # Connect to laptop
        if not self.connect_to_laptop():
            return
        
        frame_count = 0
        start_time = time.time()
        
        while self.connected:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Send frame
            if not self.send_frame(frame):
                print("Failed to send frame, attempting to reconnect...")
                time.sleep(1)
                if not self.connect_to_laptop():
                    break
                continue
            
            frame_count += 1
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame locally (optional)
            cv2.imshow('iPhone Camera Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if self.cap:
            self.cap.release()
        if self.socket:
            self.socket.close()
        cv2.destroyAllWindows()
        
        print("Camera sender stopped")


def find_laptop_ip():
    """Find the laptop's IP address on the local network."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description='iPhone Camera Sender')
    parser.add_argument('--target-ip', type=str, default=None,
                       help='Target laptop IP address')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port number (default: 8080)')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    # Determine target IP
    if args.target_ip is None:
        target_ip = input(f"Enter laptop IP address (or press Enter for {find_laptop_ip()}): ").strip()
        if not target_ip:
            target_ip = find_laptop_ip()
    else:
        target_ip = args.target_ip
    
    # Create and run camera sender
    sender = CameraSender(target_ip, args.port)
    sender.run(args.camera_id)


if __name__ == "__main__":
    main()
