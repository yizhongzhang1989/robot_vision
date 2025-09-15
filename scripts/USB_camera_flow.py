"""
Realtime Camera Optical Flow Script
===================================

This script uses a USB camera to compute optical flow in real-time.
It captures frames from the camera and computes optical flow between consecutive frames
or between the current frame and a reference frame.
"""

import os
import cv2
import json
import numpy as np
import sys
import time
import threading
from queue import Queue
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from core.keypoint_tracker import KeypointTracker
from core.utils import load_keypoints, resize_keypoints, get_project_paths


class RealtimeCameraFlow:
    """Real-time camera optical flow computation."""
    
    def __init__(self, server_url="http://msraig-ubuntu-3:5000", camera_id=0):
        """Initialize the realtime camera flow tracker.
        
        Args:
            server_url: URL of the FlowFormer server
            camera_id: Camera device ID (usually 0 for default camera)
        """
        self.server_url = server_url
        self.camera_id = camera_id
        self.tracker = KeypointTracker(server_url)
        self.paths = get_project_paths()
        
        # Camera setup
        self.cap = None
        self.frame_queue = Queue(maxsize=10)
        self.is_running = False
        
        # Reference frame and keypoints
        self.reference_frame = None
        self.keypoints = None
        self.original_keypoints = None
        
        # Display settings - fixed size to match reference image
        self.target_size = (800, 600)  # (width, height)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def initialize_camera(self):
        """Initialize the camera."""
        print(f"📷 Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {width}x{height} @ {fps} FPS")
        return True
        
    def capture_frames_thread(self):
        """Thread function to continuously capture frames."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to fixed target size
                frame_pil = Image.fromarray(frame_rgb)
                frame_resized = frame_pil.resize(self.target_size, Image.Resampling.LANCZOS)
                frame_rgb = np.array(frame_resized)
                
                # Add to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_rgb)
            else:
                print("❌ Failed to read frame from camera")
                break
                
    def load_reference_and_keypoints(self, ref_img_path, keypoints_json_path):
        """Load reference image and keypoints from files (like video_task.py).
        
        Args:
            ref_img_path: Path to reference image file
            keypoints_json_path: Path to keypoints JSON file
        """
        # Load reference image from file
        print(f"🖼️  Loading reference image: {ref_img_path}")
        ref_img_pil = Image.open(ref_img_path)
        
        # Resize to fixed target size
        ref_img_pil = ref_img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
        # print(f"📏 Reference image resized to: {ref_img_pil.size}")
        
        self.reference_frame = np.array(ref_img_pil)
        print(f"✅ Reference image loaded: {self.reference_frame.shape}")
        
        # Load keypoints
        print(f"📍 Loading keypoints: {keypoints_json_path}")
        keypoints, original_size = load_keypoints(keypoints_json_path)
        print(f"📍 Original keypoints image size: {original_size}")
        print(f"📍 Current reference image size: {ref_img_pil.size}")
        
        # Resize keypoints to match reference image size
        self.keypoints = resize_keypoints(keypoints, original_size, ref_img_pil.size)
        self.original_keypoints = keypoints.copy()
        print(f"✅ Loaded and resized {len(self.keypoints)} keypoints")
        
        return True
        
    def run_realtime_flow(self, use_keypoints=True, show_flow_vectors=False, save_results=False):
        """Run real-time optical flow computation.
        
        Args:
            use_keypoints: Whether to track keypoints
            show_flow_vectors: Whether to show optical flow vectors
            save_results: Whether to save results to files
        """
        print("🚀 Starting real-time optical flow...")
        print("Press 'q' to quit, 's' to save current frame")
        print(f"📍 Using fixed reference image with {len(self.keypoints) if self.keypoints else 0} keypoints")
        
        self.is_running = True
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self.capture_frames_thread)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Wait for first frame
        time.sleep(1)
        
        # Main processing loop
        prev_frame = None
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        try:
            while self.is_running:
                if not self.frame_queue.empty():
                    current_frame = self.frame_queue.get()
                    
                    # Update FPS counter
                    self.fps_counter += 1
                    if self.fps_counter % 30 == 0:
                        elapsed = time.time() - self.fps_start_time
                        fps = 30 / elapsed
                        print(f"📊 Processing FPS: {fps:.1f}")
                        self.fps_start_time = time.time()
                    
                    # Compute optical flow
                    if self.reference_frame is not None:
                        try:
                            # Ensure correct data type (both images should already be same size)
                            if self.reference_frame.dtype != np.uint8:
                                self.reference_frame = self.reference_frame.astype(np.uint8)
                            if current_frame.dtype != np.uint8:
                                current_frame = current_frame.astype(np.uint8)
                            
                            # Compute flow from reference to current frame
                            flow = self.tracker.client.compute_flow(self.reference_frame, current_frame)
                            
                            # Create side-by-side visualization (like video_task.py)
                            height, width = current_frame.shape[:2]
                            vis_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
                            
                            # Left side: reference image with original keypoints
                            ref_frame_display = self.reference_frame.copy()
                            if use_keypoints and self.keypoints is not None:
                                for i, kp in enumerate(self.keypoints):
                                    color = colors[i % len(colors)]
                                    cv2.circle(ref_frame_display, (int(kp['x']), int(kp['y'])), 4, color, -1)
                                    cv2.circle(ref_frame_display, (int(kp['x']), int(kp['y'])), 6, (255, 255, 255), 2)
                            
                            vis_frame[:, :width] = ref_frame_display
                            
                            # Right side: current frame with tracked keypoints
                            current_frame_display = current_frame.copy()
                            if use_keypoints and self.keypoints is not None:
                                tracked_keypoints = self.tracker.track_keypoints_with_flow(self.keypoints, flow)
                                
                                for i, kp in enumerate(tracked_keypoints):
                                    color = colors[i % len(colors)]
                                    # Draw tracked keypoint
                                    cv2.circle(current_frame_display, (int(kp['new_x']), int(kp['new_y'])), 4, color, -1)
                                    cv2.circle(current_frame_display, (int(kp['new_x']), int(kp['new_y'])), 6, (255, 255, 255), 2)
                            
                            # Show flow vectors on current frame if requested
                            if show_flow_vectors:
                                self.draw_flow_vectors(current_frame_display, flow)
                            
                            vis_frame[:, width:] = current_frame_display
                            
                            # Add titles and status
                            cv2.putText(vis_frame, "Reference", (width // 4, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(vis_frame, "Live Camera", (width + width // 4, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(vis_frame, f"Keypoints: {len(self.keypoints) if self.keypoints else 0}", (10, height - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(vis_frame, "Press 'q' to quit, 's' to save", (width, height - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                        except Exception as e:
                            print(f"⚠️  Flow computation error: {e}")
                            # Single frame with error message
                            vis_frame = current_frame.copy()
                            cv2.putText(vis_frame, f"Flow Error: {str(e)[:50]}...", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # No reference frame
                        vis_frame = current_frame.copy()
                        cv2.putText(vis_frame, "No reference frame loaded", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Convert RGB to BGR for display
                    vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                    
                    # Display frame
                    cv2.imshow('Realtime Optical Flow', vis_frame_bgr)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_results:
                        # Save current frame
                        timestamp = int(time.time())
                        output_path = os.path.join(self.paths['output'], f'camera_frame_{timestamp}.jpg')
                        vis_frame_pil = Image.fromarray(vis_frame)
                        vis_frame_pil.save(output_path)
                        print(f"💾 Frame saved: {output_path}")
                
                else:
                    time.sleep(0.01)  # Small delay when no frames available
                    
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        finally:
            self.cleanup()
    
    def draw_flow_vectors(self, image, flow, step=20):
        """Draw optical flow vectors on image.
        
        Args:
            image: Image to draw on
            flow: Optical flow field
            step: Step size for sampling flow vectors
        """
        h, w = flow.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        # Create line endpoints
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        # Draw lines
        for (x1, y1), (x2, y2) in lines:
            cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")


def main():
    """Main function to run realtime camera optical flow."""
    # Initialize paths
    paths = get_project_paths()
    
    # Setup output directory
    output_dir = paths['output']
    os.makedirs(output_dir, exist_ok=True)
    
    # Use predefined reference image and keypoints (like video_task.py)
    ref_img_path = os.path.join(paths['test_data'], 'test_image_realtime.JPG')
    keypoints_json_path = os.path.join(paths['test_data'], 'test_label_realtime.json')
    
    print("🚀 Initializing realtime camera flow tracker...")
    
    try:
        # Initialize tracker
        camera_flow = RealtimeCameraFlow(camera_id=0)  # Change camera_id if needed
        
        # Initialize camera
        camera_flow.initialize_camera()
        
        # Check if reference files exist
        for file_path, name in [(ref_img_path, 'Reference image'), (keypoints_json_path, 'Keypoints JSON')]:
            if not os.path.exists(file_path):
                print(f"❌ {name} not found: {file_path}")
                return False
        
        # Load reference image and keypoints (required)
        camera_flow.load_reference_and_keypoints(ref_img_path, keypoints_json_path)
        
        # Run real-time flow
        camera_flow.run_realtime_flow(
            use_keypoints=True,  # Always use keypoints
            show_flow_vectors=False,  # Set to True to show flow field
            save_results=True
        )
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("   Make sure:")
        print("   1. Camera is connected and accessible")
        print("   2. FlowFormer server is running")
        return False
    
    print("👋 Goodbye!")
    return True


if __name__ == "__main__":
    main()