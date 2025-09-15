
import cv2
import numpy as np
import threading
import time
import os
import sys
import json
from datetime import datetime
from PIL import Image
from queue import Queue

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from core.keypoint_tracker import KeypointTracker
from core.utils import load_keypoints, resize_keypoints, get_project_paths

class IPCameraStream:
    def __init__(self, rtsp_url):
        """
       initialize the camera stream
        Args:
            rtsp_url: RTSP stream URL
        """
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.width = None
        self.height = None
        self.fps = None
        
    def start(self):
        """Start the video stream"""
        if self.running:
            print("Stream is already running")
            return True
            
        print(f"Connecting to: {self.rtsp_url}")
        
        try:
            # Create VideoCapture object
            self.cap = cv2.VideoCapture(self.rtsp_url)

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate
            
            # Check if opened successfully
            if not self.cap.isOpened():
                raise Exception("Unable to open RTSP stream")
            
            # Get stream information
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Stream information: {self.width}x{self.height} @ {self.fps}fps")
            
            # read a test frame
            ret, test_frame = self.cap.read()
            if not ret:
                raise Exception("Unable to read video frame")
            
            # start the frame update thread
            self.running = True
            self.thread = threading.Thread(target=self._update_frames, daemon=True)
            self.thread.start()
            
            print("Started video stream successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start video stream: {e}")
            self.running = False
            if self.cap:
                self.cap.release()
            return False
    
    def _update_frames(self):
        """Update frame data in the background thread"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Successfully read frame
                    with self.lock:
                        self.frame = frame.copy()
                    consecutive_failures = 0
                else:
                    # Failed to read frame
                    consecutive_failures += 1
                    print(f"Failed to read frame ({consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        print("Consecutive frame read failures exceeded, stopping stream")
                        break
                
                time.sleep(0.01) # Slight delay to reduce CPU usage
                
            except Exception as e:
                if self.running:
                    print(f"Error reading frame: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                time.sleep(0.1)
        
        print("Exiting frame update thread")
    
    def get_frame(self):
        """Get the current frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop the video stream"""
        if not self.running:
            return
            
        print("Stop the stream...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
            
        if self.cap:
            self.cap.release()
            
        with self.lock:
            self.frame = None
            
        print("Stream stopped")
    
    def is_running(self):
        """Check if the stream is running"""
        return self.running
    
    def get_stream_info(self):
        """Get stream information"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'url': self.rtsp_url
        }


class IPCameraFlow:
    """IP Camera with optical flow detection."""
    
    def __init__(self, rtsp_url, server_url="http://msraig-ubuntu-3:5000"):
        """Initialize the IP camera flow tracker.
        
        Args:
            rtsp_url: RTSP stream URL
            server_url: URL of the FlowFormer server
        """
        self.rtsp_url = rtsp_url
        self.server_url = server_url
        self.tracker = KeypointTracker(server_url)
        self.paths = get_project_paths()
        
        # IP Camera setup
        self.camera = IPCameraStream(rtsp_url)
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
        
    def load_reference_and_keypoints(self, ref_img_path, keypoints_json_path):
        """Load reference image and keypoints from files.
        
        Args:
            ref_img_path: Path to reference image file
            keypoints_json_path: Path to keypoints JSON file
        """
        # Load reference image from file
        print(f"🖼️  Loading reference image: {ref_img_path}")
        ref_img_pil = Image.open(ref_img_path)
        
        # Resize to fixed target size
        ref_img_pil = ref_img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
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
        
    def capture_frames_thread(self):
        """Thread function to continuously capture frames from IP camera."""
        while self.is_running:
            frame = self.camera.get_frame()
            if frame is not None:
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
                time.sleep(0.01)  # Small delay if no frame available
                
    def draw_flow_vectors(self, image, flow, step=16):
        """Draw optical flow vectors on image."""
        h, w = flow.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        for (x1, y1), (x2, y2) in lines:
            cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
            
    def run_realtime_flow(self, use_keypoints=True, show_flow_vectors=False, save_results=False):
        """Run real-time optical flow computation.
        
        Args:
            use_keypoints: Whether to track keypoints
            show_flow_vectors: Whether to show optical flow vectors
            save_results: Whether to save results to files
        """
        print("🚀 Starting IP camera real-time optical flow...")
        print("Press 'q' to quit, 's' to save current frame")
        print(f"📍 Using reference image with {len(self.keypoints) if self.keypoints else 0} keypoints")
        
        # Start IP camera
        if not self.camera.start():
            print("❌ Failed to start IP camera")
            return
            
        self.is_running = True
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self.capture_frames_thread)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Wait for first frame
        time.sleep(2)
        
        # Main processing loop
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
                            # Ensure correct data type
                            if self.reference_frame.dtype != np.uint8:
                                self.reference_frame = self.reference_frame.astype(np.uint8)
                            if current_frame.dtype != np.uint8:
                                current_frame = current_frame.astype(np.uint8)
                            
                            # Compute flow from reference to current frame
                            flow = self.tracker.client.compute_flow(self.reference_frame, current_frame)
                            
                            # Create side-by-side visualization
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
                            cv2.putText(vis_frame, "IP Camera Live", (width + width // 4, 30),
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
                    cv2.imshow('IP Camera Optical Flow', vis_frame_bgr)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_results:
                        # Save current frame
                        timestamp = int(time.time())
                        save_path = os.path.join("test_data", f"ip_camera_flow_{timestamp}.jpg")
                        cv2.imwrite(save_path, vis_frame_bgr)
                        print(f"💾 Saved frame: {save_path}")
                
                else:
                    time.sleep(0.01)  # Small delay if queue is empty
                    
        except KeyboardInterrupt:
            print("\n🛑 User interrupted")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the IP camera flow tracker."""
        print("🛑 Stopping IP camera flow...")
        self.is_running = False
        self.camera.stop()
        cv2.destroyAllWindows()
        print("✅ IP camera flow stopped")


def run_optical_flow(rtsp_url):
    """Run IP camera with optical flow tracking."""
    print("🚀 Starting optical flow mode...")
    print("=" * 60)
    
    # Initialize IP camera flow tracker
    flow_tracker = IPCameraFlow(rtsp_url)
    
    # Load reference image and keypoints
    ref_img_path = os.path.join("test_data", "test_image_ip.jpg")
    keypoints_path = os.path.join("test_data", "test_label_ip.json")
    
    if not os.path.exists(ref_img_path):
        print(f"❌ Reference image not found: {ref_img_path}")
        return
        
    if not os.path.exists(keypoints_path):
        print(f"❌ Keypoints file not found: {keypoints_path}")
        return
    
    try:
        # Load reference data
        flow_tracker.load_reference_and_keypoints(ref_img_path, keypoints_path)
        
        # Run optical flow tracking
        flow_tracker.run_realtime_flow(
            use_keypoints=True, 
            show_flow_vectors=False, 
            save_results=True
        )
        
    except Exception as e:
        print(f"❌ Error in optical flow mode: {e}")
    finally:
        flow_tracker.stop()


def run_normal_stream(rtsp_url, resolution):
    """Run normal IP camera stream."""
    print(f"Using {resolution} resolution: {rtsp_url}")
    print("=" * 60)

    camera = IPCameraStream(rtsp_url)
    
    try:

        if not camera.start():
            print("Cannot start camera stream")
            return
        
        print("Waiting for video stream to stabilize...")
        ready_count = 0
        for i in range(100):  # 10 seconds max wait
            if camera.get_frame() is not None:
                ready_count += 1
                if ready_count >= 5:  # 5 consecutive frames to be stable
                    break
            else:
                ready_count = 0
            time.sleep(0.1)
        
        if camera.get_frame() is None:
            print("Cannot get stable video frame")
            return
        
        info = camera.get_stream_info()
        print(f"Stream information: {info['width']}x{info['height']} @ {info['fps']}fps")
        print("=" * 60)
        print("Video stream is ready!")
        print("Operations:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'i' to show stream information")
        print("  - Press 'r' to reconnect")
        print("=" * 60)
        
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                # resize frame to 600x800
                frame_display = cv2.resize(frame, (800, 600))
                
                # get camera actual FPS
                actual_fps = camera.get_stream_info()['fps']
                
                # add overlay information
                cv2.putText(frame_display, f"FPS: {actual_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # add resolution info (show both original and display size)
                h_orig, w_orig = frame.shape[:2]
                h_disp, w_disp = frame_display.shape[:2]
                cv2.putText(frame_display, f"Orig: {w_orig}x{h_orig}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame_display, f"Disp: {w_disp}x{h_disp}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # show the frame
                cv2.imshow('IP Camera Stream (OpenCV)', frame_display)
                
                # check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_screenshot(frame, resolution)
                elif key == ord('i'):
                    print_stream_info(camera)
                elif key == ord('r'):
                    print("Reconnecting stream...")
                    camera.stop()
                    time.sleep(2)
                    if not camera.start():
                        print("Reconnection failed")
                        break
            else:
                print(".", end="", flush=True)
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nUser interrupted the program")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        # release resources
        camera.stop()
        cv2.destroyAllWindows()
        print("Program ended")


def save_screenshot(frame, resolution):
    """Save screenshot"""
    if frame is None:
        print("Cannot save screenshot: Current frame is empty")
        return
    
    try:
        # create screenshot directory
        screenshot_dir = "test_data"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        
        # create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{resolution}_{timestamp}.jpg"
        filepath = os.path.join(screenshot_dir, filename)
        
        # save the image
        success = cv2.imwrite(filepath, frame)
        if success:
            print(f"\nScreenshot saved: {filepath}")
        else:
            print(f"\nFailed to save screenshot: Unable to write file")
        
    except Exception as e:
        print(f"\nFailed to save screenshot: {e}")


def print_stream_info(camera):
    """Print stream information"""
    info = camera.get_stream_info()
    print("\n" + "=" * 40)
    print("Stream Information:")
    print(f"  URL: {info['url']}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']}")
    print(f"  Status: {'Running' if camera.is_running() else 'Stopped'}")
    print("=" * 40)

def main():
    """Main function with mode selection."""
    RTSP_URLS = "rtsp://admin:123456@192.168.1.102/stream0"
    
    print("=" * 60)
    print("IP Camera Stream Application")
    print("=" * 60)
    print("Select mode:")
    print("  1. Normal stream (basic display)")
    print("  2. Optical flow tracking (with keypoints)")
    print("=" * 60)
    
    while True:
        try:
            mode = input("Enter mode (1 or 2): ").strip()
            if mode in ['1', '2']:
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    resolution = "1080p"
    rtsp_url = RTSP_URLS
    
    if mode == '1':
        # Normal stream mode
        run_normal_stream(rtsp_url, resolution)
    elif mode == '2':
        # Optical flow mode
        run_optical_flow(rtsp_url)

if __name__ == "__main__":

    print("\nStarting IP camera stream...") 
    main()