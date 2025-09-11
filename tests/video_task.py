"""
Video Keypoint Tracking Script
==============================

This script tracks keypoints across all frames in a video using optical flow.
Uses the reference image and keypoint labels to track movement throughout the video.
"""

import os
import cv2
import json
import numpy as np
import sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from core.keypoint_tracker import KeypointTracker
from core.utils import load_keypoints, resize_keypoints, get_project_paths


class VideoKeypointTracker:
    """Track keypoints across all frames in a video."""
    
    def __init__(self, server_url="http://msraig-ubuntu-3:5000"):
        """Initialize the video keypoint tracker.
        
        Args:
            server_url: URL of the FlowFormer server
        """
        self.tracker = KeypointTracker(server_url)
        self.paths = get_project_paths()
        
    def extract_frames_from_video(self, video_path, max_frames=None):
        """Extract frames from video.
        
        Args:
            video_path: Path to input video
            max_frames: Maximum number of frames to extract (None for all)
            
        Returns:
            list: List of numpy arrays (frames)
        """
        print(f"🎬 Extracting frames from video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        print(f"📹 Extracted {len(frames)} frames")
        return frames
    
    def track_keypoints_in_video(self, video_path, ref_img_path, keypoints_json_path, 
                                output_video_path, target_width=800, max_frames=None):
        """Track keypoints throughout a video.
        
        Args:
            video_path: Path to input video
            ref_img_path: Path to reference image
            keypoints_json_path: Path to keypoints JSON file
            output_video_path: Path for output video
            target_width: Target width for processing (None to keep original)
            max_frames: Maximum frames to process (None for all)
            
        Returns:
            dict: Tracking results
        """
        try:
            print("🎯 Starting video keypoint tracking...")
            print(f"   Input video: {video_path}")
            print(f"   Reference image: {ref_img_path}")
            print(f"   Keypoints file: {keypoints_json_path}")
            print(f"   Output video: {output_video_path}")
            
            # Load keypoints
            print("📍 Loading keypoints...")
            keypoints, original_size = load_keypoints(keypoints_json_path)
            print(f"   Loaded {len(keypoints)} keypoints")
            print(f"   Original image size: {original_size}")
            
            # Load reference image
            print("🖼️  Loading reference image...")
            ref_img_pil = Image.open(ref_img_path)
            
            # Resize reference image if needed
            if target_width is not None:
                orig_w, orig_h = ref_img_pil.size
                aspect_ratio = orig_h / orig_w
                target_height = int(target_width * aspect_ratio)
                ref_img_pil = ref_img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
                print(f"   Reference image resized to: {ref_img_pil.size}")
            
            ref_img = np.array(ref_img_pil)
            
            # Resize keypoints to match reference image size
            print("🔄 Resizing keypoints...")
            resized_keypoints = resize_keypoints(keypoints, original_size, ref_img_pil.size)
            
            # Extract frames from video
            frames = self.extract_frames_from_video(video_path, max_frames)
            
            if len(frames) == 0:
                raise ValueError("No frames extracted from video")
            
            # Resize frames to match reference image size if needed
            if target_width is not None:
                print("🔄 Resizing video frames...")
                resized_frames = []
                for i, frame in enumerate(frames):
                    frame_pil = Image.fromarray(frame)
                    frame_resized = frame_pil.resize(ref_img_pil.size, Image.Resampling.LANCZOS)
                    resized_frames.append(np.array(frame_resized))
                    if (i + 1) % 10 == 0:
                        print(f"Resized {i + 1}/{len(frames)} frames")
                frames = resized_frames
            
            # Track keypoints across all frames
            print("🎯 Tracking keypoints across frames...")
            all_tracked_keypoints = []
            
            for i, frame in enumerate(frames):
                print(f"Processing frame {i + 1}/{len(frames)}")
                
                # Compute optical flow from reference to current frame
                flow = self.tracker.client.compute_flow(ref_img, frame)
                
                # Track keypoints using flow
                tracked_keypoints = self.tracker.track_keypoints_with_flow(resized_keypoints, flow)
                all_tracked_keypoints.append(tracked_keypoints)
            
            # Create output video with tracked keypoints
            print("🎬 Creating output video...")
            self.create_tracking_video(frames, resized_keypoints, all_tracked_keypoints, 
                                     output_video_path, ref_img)
            
            # Save tracking data
            output_dir = os.path.dirname(output_video_path)
            tracking_data_path = os.path.join(output_dir, "video_tracking_data.json")
            self.save_tracking_data(all_tracked_keypoints, tracking_data_path)
            
            print("✅ Video keypoint tracking completed successfully!")
            print(f"   Output video: {output_video_path}")
            print(f"   Tracking data: {tracking_data_path}")
            
            return {
                'success': True,
                'output_video_path': output_video_path,
                'tracking_data_path': tracking_data_path,
                'frames_processed': len(frames),
                'keypoints_count': len(keypoints)
            }
            
        except Exception as e:
            print(f"❌ Error during video tracking: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_tracking_video(self, frames, original_keypoints, all_tracked_keypoints, 
                            output_path, ref_img):
        """Create video showing keypoint tracking results.
        
        Args:
            frames: List of video frames
            original_keypoints: Original keypoints from reference
            all_tracked_keypoints: Tracked keypoints for each frame
            output_path: Output video path
            ref_img: Reference image
        """
        if len(frames) == 0:
            raise ValueError("No frames to process")
        
        # Get video properties
        height, width = frames[0].shape[:2]
        fps = 30  # Default FPS
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        # Colors for keypoints
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        for i, (frame, tracked_kps) in enumerate(zip(frames, all_tracked_keypoints)):
            # Create side-by-side visualization
            vis_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
            
            # Left side: reference image with original keypoints
            ref_frame = ref_img.copy()
            for j, kp in enumerate(original_keypoints):
                color = colors[j % len(colors)]
                cv2.circle(ref_frame, (int(kp['x']), int(kp['y'])), 4, color, -1)
                cv2.circle(ref_frame, (int(kp['x']), int(kp['y'])), 6, (255, 255, 255), 2)
            
            vis_frame[:, :width] = ref_frame
            
            # Right side: current frame with tracked keypoints
            current_frame = frame.copy()
            for j, kp in enumerate(tracked_kps):
                color = colors[j % len(colors)]
                
                # Draw tracked keypoint
                cv2.circle(current_frame, (int(kp['new_x']), int(kp['new_y'])), 4, color, -1)
                cv2.circle(current_frame, (int(kp['new_x']), int(kp['new_y'])), 6, (255, 255, 255), 2)
            
            vis_frame[:, width:] = current_frame
            
            # Add frame number
            cv2.putText(vis_frame, f"Frame: {i + 1}/{len(frames)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add titles
            cv2.putText(vis_frame, "Reference", (width // 4, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis_frame, "Current Frame", (width + width // 4, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Convert RGB to BGR for OpenCV
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            out.write(vis_frame_bgr)
        
        out.release()
        print(f"✅ Video saved with {len(frames)} frames")
    
    def save_tracking_data(self, all_tracked_keypoints, output_path):
        """Save tracking data to JSON file.
        
        Args:
            all_tracked_keypoints: List of tracked keypoints for each frame
            output_path: Output JSON file path
        """
        # Convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        tracking_data = {
            "total_frames": len(all_tracked_keypoints),
            "keypoint_count": len(all_tracked_keypoints[0]) if all_tracked_keypoints else 0,
            "frame_data": []
        }
        
        for i, frame_keypoints in enumerate(all_tracked_keypoints):
            frame_data = {
                "frame_number": i + 1,
                "keypoints": convert_numpy_types(frame_keypoints)
            }
            tracking_data["frame_data"].append(frame_data)
        
        with open(output_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"✅ Tracking data saved: {output_path}")


def main():
    """Main function to run video keypoint tracking."""
    # Initialize paths
    paths = get_project_paths()
    
    # Input files
    video_path = os.path.join(paths['test_data'], 'test_video.mp4')
    ref_img_path = os.path.join(paths['test_data'], 'test_ref.jpg')
    keypoints_json_path = os.path.join(paths['test_data'], 'test_label.json')
    
    # Output files
    output_dir = paths['output']
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'tracked_keypoints_video.mp4')
    
    # Check if input files exist
    for file_path, name in [(video_path, 'Video'), (ref_img_path, 'Reference image'), 
                           (keypoints_json_path, 'Keypoints JSON')]:
        if not os.path.exists(file_path):
            print(f"❌ {name} not found: {file_path}")
            return False
    
    # Initialize tracker
    print("🚀 Initializing video keypoint tracker...")
    try:
        tracker = VideoKeypointTracker()
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        print("   Make sure FlowFormer server is running")
        return False
    
    # Run tracking
    result = tracker.track_keypoints_in_video(
        video_path=video_path,
        ref_img_path=ref_img_path,
        keypoints_json_path=keypoints_json_path,
        output_video_path=output_video_path,
        target_width=800,  # Resize for faster processing
        max_frames=None  # Process all frames (set to number for testing)
    )
    
    if result['success']:
        print("\n🎉 Video keypoint tracking completed successfully!")
        print(f"   Frames processed: {result['frames_processed']}")
        print(f"   Keypoints tracked: {result['keypoints_count']}")
        print(f"   Output video: {result['output_video_path']}")
        print(f"   Tracking data: {result['tracking_data_path']}")
        return True
    else:
        print(f"\n❌ Video tracking failed: {result['error']}")
        return False


if __name__ == "__main__":
    main()
