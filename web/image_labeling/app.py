#!/usr/bin/env python3
"""
Image Labeling Service Wrapper
=============================

Wrapper service that launches the existing ImageLabelingWeb interface
as part of the Robot Vision Services architecture.
"""

import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageLabelingWrapper:
    def __init__(self, port=8003):
        self.port = port
        self.project_root = Path(project_root)
        self.labeling_web_dir = self.project_root / "ThirdParty" / "ImageLabelingWeb"
        self.launcher_script = self.labeling_web_dir / "launch_server.py"
        self.process = None
        
    def start_server(self):
        """Start the existing ImageLabelingWeb server."""
        if not self.launcher_script.exists():
            logger.error(f"❌ ImageLabelingWeb launcher not found: {self.launcher_script}")
            return False
            
        logger.info("🚀 Starting Image Labeling Web Interface...")
        logger.info(f"📁 Location: {self.labeling_web_dir}")
        logger.info(f"🌐 Port: {self.port}")
        
        try:
            # Start the existing server with our port and no auto-browser
            self.process = subprocess.Popen([
                sys.executable, 
                str(self.launcher_script),
                "--port", str(self.port),
                "--host", "0.0.0.0",
                "--no-browser"
            ], cwd=str(self.labeling_web_dir))
            
            logger.info(f"✅ Image Labeling Web started with PID {self.process.pid}")
            logger.info(f"🎯 Access at: http://localhost:{self.port}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Image Labeling Web: {e}")
            return False
    
    def wait_for_server(self):
        """Wait for the server process to finish."""
        if not self.process:
            logger.warning("No server process to wait for")
            return
            
        try:
            logger.info("🔄 Image Labeling Web is running...")
            logger.info("⏸️  Press Ctrl+C to stop the service")
            
            # Wait for process to complete
            self.process.wait()
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Received stop signal, shutting down...")
            self.stop_server()
        except Exception as e:
            logger.error(f"Error waiting for server: {e}")
    
    def stop_server(self):
        """Stop the server process."""
        if self.process:
            try:
                logger.info("Stopping Image Labeling Web...")
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("✅ Image Labeling Web stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Force killing Image Labeling Web...")
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
        else:
            logger.warning("No server process to stop")

def main():
    """Main entry point."""
    print("🏷️ Image Labeling Service Wrapper")
    print("=" * 50)
    print("This service launches the existing ImageLabelingWeb interface")
    print("as part of the Robot Vision Services architecture.")
    print()
    
    # Create wrapper instance
    wrapper = ImageLabelingWrapper(port=8003)
    
    # Start the server
    if wrapper.start_server():
        print("🎉 Image Labeling Web Interface Ready!")
        print()
        print("📋 Features Available:")
        print("   - ✅ Interactive image upload and annotation")
        print("   - 🎯 Sub-pixel precision keypoint placement")
        print("   - 🔍 Zoom and pan functionality") 
        print("   - 💾 Export labels as JSON")
        print("   - �️ Drag to move keypoints, right-click to delete")
        print()
        print("🌐 Access URL: http://localhost:8003")
        print("🏠 Control Center: http://localhost:8000")
        print()
        
        # Wait for server
        wrapper.wait_for_server()
    else:
        print("❌ Failed to start Image Labeling Web Interface")
        sys.exit(1)

if __name__ == "__main__":
    main()