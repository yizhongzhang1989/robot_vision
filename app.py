#!/usr/bin/env python3
"""
Robot Vision Services - Legacy Entry Point
==========================================

⚠️  NOTICE: This service has been restructured into a microservices architecture.

The keypoint tracking functionality has been moved to:
- web/keypoint_tracking/app.py (port 8001)

New Robot Vision Services Architecture:
- Gateway/Control Center: web/gateway/app.py (port 8000)  
- Keypoint Tracking Service: web/keypoint_tracking/app.py (port 8001)
- Image Labeling Service: web/image_labeling/app.py (port 8003)

Quick Start:
  python start_services.py

Or use the service manager:
  python scripts/manage_services.py start --wait

This legacy entry point redirects to the new architecture.
"""

import os
import sys
import webbrowser
import subprocess
import time
from pathlib import Path

def show_migration_info():
    """Show information about the new services architecture."""
    print("🤖 Robot Vision Services - Architecture Migration")
    print("=" * 60)
    print("⚠️  This application has been restructured into microservices!")
    print()
    print("🏗️  New Architecture:")
    print("   🤖 Control Center (Gateway): http://localhost:8000")
    print("   🎯 Keypoint Tracking:        http://localhost:8001") 
    print("   🏷️ Image Labeling:           http://localhost:8003")
    print()
    print("🚀 Quick Start Options:")
    print("   1. python start_services.py")
    print("   2. python scripts/manage_services.py start --wait")
    print()
    print("📖 Individual Services:")
    print("   • python web/gateway/app.py")
    print("   • python web/keypoint_tracking/app.py") 
    print("   • python web/image_labeling/app.py")
    print()
    
def start_new_services():
    """Attempt to start the new services architecture."""
    project_root = Path(__file__).parent
    
    # Check if the quick start script exists
    quick_start = project_root / "start_services.py"
    if quick_start.exists():
        print("🚀 Starting new Robot Vision Services...")
        try:
            subprocess.run([sys.executable, str(quick_start)], cwd=str(project_root))
        except KeyboardInterrupt:
            print("\n👋 Services stopped")
        except Exception as e:
            print(f"❌ Error starting services: {e}")
            print("Please run manually: python start_services.py")
    else:
        print("❌ New services not found. Please check the installation.")

def main():
    """Main entry point with migration guidance."""
    show_migration_info()
    
    # Ask user what they want to do
    while True:
        print("Choose an option:")
        print("  [1] Start new Robot Vision Services")
        print("  [2] Show migration info again") 
        print("  [3] Exit")
        print()
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            start_new_services()
            break
        elif choice == "2":
            print()
            show_migration_info()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice, please enter 1, 2, or 3")
            print()

if __name__ == "__main__":
    main()