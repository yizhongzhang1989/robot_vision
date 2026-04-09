#!/usr/bin/env python3
"""
Quick Start Script for Robot Vision Services
============================================

Simple script to start all services quickly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the service manager
try:
    from scripts.manage_services import ServiceManager
    
    def main():
        print("🤖 Robot Vision Services - Quick Start")
        print("=" * 50)
        
        manager = ServiceManager()
        
        if manager.start_all_services():
            # Get ports from config
            gateway_port = manager.config.get('gateway', {}).get('port', 8000)
            
            print("\n🎉 All services started successfully!")
            print("\n📋 Service URLs:")
            print(f"   🤖 Control Center: http://localhost:{gateway_port}")
            for svc_name, svc_info in manager.config.get('services', {}).items():
                print(f"   🔹 {svc_info['name']}: http://localhost:{svc_info['port']}")
            print("\n💡 Tip: Access the Control Center to manage all services")
            
            # Wait for user input to keep services running
            try:
                print("\n⏸️  Press Ctrl+C to stop all services")
                manager.wait_for_services()
            except KeyboardInterrupt:
                print("\n🛑 Stopping services...")
                manager.stop_all_services()
                print("👋 Services stopped. Goodbye!")
        else:
            print("❌ Failed to start services")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")

if __name__ == "__main__":
    main()