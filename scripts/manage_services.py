#!/usr/bin/env python3
"""
Service Management Script
========================

Utility script to start, stop, and manage Robot Vision Services.
"""

import os
import sys
import subprocess
import time
import signal
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ServiceManager:
    def __init__(self):
        self.project_root = project_root
        self.config_path = self.project_root / "config" / "services.yaml"
        self.config = self.load_config()
        self.running_processes = {}
        
    def load_config(self):
        """Load service configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return None
    
    def start_service(self, service_name):
        """Start a specific service."""
        if not self.config or 'services' not in self.config:
            print("❌ No services configured")
            return False
            
        if service_name not in self.config['services']:
            print(f"❌ Service '{service_name}' not found in configuration")
            return False
            
        service_info = self.config['services'][service_name]
        service_type = service_info.get('type', 'fastapi')
        
        print(f"🚀 Starting {service_info['name']} on port {service_info['port']}...")
        
        try:
            if service_type == 'static_web':
                # Handle static web services (like ImageLabelingWeb)
                service_path = self.project_root / service_info['path']
                launcher_file = service_path / service_info.get('launcher', 'launch_server.py')
                
                if not launcher_file.exists():
                    print(f"❌ Service launcher not found: {launcher_file}")
                    return False
                

                
                # Start the static web service
                process = subprocess.Popen([
                    sys.executable, str(launcher_file.absolute()),
                    "--port", str(service_info['port']),
                    "--host", "0.0.0.0",
                    "--no-browser"
                ], cwd=str(service_path.absolute()))
                
            else:
                # Handle FastAPI services
                service_path = self.project_root / service_info['path']
                app_file = service_path / "app.py"
                
                if not app_file.exists():
                    print(f"❌ Service app file not found: {app_file}")
                    return False
                
                # Start the FastAPI service
                process = subprocess.Popen([
                    sys.executable, str(app_file)
                ], cwd=str(service_path))
            
            self.running_processes[service_name] = process
            print(f"✅ {service_info['name']} started with PID {process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {service_name}: {e}")
            return False
    
    def start_gateway(self):
        """Start the gateway service."""
        if not self.config or 'gateway' not in self.config:
            print("❌ Gateway not configured")
            return False
            
        gateway_path = self.project_root / "web" / "gateway"
        app_file = gateway_path / "app.py"
        
        if not app_file.exists():
            print(f"❌ Gateway app file not found: {app_file}")
            return False
            
        print(f"🤖 Starting Robot Vision Gateway on port {self.config['gateway']['port']}...")
        
        try:
            process = subprocess.Popen([
                sys.executable, str(app_file)
            ], cwd=str(gateway_path))
            
            self.running_processes['gateway'] = process
            print(f"✅ Gateway started with PID {process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start gateway: {e}")
            return False
    
    def start_all_services(self):
        """Start all configured services."""
        print("🚀 Starting Robot Vision Services...")
        
        # Start gateway first
        if not self.start_gateway():
            return False
            
        # Start each service
        if self.config and 'services' in self.config:
            for service_name in self.config['services']:
                self.start_service(service_name)
                time.sleep(2)  # Small delay between service starts
        
        print(f"\n🎉 Started {len(self.running_processes)} services")
        print("🌐 Access Control Center at: http://localhost:8000")
        return True
    
    def stop_all_services(self):
        """Stop all running services."""
        print("🛑 Stopping all services...")
        
        for service_name, process in self.running_processes.items():
            try:
                print(f"Stopping {service_name}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {service_name} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {service_name}...")
                process.kill()
            except Exception as e:
                print(f"❌ Error stopping {service_name}: {e}")
        
        self.running_processes.clear()
    
    def list_services(self):
        """List all configured services."""
        if not self.config:
            print("❌ No configuration loaded")
            return
            
        print("📋 Configured Services:")
        print("=" * 50)
        
        # Gateway
        if 'gateway' in self.config:
            gateway = self.config['gateway']
            print(f"🤖 Gateway - {gateway['title']}")
            print(f"   Port: {gateway['port']}")
            print(f"   URL: http://localhost:{gateway['port']}")
            print()
        
        # Services
        if 'services' in self.config:
            for service_name, service_info in self.config['services'].items():
                print(f"🔧 {service_info['name']}")
                print(f"   Description: {service_info['description']}")
                print(f"   Port: {service_info['port']}")
                print(f"   Type: {service_info['type']}")
                print(f"   URL: http://localhost:{service_info['port']}")
                print()
    
    def wait_for_services(self):
        """Wait for all services to finish (blocking)."""
        if not self.running_processes:
            print("No services running")
            return
            
        print(f"🔄 Monitoring {len(self.running_processes)} services...")
        print("Press Ctrl+C to stop all services")
        
        try:
            while self.running_processes:
                time.sleep(1)
                # Remove finished processes
                finished = []
                for name, process in self.running_processes.items():
                    if process.poll() is not None:
                        finished.append(name)
                        print(f"⚠️ Service {name} finished unexpectedly")
                
                for name in finished:
                    del self.running_processes[name]
                    
        except KeyboardInterrupt:
            print("\n🛑 Received stop signal, shutting down services...")
            self.stop_all_services()

def main():
    parser = argparse.ArgumentParser(description="Robot Vision Services Manager")
    parser.add_argument('action', choices=['start', 'stop', 'list', 'start-service'], 
                       help='Action to perform')
    parser.add_argument('--service', help='Specific service name (for start-service action)')
    parser.add_argument('--wait', action='store_true', 
                       help='Wait for services to finish (use with start)')
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    if args.action == 'list':
        manager.list_services()
    elif args.action == 'start':
        if manager.start_all_services():
            if args.wait:
                manager.wait_for_services()
    elif args.action == 'stop':
        manager.stop_all_services()
    elif args.action == 'start-service':
        if not args.service:
            print("❌ --service parameter required for start-service action")
            return
        manager.start_service(args.service)
        if args.wait:
            manager.wait_for_services()

if __name__ == "__main__":
    main()