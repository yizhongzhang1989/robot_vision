#!/usr/bin/env python3
"""
Robot Vision Services Gateway
============================

Main control interface for all robot vision services.
Provides service discovery, health monitoring, and unified access point.
"""

import os
import sys
import json
import time
import logging
import requests
import socket
import subprocess
from typing import Dict, List, Optional
from flask import Flask, jsonify, request, render_template, url_for
import yaml

# Add project root to path
try:
    # Try to get the file path
    current_file = __file__
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
except NameError:
    # Fallback if __file__ is not defined
    project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
    if not os.path.exists(os.path.join(project_root, 'config', 'services.yaml')):
        project_root = os.path.abspath('.')
        
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_network_interfaces():
    """Get all available network interfaces and their IP addresses."""
    interfaces = {}
    try:
        result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
        current_interface = None
        
        for line in result.stdout.split('\n'):
            # Look for interface names
            if line and not line.startswith(' '):
                parts = line.split(':')
                if len(parts) >= 2:
                    current_interface = parts[1].strip()
            # Look for IPv4 addresses
            elif 'inet ' in line and current_interface:
                ip = line.split()[1].split('/')[0]
                if ip != '127.0.0.1':  # Skip loopback
                    interfaces[current_interface] = ip
                    
    except Exception as e:
        logger.warning(f"Could not detect network interfaces: {e}")
        
    return interfaces

def get_hostname():
    """Get the machine's hostname."""
    try:
        return socket.gethostname()
    except Exception:
        return "localhost"

def get_best_server_address():
    """Get the best server address to use for service URLs."""
    # Priority 1: Manual override
    override = os.environ.get('ROBOT_VISION_SERVER_IP')
    if override:
        return override
    
    # Priority 2: Use hostname if it resolves properly
    hostname = get_hostname()
    try:
        socket.gethostbyname(hostname)
        return hostname
    except Exception:
        pass
    
    # Priority 3: Choose best IP from available interfaces
    interfaces = get_all_network_interfaces()
    
    # Preferred interface order (most to least preferred)
    interface_priority = ['eth0', 'enp0s3', 'wlan0', 'ens33', 'ens160']
    
    # Try preferred interfaces first
    for preferred in interface_priority:
        for iface_name, ip in interfaces.items():
            if preferred in iface_name:
                return ip
    
    # If no preferred interface found, avoid Docker/bridge interfaces
    for iface_name, ip in interfaces.items():
        if not any(skip in iface_name for skip in ['docker', 'br-', 'veth', 'lo']):
            return ip
    
    # Fallback: any available IP
    if interfaces:
        return list(interfaces.values())[0]
    
    # Last resort: auto-detect via socket connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def get_dynamic_server_address(request_obj):
    """Get server address based on how the client is accessing the gateway."""
    if not request_obj:
        return get_best_server_address()
    
    # Get the Host header from the request
    host_header = request_obj.headers.get('Host', '')
    if host_header:
        # Extract just the hostname/IP part (remove port if present)
        # Split by colon, but only remove the last part if it's a number (port)
        parts = host_header.rsplit(':', 1)
        if len(parts) == 2:
            # Check if the last part is a port number
            try:
                int(parts[1])
                # It's a port number, use the hostname part
                host_part = parts[0]
            except ValueError:
                # Not a port number (might be part of hostname), keep the full string
                host_part = host_header
        else:
            host_part = host_header
        
        # Return the full hostname/domain as provided by the client
        # This preserves full domain names like "msraig-ubuntu-4.guest.corp.microsoft.com"
        return host_part
    
    # Fallback to best available address
    return get_best_server_address()

# Get default server address for service URLs
SERVER_ADDRESS = get_best_server_address()
HOSTNAME = get_hostname()
ALL_INTERFACES = get_all_network_interfaces()

# Load service configuration
def load_service_config():
    """Load service configuration from YAML file."""
    config_path = os.path.join(project_root, 'config', 'services.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load service config: {e}")
        return None

config = load_service_config()
GATEWAY_PORT = config.get('gateway', {}).get('port', 8000) if config else 8000

# Initialize Flask app with template and static folder configuration
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Data classes for responses
class ServiceStatus:
    def __init__(self, name: str, port: int, status: str, health_url: str, 
                 docs_url: str, last_checked: str, response_time: Optional[float] = None, 
                 error: Optional[str] = None):
        self.name = name
        self.port = port
        self.status = status
        self.health_url = health_url
        self.docs_url = docs_url
        self.last_checked = last_checked
        self.response_time = response_time
        self.error = error
    
    def to_dict(self):
        return {
            'name': self.name,
            'port': self.port,
            'status': self.status,
            'health_url': self.health_url,
            'docs_url': self.docs_url,
            'last_checked': self.last_checked,
            'response_time': self.response_time,
            'error': self.error
        }

def create_gateway_response(success: bool, message: str, data: Optional[Dict] = None) -> Dict:
    return {
        'success': success,
        'message': message,
        'data': data,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

def check_service_health(service_name: str, service_info: Dict, server_address: str = None) -> ServiceStatus:
    """Check health of a specific service."""
    port = service_info['port']
    service_type = service_info.get('type', 'fastapi')
    
    if not server_address:
        server_address = SERVER_ADDRESS
    
    # Determine health and docs URLs based on service type
    if service_type == 'static_web':
        health_url = f"http://localhost:{port}/"  # Keep localhost for health checks (internal)
        docs_url = None  # Static web services typically don't have API docs
    else:
        health_url = f"http://localhost:{port}/health"  # Keep localhost for health checks (internal)
        docs_url = f"http://{server_address}:{port}/docs"
    
    start_time = time.time()
    
    try:
        response = requests.get(health_url, timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return ServiceStatus(
                name=service_info['name'],
                port=port,
                status="healthy",
                health_url=health_url,
                docs_url=docs_url,
                last_checked=time.strftime('%Y-%m-%d %H:%M:%S'),
                response_time=round(response_time * 1000, 2)  # ms
            )
        else:
            return ServiceStatus(
                name=service_info['name'],
                port=port,
                status="unhealthy",
                health_url=health_url,
                docs_url=docs_url,
                last_checked=time.strftime('%Y-%m-%d %H:%M:%S'),
                response_time=round(response_time * 1000, 2),
                error=f"HTTP {response.status_code}"
            )
    except Exception as e:
        response_time = time.time() - start_time
        return ServiceStatus(
            name=service_info['name'],
            port=port,
            status="down",
            health_url=health_url,
            docs_url=docs_url,
            last_checked=time.strftime('%Y-%m-%d %H:%M:%S'),
            response_time=round(response_time * 1000, 2),
            error=str(e)
        )

@app.route("/")
def dashboard():
    """Main dashboard with service overview."""
    if not config:
        return render_template('error.html', 
                             error_message="Service configuration not loaded"), 500
    
    # Pass configuration to template
    template_data = {
        'services': config.get('services', {}),
        'gateway_config': config.get('gateway', {}),
        'server_address': get_dynamic_server_address(request)
    }
    
    return render_template('dashboard.html', **template_data)

@app.route("/health")
def gateway_health():
    """Gateway health check."""
    return jsonify(create_gateway_response(
        success=True,
        message="Robot Vision Gateway is running",
        data={
            "service": "Robot Vision Gateway",
            "version": "2.0.0",
            "port": 8000,
            "services_configured": len(config.get('services', {})) if config else 0
        }
    ))

@app.route("/services/status")
def services_status():
    """Get status of all configured services."""
    if not config or 'services' not in config:
        return jsonify(create_gateway_response(
            success=False,
            message="Service configuration not available"
        )), 500
    
    # Get server address based on how client is accessing the gateway
    server_address = get_dynamic_server_address(request)
    
    service_statuses = []
    
    # Check each service synchronously
    for service_name, service_info in config['services'].items():
        try:
            status = check_service_health(service_name, service_info, server_address)
            service_statuses.append(status)
        except Exception as e:
            logger.error(f"Error checking service {service_name}: {e}")
    
    # Calculate summary statistics
    total_services = len(service_statuses)
    healthy_services = len([s for s in service_statuses if s.status == "healthy"])
    
    return jsonify(create_gateway_response(
        success=True,
        message=f"Status check completed for {total_services} services",
        data={
            "services": [s.to_dict() for s in service_statuses],
            "summary": {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": total_services - healthy_services
            }
        }
    ))

@app.route("/services/list")
def list_services():
    """List all configured services."""
    if not config or 'services' not in config:
        return jsonify(create_gateway_response(
            success=False,
            message="No services configured",
            data={"services": []}
        ))
    
    # Get server address based on how client is accessing the gateway
    server_address = get_dynamic_server_address(request)
    
    services = []
    for service_name, service_info in config['services'].items():
        service_type = service_info.get('type', 'fastapi')
        
        # Build service info based on type - use consistent server address
        service_data = {
            "name": service_info['name'],
            "description": service_info['description'],
            "port": service_info['port'],
            "type": service_type,
            "service_url": f"http://{server_address}:{service_info['port']}",
        }
        
        # Add docs and health URLs based on service type
        if service_type == 'static_web':
            service_data["health_url"] = f"http://localhost:{service_info['port']}/"  # Internal health check
            service_data["docs_url"] = None
        else:
            service_data["health_url"] = f"http://localhost:{service_info['port']}/health"  # Internal health check
            service_data["docs_url"] = f"http://{server_address}:{service_info['port']}/docs"
        
        services.append(service_data)
    
    return jsonify(create_gateway_response(
        success=True,
        message=f"Found {len(services)} configured services",
        data={"services": services}
    ))

@app.route("/docs")
def api_docs():
    """API documentation page."""
    template_data = {
        'base_url': f"http://{get_dynamic_server_address(request)}:8000",
        'hostname': get_dynamic_server_address(request)
    }
    return render_template('api_docs.html', **template_data)

def initialize_gateway():
    """Initialize the gateway when the app starts."""
    logger.info("🚀 Starting Robot Vision Gateway...")
    if config:
        logger.info(f"✅ Configuration loaded with {len(config.get('services', {}))} services")
    else:
        logger.warning("⚠️ Failed to load service configuration")
    logger.info("✅ Robot Vision Gateway ready!")

if __name__ == "__main__":
    print("🤖 Starting Robot Vision Control Center")
    print("📋 Features:")
    print("   - ✅ Service discovery and monitoring")
    print("   - 🎛️ Centralized control dashboard")
    print("   - 📊 Health status tracking")
    print("   - 🔧 Gateway API for service management")
    print(f"🌐 Access at: http://localhost:{GATEWAY_PORT} (local) or http://{SERVER_ADDRESS}:{GATEWAY_PORT} (external)")
    print("📖 Flask routes: /health, /services/status, /services/list")
    print(f"🔗 Service links will use address: {SERVER_ADDRESS}")
    print(f"🏠 Hostname: {HOSTNAME}")
    print(f"🌐 Available interfaces: {list(ALL_INTERFACES.keys())}")
    
    initialize_gateway()
    
    # Add a new endpoint to show network info
    @app.route("/network/info")
    def network_info():
        """Show network configuration information."""
        return jsonify(create_gateway_response(
            success=True,
            message="Network configuration information",
            data={
                "hostname": HOSTNAME,
                "interfaces": ALL_INTERFACES,
                "default_address": SERVER_ADDRESS,
                "current_request_host": request.headers.get('Host', 'unknown'),
                "dynamic_address": get_dynamic_server_address(request)
            }
        ))
    
    app.run(
        host="0.0.0.0",
        port=GATEWAY_PORT,
        debug=False
    )