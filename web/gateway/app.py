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
from flask import Flask, jsonify, request
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
        # Extract just the hostname/IP part (remove port)
        host_part = host_header.split(':')[0]
        
        # If client is using hostname, return hostname for consistency
        hostname = get_hostname()
        if host_part == hostname:
            return hostname
            
        # If client is using specific IP, return that IP
        interfaces = get_all_network_interfaces()
        if host_part in interfaces.values():
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

# Initialize Flask app
app = Flask(__name__)

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
        return "<h1>Error: Service configuration not loaded</h1>"
    
    # Get server address based on how client is accessing the gateway
    server_address = get_dynamic_server_address(request)
    
    services_html = ""
    if 'services' in config:
        for service_name, service_info in config['services'].items():
            port = service_info['port']
            status_class = "unknown"
            
            # Build service actions - use consistent address
            actions_html = f'<a href="http://{server_address}:{port}" class="btn btn-primary" target="_blank">Open Service</a>'
            
            # Add API docs button only if the service has docs endpoint
            if service_info.get('docs_endpoint'):
                actions_html += f'<a href="http://{server_address}:{port}{service_info["docs_endpoint"]}" class="btn btn-info" target="_blank">API Docs</a>'
            
            services_html += f"""
                <div class="service-card {status_class}" id="service-{service_name}">
                    <h3>{service_info['name']}</h3>
                    <p>{service_info['description']}</p>
                    <div class="service-info">
                        <span class="port">Port: {port}</span>
                        <span class="status" id="status-{service_name}">Checking...</span>
                    </div>
                    <div class="service-actions">
                        {actions_html}
                    </div>
                </div>
            """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Robot Vision Control Center</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; 
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                color: #2c3e50;
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }}
            .header p {{
                color: #7f8c8d;
                margin: 0;
                font-size: 1.2em;
            }}
            .services-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .service-card {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                border-left: 4px solid #bdc3c7;
                transition: all 0.3s ease;
            }}
            .service-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.15);
            }}
            .service-card.healthy {{
                border-left-color: #27ae60;
            }}
            .service-card.unhealthy {{
                border-left-color: #f39c12;
            }}
            .service-card.down {{
                border-left-color: #e74c3c;
            }}
            .service-card h3 {{
                color: #2c3e50;
                margin: 0 0 10px 0;
                font-size: 1.4em;
            }}
            .service-card p {{
                color: #7f8c8d;
                margin: 0 0 15px 0;
                line-height: 1.4;
            }}
            .service-info {{
                display: flex;
                justify-content: space-between;
                margin: 15px 0;
                font-size: 0.9em;
            }}
            .port {{
                background: #ecf0f1;
                padding: 4px 8px;
                border-radius: 4px;
                color: #2c3e50;
            }}
            .status {{
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .status.healthy {{
                background: #d5f4e6;
                color: #27ae60;
            }}
            .status.unhealthy {{
                background: #fef9e7;
                color: #f39c12;
            }}
            .status.down {{
                background: #fadbd8;
                color: #e74c3c;
            }}
            .service-actions {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }}
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                text-decoration: none;
                font-size: 0.9em;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            .btn-primary {{
                background: #3498db;
                color: white;
            }}
            .btn-primary:hover {{
                background: #2980b9;
            }}
            .btn-info {{
                background: #17a2b8;
                color: white;
            }}
            .btn-info:hover {{
                background: #138496;
            }}
            .btn-success {{
                background: #28a745;
                color: white;
            }}
            .btn-success:hover {{
                background: #218838;
            }}
            .control-panel {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }}
            .control-panel h2 {{
                color: #2c3e50;
                margin: 0 0 20px 0;
            }}
            .control-actions {{
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }}
            .refresh-indicator {{
                display: inline-block;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Robot Vision Control Center</h1>
                <p>Centralized management for robot vision services</p>
                <div style="margin-top: 15px; padding: 10px; background: rgba(52, 152, 219, 0.1); border-radius: 6px; font-size: 0.9em;">
                    <strong>🔗 Smart Address Resolution:</strong> Service links automatically match how you access this gateway
                    <br>• <strong>Hostname access:</strong> Links use hostname for consistency
                    <br>• <strong>IP access:</strong> Links use the same IP address
                </div>
            </div>
            
            <div class="services-grid">
                {services_html}
            </div>
            
            <div class="control-panel">
                <h2>🎛️ Control Panel</h2>
                <div class="control-actions">
                    <button onclick="refreshServices()" class="btn btn-success">
                        <span id="refresh-text">🔄 Refresh Status</span>
                    </button>
                    <a href="/services/status" class="btn btn-info">📊 Detailed Status</a>
                    <a href="/network/info" class="btn btn-info">🌐 Network Info</a>
                    <a href="/docs" class="btn btn-primary">📖 Gateway API</a>
                </div>
            </div>
        </div>
        
        <script>
            // Auto-refresh service status
            async function refreshServices() {{
                document.getElementById('refresh-text').innerHTML = '<span class="refresh-indicator">🔄</span> Refreshing...';
                
                try {{
                    const response = await fetch('/services/status');
                    const data = await response.json();
                    
                    if (data.success) {{
                        data.data.services.forEach(service => {{
                            const serviceCard = document.getElementById('service-' + service.name.toLowerCase().replace(/ /g, '_'));
                            const statusSpan = document.getElementById('status-' + service.name.toLowerCase().replace(/ /g, '_'));
                            
                            if (serviceCard && statusSpan) {{
                                // Update card class
                                serviceCard.className = 'service-card ' + service.status;
                                
                                // Update status text
                                statusSpan.textContent = service.status.charAt(0).toUpperCase() + service.status.slice(1);
                                statusSpan.className = 'status ' + service.status;
                                
                                if (service.response_time) {{
                                    statusSpan.textContent += ' (' + service.response_time + 'ms)';
                                }}
                            }}
                        }});
                    }}
                }} catch (error) {{
                    console.error('Failed to refresh services:', error);
                }}
                
                document.getElementById('refresh-text').innerHTML = '🔄 Refresh Status';
            }}
            
            // Initial load and periodic refresh
            refreshServices();
            setInterval(refreshServices, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    """
    return html_content

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
    """Simple API documentation."""
    docs_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Vision Gateway API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #0066cc; }
        </style>
    </head>
    <body>
        <h1>Robot Vision Gateway API</h1>
        
        <div class="endpoint">
            <div class="method">GET /</div>
            <p>Main dashboard with service overview</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /health</div>
            <p>Gateway health check</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /services/status</div>
            <p>Get status of all configured services</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /services/list</div>
            <p>List all configured services</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /docs</div>
            <p>This API documentation</p>
        </div>
    </body>
    </html>
    """
    return docs_html

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
    print(f"🌐 Access at: http://localhost:8000 (local) or http://{SERVER_ADDRESS}:8000 (external)")
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
        port=8000,
        debug=False
    )