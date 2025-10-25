// FlowFormer++ Real-Time Monitoring Dashboard
// Optimized for 16:9 Big Screen Display

class DashboardMonitor {
    constructor() {
        this.eventSource = null;
        this.isConnected = false;
        this.totalCalls = 0;
        this.init();
    }

    init() {
        console.log('🎯 Dashboard Monitor initializing...');
        this.connectRealTime();
    }

    connectRealTime() {
        try {
            console.log('🔌 Connecting to real-time monitoring...');
            this.eventSource = new EventSource('/api_events');
            
            this.eventSource.onopen = () => {
                console.log('✅ Real-time connection established');
                this.isConnected = true;
                this.updateConnectionStatus('connected');
            };
            
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleRealTimeEvent(data);
                } catch (error) {
                    console.error('Error parsing SSE data:', error);
                }
            };
            
            this.eventSource.onerror = (error) => {
                console.error('❌ Real-time connection error:', error);
                this.isConnected = false;
                this.updateConnectionStatus('reconnecting');
                
                // Auto-reconnect after 5 seconds
                setTimeout(() => {
                    console.log('🔄 Attempting to reconnect...');
                    this.connectRealTime();
                }, 5000);
            };
            
        } catch (error) {
            console.error('Failed to establish real-time connection:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    handleRealTimeEvent(event) {
        console.log('📡 Real-time event received:', event.type);
        
        switch (event.type) {
            case 'connected':
                console.log('✅ Dashboard connected:', event.message);
                break;
                
            case 'initial_data':
                if (event.data && event.data.logs && event.data.logs.length > 0) {
                    this.updateDashboard(event.data.logs[0]);
                    this.totalCalls = event.data.total || 0;
                    this.updateStats();
                }
                break;
                
            case 'api_call_update':
                console.log('🆕 New API call detected:', event.data.new_call.endpoint);
                this.updateDashboard(event.data.new_call);
                this.totalCalls = event.data.total_calls;
                this.updateStats();
                this.flashBorders();
                break;
                
            case 'keepalive':
                // Connection is alive
                break;
                
            default:
                console.log('Unknown event type:', event.type);
        }
    }

    updateDashboard(call) {
        console.log('🔄 Updating dashboard with new call:', call.id);

        // Check if we need to rebuild the entire content (transitioning from empty state)
        const dashboardContent = document.getElementById('dashboard-content');
        const refPanel = document.getElementById('ref-panel');
        const targetPanel = document.getElementById('target-panel');
        
        // If panels don't exist, we need to rebuild the entire structure
        if (!refPanel || !targetPanel) {
            console.log('📦 Rebuilding dashboard structure from empty state');
            this.rebuildDashboardStructure();
        }

        // Update reference image panel
        this.updateImagePanel('ref', call.ref_image_url, call.original_keypoints, call.keypoints_count);

        // Update target image panel
        this.updateImagePanel('target', call.target_image_url, call.tracked_keypoints, call.tracked_points);

        // Update metadata
        this.updateMetadata(call);

        // Update last update time
        const lastUpdateEl = document.getElementById('last-update');
        if (lastUpdateEl) {
            lastUpdateEl.textContent = call.timestamp;
        }
    }

    rebuildDashboardStructure() {
        const dashboardContent = document.getElementById('dashboard-content');
        if (!dashboardContent) return;

        // Replace the empty state with the proper panel structure
        dashboardContent.innerHTML = `
            <!-- Reference Image Panel -->
            <div class="image-panel" id="ref-panel">
                <div class="panel-header">
                    <div class="panel-title">
                        📷 Reference Image
                    </div>
                    <div class="panel-info">
                        0 keypoints
                    </div>
                </div>
                <div class="image-container">
                    <div class="empty-state">
                        <span class="empty-state-icon">📷</span>
                        <h3>Loading...</h3>
                    </div>
                </div>
            </div>

            <!-- Target Image Panel -->
            <div class="image-panel" id="target-panel">
                <div class="panel-header">
                    <div class="panel-title">
                        🔍 Tracked Result
                    </div>
                    <div class="panel-info">
                        0 tracked
                    </div>
                </div>
                <div class="image-container">
                    <div class="empty-state">
                        <span class="empty-state-icon">🔍</span>
                        <h3>Loading...</h3>
                    </div>
                </div>
            </div>

            <!-- Metadata Panel -->
            <div class="metadata-panel">
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <span class="metadata-label">Status</span>
                        <span class="metadata-value">Loading...</span>
                    </div>
                </div>
            </div>
        `;
    }

    updateImagePanel(type, imageUrl, keypoints, count) {
        const panelId = type === 'ref' ? 'ref-panel' : 'target-panel';
        const panel = document.getElementById(panelId);
        if (!panel) return;

        const panelInfo = panel.querySelector('.panel-info');
        const imageContainer = panel.querySelector('.image-container');
        
        if (!imageUrl) {
            // Show empty state
            imageContainer.innerHTML = `
                <div class="empty-state">
                    <span class="empty-state-icon">${type === 'ref' ? '📷' : '🔍'}</span>
                    <h3>No ${type === 'ref' ? 'Reference' : 'Tracking'} Image</h3>
                    <p>Waiting for API call...</p>
                </div>
            `;
            if (panelInfo) panelInfo.textContent = '0 keypoints';
            return;
        }

        // Update info
        if (panelInfo) {
            panelInfo.textContent = `${count || 0} ${type === 'ref' ? 'keypoints' : 'tracked'}`;
        }

        // Check if the image URL has actually changed
        const existingImg = imageContainer.querySelector('.dashboard-image');
        const currentSrc = existingImg ? existingImg.src : null;
        
        // Add cache-busting timestamp to ensure fresh load
        const cacheBustedUrl = imageUrl + (imageUrl.includes('?') ? '&' : '?') + '_t=' + Date.now();
        
        // If image already exists and URL is the same (ignoring cache-buster), update in place
        if (existingImg && currentSrc && currentSrc.split('?')[0] === new URL(imageUrl, window.location.origin).href) {
            // Image hasn't changed, just update keypoints without flickering
            const overlay = imageContainer.querySelector('.keypoints-overlay');
            if (overlay && keypoints && keypoints.length > 0) {
                const imgWidth = existingImg.naturalWidth;
                const imgHeight = existingImg.naturalHeight;
                this.positionKeypoints(existingImg, overlay, keypoints, imgWidth, imgHeight, type === 'ref' ? 'original' : 'tracked');
            }
            return;
        }

        // Create a new image element for preloading (prevent flicker)
        const img = new Image();
        
        // Set up error handler
        img.onerror = () => {
            console.error(`Failed to load image: ${imageUrl}`);
            imageContainer.innerHTML = `
                <div class="empty-state">
                    <span class="empty-state-icon">⚠️</span>
                    <h3>Failed to load image</h3>
                    <p>Image may not be available</p>
                </div>
            `;
        };
        
        img.onload = () => {
            // Store image dimensions for keypoint positioning
            const imgWidth = img.naturalWidth;
            const imgHeight = img.naturalHeight;
            
            // Check if the container still exists (component might have unmounted)
            if (!imageContainer || !imageContainer.parentElement) return;
            
            // Use smooth transition instead of innerHTML replacement to prevent flicker
            const existingImg = imageContainer.querySelector('.dashboard-image');
            const existingOverlay = imageContainer.querySelector('.keypoints-overlay');
            
            if (existingImg && existingOverlay) {
                // Update existing image smoothly
                existingImg.style.opacity = '0';
                setTimeout(() => {
                    if (existingImg.parentElement) {
                        existingImg.src = cacheBustedUrl;
                        existingImg.onload = () => {
                            existingImg.style.opacity = '1';
                            // Update keypoints after image transition
                            if (keypoints && keypoints.length > 0) {
                                this.positionKeypoints(existingImg, existingOverlay, keypoints, imgWidth, imgHeight, type === 'ref' ? 'original' : 'tracked');
                            }
                        };
                    }
                }, 150); // Match CSS transition time
            } else {
                // Create new image structure
                imageContainer.innerHTML = `
                    <img src="${cacheBustedUrl}" 
                         alt="${type === 'ref' ? 'Reference' : 'Target'} Image" 
                         class="dashboard-image"
                         style="opacity: 0; transition: opacity 0.3s ease-in-out;">
                    <div class="keypoints-overlay"></div>
                `;
                
                const displayedImg = imageContainer.querySelector('.dashboard-image');
                const overlay = imageContainer.querySelector('.keypoints-overlay');
                
                // Ensure image loads before showing
                if (displayedImg) {
                    displayedImg.onload = () => {
                        displayedImg.style.opacity = '1';
                        
                        // Position keypoints after image is fully rendered
                        if (overlay && keypoints && keypoints.length > 0) {
                            // Use requestAnimationFrame for better timing
                            requestAnimationFrame(() => {
                                this.positionKeypoints(displayedImg, overlay, keypoints, imgWidth, imgHeight, type === 'ref' ? 'original' : 'tracked');
                            });
                        }
                        
                        // Add resize handler for responsive keypoint positioning
                        const resizeHandler = () => {
                            if (overlay && keypoints && keypoints.length > 0) {
                                this.positionKeypoints(displayedImg, overlay, keypoints, imgWidth, imgHeight, type === 'ref' ? 'original' : 'tracked');
                            }
                        };
                        
                        // Clean up old resize handler if exists
                        if (displayedImg.dataset.resizeHandler === 'attached') {
                            window.removeEventListener('resize', displayedImg._resizeHandler);
                        }
                        
                        // Store handler for cleanup
                        displayedImg._resizeHandler = resizeHandler;
                        displayedImg.dataset.resizeHandler = 'attached';
                        window.addEventListener('resize', resizeHandler);
                    };
                }
            }
        };
        
        // Start loading the image
        img.src = cacheBustedUrl;
    }

    positionKeypoints(img, overlay, keypoints, originalWidth, originalHeight, type) {
        if (!img || !overlay || !keypoints) return;
        
        // Get the actual displayed dimensions and position of the image
        const imgRect = img.getBoundingClientRect();
        const containerRect = img.parentElement.getBoundingClientRect();
        
        // Calculate the offset of the image within its container
        const offsetX = imgRect.left - containerRect.left;
        const offsetY = imgRect.top - containerRect.top;
        
        // Set overlay to match image position and size
        overlay.style.left = `${offsetX}px`;
        overlay.style.top = `${offsetY}px`;
        overlay.style.width = `${imgRect.width}px`;
        overlay.style.height = `${imgRect.height}px`;
        
        // Calculate scale factors
        const scaleX = imgRect.width / originalWidth;
        const scaleY = imgRect.height / originalHeight;
        
        // Clear existing markers
        overlay.innerHTML = '';
        
        // Position each keypoint
        keypoints.forEach(kp => {
            const x = kp.x !== undefined ? kp.x : (Array.isArray(kp) ? kp[0] : 0);
            const y = kp.y !== undefined ? kp.y : (Array.isArray(kp) ? kp[1] : 0);
            
            // Scale the coordinates to match displayed image
            const scaledX = x * scaleX;
            const scaledY = y * scaleY;
            
            // Create marker
            const marker = document.createElement('div');
            marker.className = `keypoint-marker ${type}`;
            marker.style.left = `${scaledX}px`;
            marker.style.top = `${scaledY}px`;
            
            overlay.appendChild(marker);
        });
    }

    createKeypointsOverlay(keypoints, type) {
        // This function is now deprecated in favor of positionKeypoints
        // Kept for compatibility but not used
        if (!keypoints || keypoints.length === 0) return '';
        return '<div class="keypoints-overlay"></div>';
    }

    updateMetadata(call) {
        // Find metadata panel
        const metadataPanel = document.querySelector('.metadata-panel');
        if (!metadataPanel) return;

        const statusClass = call.success ? 'success' : 'error';
        const statusText = call.success ? '✅ Success' : '❌ Failed';

        metadataPanel.innerHTML = `
            <div class="metadata-grid">
                <div class="metadata-item ${statusClass}">
                    <span class="metadata-label">Status</span>
                    <span class="metadata-value">${statusText}</span>
                </div>
                <div class="metadata-item highlight">
                    <span class="metadata-label">Processing Time</span>
                    <span class="metadata-value">${call.processing_time}s</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Endpoint</span>
                    <span class="metadata-value" style="font-size: 1.2em;">${call.endpoint}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Input Keypoints</span>
                    <span class="metadata-value">${call.keypoints_count || 0}</span>
                </div>
                <div class="metadata-item highlight">
                    <span class="metadata-label">Tracked Points</span>
                    <span class="metadata-value">${call.tracked_points || 0}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Timestamp</span>
                    <span class="metadata-value" style="font-size: 1em;">${call.timestamp}</span>
                </div>
            </div>
        `;
    }

    updateStats() {
        const totalCallsEl = document.getElementById('total-calls');
        if (totalCallsEl) {
            totalCallsEl.textContent = this.totalCalls;
        }
    }

    updateConnectionStatus(status) {
        const indicator = document.getElementById('connection-indicator');
        const text = document.getElementById('connection-text');
        
        if (!indicator || !text) return;

        indicator.className = 'connection-indicator';
        
        switch (status) {
            case 'connected':
                indicator.classList.add('connected');
                text.textContent = 'Real-time Connected';
                break;
            case 'disconnected':
                indicator.classList.add('disconnected');
                text.textContent = 'Disconnected';
                break;
            case 'reconnecting':
                indicator.classList.add('reconnecting');
                text.textContent = 'Reconnecting...';
                break;
        }
    }

    flashBorders() {
        // Flash the image panels to indicate new data
        const panels = document.querySelectorAll('.image-panel');
        panels.forEach(panel => {
            panel.classList.add('new-call-flash');
            setTimeout(() => {
                panel.classList.remove('new-call-flash');
            }, 1000);
        });
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.isConnected = false;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎯 Starting FlowFormer++ Dashboard Monitor');
    window.dashboardMonitor = new DashboardMonitor();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboardMonitor) {
        window.dashboardMonitor.disconnect();
    }
});
