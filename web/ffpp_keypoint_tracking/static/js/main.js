// FlowFormer++ API Call Monitor - JavaScript Functionality

class APICallMonitor {
    constructor() {
        this.refreshInterval = null;
        this.eventSource = null;
        this.isRealTimeEnabled = true;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkServiceStatus();
        this.connectRealTime(); // Start real-time connection
        console.log('API Call Monitor initialized with real-time updates');
    }

    setupEventListeners() {
        // Service management
        const refreshBtn = document.getElementById('refresh-status');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.checkServiceStatus();
                this.refreshAPILogs();
            });
        }

        // Manual refresh for API logs
        const refreshApiBtn = document.getElementById('refresh-api-logs');
        if (refreshApiBtn) {
            refreshApiBtn.addEventListener('click', () => {
                this.refreshAPILogs();
                this.showNotification('Refreshing API logs...', 'info');
            });
        }

        // Real-time toggle (replaces auto-refresh)
        const realTimeToggle = document.getElementById('auto-refresh-toggle');
        if (realTimeToggle) {
            realTimeToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.connectRealTime();
                } else {
                    this.disconnectRealTime();
                    this.startPollingFallback();
                }
            });
        }
    }

    connectRealTime() {
        // Disconnect any existing connection
        this.disconnectRealTime();
        
        try {
            console.log('Connecting to real-time API monitoring...');
            this.eventSource = new EventSource('/api_events');
            
            this.eventSource.onopen = () => {
                console.log('✅ Real-time connection established');
                this.isRealTimeEnabled = true;
                this.showNotification('Real-time monitoring connected', 'success');
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
                console.error('Real-time connection error:', error);
                this.updateConnectionStatus('error');
                
                // Auto-reconnect after 5 seconds
                setTimeout(() => {
                    if (this.isRealTimeEnabled) {
                        console.log('Attempting to reconnect...');
                        this.connectRealTime();
                    }
                }, 5000);
            };
            
        } catch (error) {
            console.error('Failed to establish real-time connection:', error);
            this.startPollingFallback();
        }
    }

    disconnectRealTime() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.isRealTimeEnabled = false;
        this.updateConnectionStatus('disconnected');
    }

    handleRealTimeEvent(event) {
        console.log('Real-time event received:', event.type);
        
        switch (event.type) {
            case 'connected':
                console.log(event.message);
                break;
                
            case 'initial_data':
                this.updateAPILogsTable(event.data.logs);
                this.updateStats(event.data);
                break;
                
            case 'api_call_update':
                // Add new API call to the top of the list
                this.addNewAPICall(event.data.new_call);
                this.updateStats({total: event.data.total_calls, logs: [event.data.new_call]});
                this.showNotification(`New ${event.data.new_call.endpoint} call completed!`, 'realtime');
                break;
                
            case 'keepalive':
                // Connection is alive, no action needed
                break;
                
            default:
                console.log('Unknown event type:', event.type);
        }
    }

    addNewAPICall(newCall) {
        const feedContainer = document.getElementById('api-call-feed');
        if (!feedContainer) return;

        // Create the new call card
        const newCallElement = document.createElement('div');
        newCallElement.innerHTML = this.createVisualCallCard(newCall);
        
        // Add highlight animation for new calls
        const callCard = newCallElement.firstElementChild;
        if (callCard) {
            callCard.classList.add('new-call-highlight');
            
            // Insert at the top
            if (feedContainer.firstChild) {
                feedContainer.insertBefore(callCard, feedContainer.firstChild);
            } else {
                feedContainer.appendChild(callCard);
            }
            
            // Remove highlight after animation
            setTimeout(() => {
                callCard.classList.remove('new-call-highlight');
            }, 3000);
            
            // Limit the number of displayed calls to prevent memory issues
            const maxCalls = 20;
            const allCalls = feedContainer.children;
            while (allCalls.length > maxCalls) {
                feedContainer.removeChild(allCalls[allCalls.length - 1]);
            }
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.querySelector('.connection-status');
        if (statusElement) {
            statusElement.className = `connection-status ${status}`;
            const statusText = {
                'connected': '🟢 Real-time',
                'disconnected': '🔴 Offline',
                'error': '🟡 Reconnecting...'
            };
            statusElement.textContent = statusText[status] || status;
        }
    }

    startPollingFallback() {
        console.log('Starting polling fallback...');
        this.stopPollingFallback();
        
        // Refresh API logs every 5 seconds as fallback
        this.refreshInterval = setInterval(() => {
            this.refreshAPILogs();
        }, 5000);
        
        this.showNotification('Using polling mode (real-time unavailable)', 'warning');
    }

    stopPollingFallback() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    async refreshAPILogs() {
        try {
            console.log('Refreshing API logs...');
            const response = await fetch('/api_logs?per_page=5');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('API logs data received:', data);
            
            if (data.success && data.logs) {
                this.updateAPILogsTable(data.logs);
                this.updateStats(data);
                console.log(`Updated dashboard with ${data.logs.length} API calls`);
            } else {
                console.warn('No logs data in response:', data);
            }
        } catch (error) {
            console.error('Error refreshing API logs:', error);
            this.showNotification('Failed to refresh API logs', 'error');
        }
    }

    updateAPILogsTable(logs) {
        const feedContainer = document.getElementById('api-call-feed');
        if (!feedContainer) return;

        if (logs.length === 0) {
            feedContainer.innerHTML = `
                <div class="no-api-calls-visual">
                    <div class="empty-state-visual">
                        <span class="empty-icon">🎯</span>
                        <h3>No API Calls to Monitor</h3>
                        <p>Start making API calls to see rich visual monitoring of FlowFormer++ computations</p>
                        <div class="endpoint-examples-visual">
                            <div class="endpoint-example">
                                <code>POST /set_reference_image</code>
                                <span>Store reference image with keypoints</span>
                            </div>
                            <div class="endpoint-example">
                                <code>POST /track_keypoints</code>
                                <span>Track keypoints across images</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            return;
        }

        feedContainer.innerHTML = logs.map(call => this.createVisualCallCard(call)).join('');
    }

    createVisualCallCard(call) {
        const metadataItems = call.detailed_metadata ? 
            Object.entries(call.detailed_metadata).map(([key, value]) => `
                <div class="metadata-item">
                    <span class="metadata-label">${this.formatLabel(key)}:</span>
                    <span class="metadata-value">${value}</span>
                </div>
            `).join('') : '';

        const refImageSection = call.ref_image_url ? `
            <div class="image-display">
                <h4>📷 Reference Image (${call.ref_image_size || 'Unknown'})</h4>
                <div class="image-container">
                    <img src="${call.ref_image_url}" alt="Reference Image" class="api-call-image">
                    ${this.createKeypointsOverlay(call.original_keypoints, 'original')}
                </div>
            </div>
        ` : '';

        const targetImageSection = call.target_image_url ? `
            <div class="image-display">
                <h4>🔍 Target Image (${call.target_image_size || 'Unknown'})</h4>
                <div class="image-container">
                    <img src="${call.target_image_url}" alt="Target Image" class="api-call-image">
                    ${this.createKeypointsOverlay(call.tracked_keypoints, 'tracked')}
                </div>
            </div>
        ` : '';

        const errorSection = !call.success && call.error ? `
            <div class="error-details">
                <h5>❌ Error Details:</h5>
                <p class="error-message">${call.error}</p>
            </div>
        ` : '';

        return `
            <div class="api-call-visual ${call.success ? 'success-visual' : 'error-visual'}" data-call-id="${call.id}">
                <div class="call-header">
                    <div class="call-info">
                        <span class="call-endpoint">${call.endpoint}</span>
                        <span class="call-timestamp">${call.timestamp}</span>
                    </div>
                    <div class="call-status">
                        <span class="status-badge ${call.success ? 'success' : 'error'}">
                            ${call.success ? '✅ Success' : '❌ Failed'}
                        </span>
                        <span class="timing-badge">${call.processing_time}s</span>
                    </div>
                </div>

                <div class="call-visual-content">
                    <div class="images-section">
                        ${refImageSection}
                        ${targetImageSection}
                    </div>

                    <div class="metadata-section">
                        <h4>📊 Computation Metadata</h4>
                        <div class="metadata-grid">
                            <div class="metadata-item highlight">
                                <span class="metadata-label">Input Keypoints:</span>
                                <span class="metadata-value">${call.keypoints_count || 0}</span>
                            </div>
                            <div class="metadata-item highlight">
                                <span class="metadata-label">Tracked Points:</span>
                                <span class="metadata-value tracked">${call.tracked_points || 0}</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Processing Time:</span>
                                <span class="metadata-value">${call.processing_time}s</span>
                            </div>
                            ${metadataItems}
                        </div>
                        ${errorSection}
                    </div>
                </div>

                <div class="call-footer">
                    <span class="call-message">${call.message}</span>
                </div>
            </div>
        `;
    }

    createKeypointsOverlay(keypoints, type) {
        if (!keypoints || keypoints.length === 0) return '';
        
        const markers = keypoints.map(kp => {
            const x = Array.isArray(kp) ? kp[0] : kp.x;
            const y = Array.isArray(kp) ? kp[1] : kp.y;
            return `<div class="keypoint-marker ${type}" style="left: ${x}px; top: ${y}px;"></div>`;
        }).join('');
        
        return `<div class="keypoints-overlay">${markers}</div>`;
    }

    formatLabel(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    updateStats(data) {
        // Update total API calls - find the first stat-value (total calls)
        const statValues = document.querySelectorAll('.stat-value');
        if (statValues.length > 0 && data.total !== undefined) {
            statValues[0].textContent = data.total;
        }

        // Update last updated time
        const lastUpdatedElement = document.getElementById('last-updated');
        if (lastUpdatedElement) {
            if (data.logs && data.logs.length > 0) {
                lastUpdatedElement.textContent = data.logs[0].timestamp;
            } else {
                lastUpdatedElement.textContent = new Date().toLocaleTimeString();
            }
        }

        // Add visual indicator that data was refreshed
        const feedContainer = document.getElementById('api-call-feed');
        if (feedContainer) {
            feedContainer.style.opacity = '0.8';
            setTimeout(() => {
                feedContainer.style.opacity = '1';
            }, 200);
        }
    }

    async checkServiceStatus() {
        try {
            const response = await fetch('/status');
            const status = await response.json();
            
            this.updateStatusDisplay(status);
        } catch (error) {
            this.updateStatusDisplay({
                status: 'error',
                error: 'Failed to connect to service'
            });
        }
    }

    updateStatusDisplay(status) {
        const statusBanner = document.querySelector('.status-banner');
        
        if (status.status === 'ready') {
            statusBanner.className = 'status-banner status-ready';
            statusBanner.innerHTML = `
                <span>✅ FlowFormer++ Service Ready</span>
                ${status.gpu_available ? '<span class="gpu-badge">GPU Accelerated</span>' : ''}
            `;
        } else {
            statusBanner.className = 'status-banner status-error';
            statusBanner.innerHTML = `
                <span>❌ Service Error: ${status.error || 'Unknown error'}</span>
            `;
        }

        // Update status cards if they exist
        this.updateStatusCards(status);
    }

    updateStatusCards(status) {
        const updateCard = (cardSelector, info) => {
            const card = document.querySelector(cardSelector);
            if (!card) return;
            
            const infoContainer = card.querySelector('.status-info-container') || card;
            infoContainer.innerHTML = Object.entries(info).map(([key, value]) => `
                <div class="status-info">
                    <span class="label">${key}:</span>
                    <span class="value ${typeof value === 'boolean' ? (value ? 'status-ok' : 'status-error') : ''}">${value}</span>
                </div>
            `).join('');
        };

        if (status.model_info) {
            updateCard('.model-status', status.model_info);
        }
        
        if (status.system_info) {
            updateCard('.system-status', status.system_info);
        }
    }

    showNotification(message, type = 'info') {
        const container = document.querySelector('.notification-container') || this.createNotificationContainer();
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    createNotificationContainer() {
        const container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
        return container;
    }
}

// Initialize the API monitor when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.apiMonitor = new APICallMonitor();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.apiMonitor) {
        window.apiMonitor.disconnectRealTime();
        window.apiMonitor.stopPollingFallback();
    }
});

// Utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
        return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(1)}s`;
    } else {
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }
}

// Demo function for testing API monitoring
async function makeDemoAPICall() {
    const demoBtn = document.getElementById('demo-btn');
    const originalText = demoBtn.textContent;
    
    demoBtn.disabled = true;
    demoBtn.textContent = '🔄 Generating...';
    
    try {
        const response = await fetch('/demo_api_call', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                demo: true,
                timestamp: new Date().toISOString()
            })
        });
        
        const result = await response.json();
        
        // Show notification
        if (window.apiMonitor) {
            window.apiMonitor.showNotification(
                result.success ? 'Demo API call succeeded!' : 'Demo API call failed (as expected)',
                result.success ? 'success' : 'info'
            );
        }
        
        // Refresh logs immediately to show the new call
        setTimeout(() => {
            if (window.apiMonitor) {
                window.apiMonitor.refreshAPILogs();
            }
        }, 500);
        
    } catch (error) {
        console.error('Error making demo API call:', error);
        if (window.apiMonitor) {
            window.apiMonitor.showNotification('Error making demo API call', 'error');
        }
    } finally {
        setTimeout(() => {
            demoBtn.disabled = false;
            demoBtn.textContent = originalText;
        }, 1000);
    }
}

// Export for global access
window.APICallMonitor = APICallMonitor;