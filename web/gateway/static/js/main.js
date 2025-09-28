// Robot Vision Control Center - Main JavaScript

// Global state
let servicesData = [];
let refreshInterval = null;

// DOM Elements
const servicesGrid = document.getElementById('services-grid');
const refreshText = document.getElementById('refresh-text');
const serviceCardTemplate = document.getElementById('service-card-template');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🤖 Robot Vision Control Center - Loading...');
    
    // Load initial services
    loadServices();
    
    // Set up auto-refresh
    startAutoRefresh();
    
    // Set up event listeners
    setupEventListeners();
    
    console.log('✅ Application initialized');
});

// Event Listeners
function setupEventListeners() {
    // Handle visibility change to pause/resume auto-refresh
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            stopAutoRefresh();
        } else {
            startAutoRefresh();
        }
    });
    
    // Handle network status changes
    window.addEventListener('online', function() {
        showNotification('Network connection restored', 'success');
        refreshServices();
    });
    
    window.addEventListener('offline', function() {
        showNotification('Network connection lost', 'warning');
    });
}

// Service Management
async function loadServices() {
    try {
        showLoading(true);
        
        // Fetch services configuration
        const response = await fetch('/services/list');
        const data = await response.json();
        
        if (data.success) {
            servicesData = data.data.services;
            renderServices();
            console.log(`📋 Loaded ${servicesData.length} services`);
        } else {
            throw new Error(data.message || 'Failed to load services');
        }
    } catch (error) {
        console.error('❌ Error loading services:', error);
        showError('Failed to load services: ' + error.message);
    } finally {
        showLoading(false);
    }
}

async function refreshServices() {
    try {
        // Update refresh button
        refreshText.innerHTML = '<span class="refresh-indicator">🔄</span> Refreshing...';
        
        // Fetch current status
        const response = await fetch('/services/status');
        const data = await response.json();
        
        if (data.success) {
            updateServicesStatus(data.data.services);
            console.log('🔄 Services status refreshed');
        } else {
            throw new Error(data.message || 'Failed to refresh services');
        }
    } catch (error) {
        console.error('❌ Error refreshing services:', error);
        showNotification('Failed to refresh services: ' + error.message, 'error');
    } finally {
        // Reset refresh button
        refreshText.innerHTML = '🔄 Refresh Status';
    }
}

function renderServices() {
    if (!servicesGrid || !serviceCardTemplate) {
        console.error('Required DOM elements not found');
        return;
    }
    
    // Clear existing services
    servicesGrid.innerHTML = '';
    
    // Create service cards
    servicesData.forEach((service, index) => {
        const serviceCard = createServiceCard(service, index);
        servicesGrid.appendChild(serviceCard);
    });
    
    // Add fade-in animation
    servicesGrid.classList.add('fade-in');
}

function createServiceCard(service, index) {
    // Clone template
    const template = serviceCardTemplate.content.cloneNode(true);
    const card = template.querySelector('.service-card');
    
    // Set service data
    card.setAttribute('data-service-id', service.name.toLowerCase().replace(/\s+/g, '-'));
    card.setAttribute('id', `service-${index}`);
    
    // Populate content
    const nameElement = card.querySelector('.service-name');
    const descriptionElement = card.querySelector('.service-description');
    const portElement = card.querySelector('.port');
    const statusElement = card.querySelector('.status');
    const actionsElement = card.querySelector('.service-actions');
    
    nameElement.textContent = service.name;
    descriptionElement.textContent = service.description;
    portElement.textContent = `Port: ${service.port}`;
    
    // Set initial status
    statusElement.textContent = 'Checking...';
    statusElement.className = 'status unknown';
    statusElement.setAttribute('id', `status-${index}`);
    
    // Create action buttons
    const openServiceBtn = createButton('🚀 Open Service', 'btn-primary', service.service_url, true);
    actionsElement.appendChild(openServiceBtn);
    
    if (service.docs_url) {
        const docsBtn = createButton('📖 API Docs', 'btn-info', service.docs_url, true);
        actionsElement.appendChild(docsBtn);
    }
    
    return card;
}

function createButton(text, className, href, newTab = false) {
    const button = document.createElement('a');
    button.textContent = text;
    button.className = `btn ${className}`;
    button.href = href;
    
    if (newTab) {
        button.target = '_blank';
        button.rel = 'noopener noreferrer';
    }
    
    return button;
}

function updateServicesStatus(statusData) {
    statusData.forEach((serviceStatus, index) => {
        const statusElement = document.getElementById(`status-${index}`);
        const serviceCard = document.getElementById(`service-${index}`);
        
        if (statusElement && serviceCard) {
            // Update status text
            let statusText = serviceStatus.status.charAt(0).toUpperCase() + serviceStatus.status.slice(1);
            if (serviceStatus.response_time) {
                statusText += ` (${serviceStatus.response_time}ms)`;
            }
            statusElement.textContent = statusText;
            
            // Update status class
            statusElement.className = `status ${serviceStatus.status}`;
            
            // Update card border color
            serviceCard.className = `service-card ${serviceStatus.status}`;
            
            // Add error info if present
            if (serviceStatus.error) {
                statusElement.title = `Error: ${serviceStatus.error}`;
            } else {
                statusElement.removeAttribute('title');
            }
        }
    });
}

// Auto-refresh Management
function startAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    
    // Refresh every 30 seconds
    refreshInterval = setInterval(() => {
        if (!document.hidden) {
            refreshServices();
        }
    }, 30000);
    
    console.log('🔄 Auto-refresh started (30s interval)');
}

function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
        console.log('⏸️ Auto-refresh stopped');
    }
}

// UI State Management
function showLoading(show) {
    const container = document.querySelector('.container');
    if (container) {
        if (show) {
            container.classList.add('loading');
        } else {
            container.classList.remove('loading');
        }
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    
    // Set background color based on type
    const colors = {
        success: '#28a745',
        error: '#dc3545',
        warning: '#ffc107',
        info: '#17a2b8'
    };
    notification.style.backgroundColor = colors[type] || colors.info;
    
    notification.textContent = message;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

function showError(message) {
    console.error('❌', message);
    showNotification(message, 'error');
}

// Network Information
async function showNetworkInfo() {
    try {
        const response = await fetch('/network/info');
        const data = await response.json();
        
        if (data.success) {
            const info = data.data;
            let message = `Hostname: ${info.hostname}\n`;
            message += `Current Access: ${info.current_request_host}\n`;
            message += `Dynamic Address: ${info.dynamic_address}\n`;
            message += `Available Interfaces:\n`;
            
            for (const [iface, ip] of Object.entries(info.interfaces)) {
                message += `  ${iface}: ${ip}\n`;
            }
            
            alert(message);
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        showError('Failed to fetch network info: ' + error.message);
    }
}

// Utility Functions
function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

function formatDuration(ms) {
    if (ms < 1000) {
        return `${ms}ms`;
    } else {
        return `${(ms / 1000).toFixed(1)}s`;
    }
}

// Export functions for global access
window.refreshServices = refreshServices;
window.showNetworkInfo = showNetworkInfo;

// Service Worker Registration (for future PWA support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Future: Register service worker for offline support
        console.log('🔧 Service Worker support available');
    });
}

// Debug utilities (development only)
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.debugUtils = {
        servicesData,
        refreshServices,
        showNotification,
        loadServices
    };
    console.log('🔧 Debug utilities available in window.debugUtils');
}