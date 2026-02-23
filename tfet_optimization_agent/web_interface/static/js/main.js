// Main JavaScript for the TFET Optimization Agent

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Add smooth scrolling for navigation links
    addSmoothScrolling();
    
    // Add interactive effects to feature cards
    addFeatureCardEffects();
    
    // Check system status
    checkSystemStatus();
}

function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function addFeatureCardEffects() {
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
            this.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 4px 20px rgba(0,0,0,0.1)';
        });
    });
}

async function checkSystemStatus() {
    try {
        // Check if the API is responsive
        const response = await fetch('/api/csv-status');
        if (response.ok) {
            console.log('System status: OK');
            showSystemStatus('online');
        } else {
            showSystemStatus('warning');
        }
    } catch (error) {
        console.error('System check failed:', error);
        showSystemStatus('offline');
    }
}

function showSystemStatus(status) {
    // Create status indicator if it doesn't exist
    let statusIndicator = document.getElementById('systemStatus');
    if (!statusIndicator) {
        statusIndicator = document.createElement('div');
        statusIndicator.id = 'systemStatus';
        statusIndicator.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            z-index: 1000;
            transition: all 0.3s ease;
        `;
        document.body.appendChild(statusIndicator);
    }
    
    switch(status) {
        case 'online':
            statusIndicator.textContent = '● System Online';
            statusIndicator.style.background = '#28a745';
            statusIndicator.style.color = 'white';
            break;
        case 'warning':
            statusIndicator.textContent = '⚠ System Warning';
            statusIndicator.style.background = '#ffc107';
            statusIndicator.style.color = '#212529';
            break;
        case 'offline':
            statusIndicator.textContent = '● System Offline';
            statusIndicator.style.background = '#dc3545';
            statusIndicator.style.color = 'white';
            break;
    }
    
    // Auto-hide after 5 seconds for online status
    if (status === 'online') {
        setTimeout(() => {
            statusIndicator.style.opacity = '0';
            setTimeout(() => {
                if (statusIndicator.parentNode) {
                    statusIndicator.parentNode.removeChild(statusIndicator);
                }
            }, 300);
        }, 3000);
    }
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        padding: 12px 24px;
        border-radius: 6px;
        font-weight: 500;
        z-index: 1001;
        transition: all 0.3s ease;
        max-width: 400px;
        text-align: center;
    `;
    
    switch(type) {
        case 'success':
            notification.style.background = '#28a745';
            notification.style.color = 'white';
            break;
        case 'error':
            notification.style.background = '#dc3545';
            notification.style.color = 'white';
            break;
        case 'warning':
            notification.style.background = '#ffc107';
            notification.style.color = '#212529';
            break;
        default:
            notification.style.background = '#17a2b8';
            notification.style.color = 'white';
    }
    
    document.body.appendChild(notification);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(-50%) translateY(-20px)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Export functions for global use
window.showNotification = showNotification;
window.checkSystemStatus = checkSystemStatus;