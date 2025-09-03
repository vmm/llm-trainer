/**
 * Frontend JavaScript for LLM Trainer Web Interface
 * 
 * This file provides client-side functionality for the LLM Trainer web interface,
 * complementing the Gradio backend with additional interactive features.
 */

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('LLM Trainer web interface initialized');
    initializeInterface();
});

/**
 * Initialize the web interface
 */
function initializeInterface() {
    setupEventListeners();
    initializeModelStatus();
    setupTooltips();
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // Model loading status
    const statusElements = document.querySelectorAll('.model-status');
    statusElements.forEach(element => {
        element.addEventListener('click', updateModelStatus);
    });
    
    // Question input enhancements
    const questionInputs = document.querySelectorAll('textarea[placeholder*="reasoning"]');
    questionInputs.forEach(input => {
        input.addEventListener('input', validateQuestionInput);
    });
}

/**
 * Initialize model status display
 */
function initializeModelStatus() {
    const statusContainer = document.createElement('div');
    statusContainer.id = 'model-status-container';
    statusContainer.innerHTML = `
        <div class="status-indicator">
            <span class="status-dot ready"></span>
            <span class="status-text">Model Ready</span>
        </div>
    `;
    
    // Insert at the top of the page if gradio interface exists
    const gradioContainer = document.querySelector('.gradio-container') || document.body;
    gradioContainer.insertBefore(statusContainer, gradioContainer.firstChild);
}

/**
 * Update model status indicator
 */
function updateModelStatus() {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    if (statusDot && statusText) {
        statusDot.className = 'status-dot loading';
        statusText.textContent = 'Processing...';
        
        // Simulate status check (in real implementation, this would call the backend)
        setTimeout(() => {
            statusDot.className = 'status-dot ready';
            statusText.textContent = 'Model Ready';
        }, 2000);
    }
}

/**
 * Validate question input and provide feedback
 */
function validateQuestionInput(event) {
    const input = event.target;
    const value = input.value.trim();
    
    // Remove existing feedback
    const existingFeedback = input.parentNode.querySelector('.input-feedback');
    if (existingFeedback) {
        existingFeedback.remove();
    }
    
    // Add feedback for reasoning questions
    if (value.length > 10) {
        const feedback = document.createElement('div');
        feedback.className = 'input-feedback';
        feedback.textContent = 'Good! Reasoning questions work best with clear, specific scenarios.';
        input.parentNode.appendChild(feedback);
    }
}

/**
 * Setup tooltips for interface elements
 */
function setupTooltips() {
    // Add tooltips to slider controls
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        slider.addEventListener('input', function() {
            updateSliderTooltip(this);
        });
    });
}

/**
 * Update slider tooltip with current value
 */
function updateSliderTooltip(slider) {
    let tooltip = slider.nextElementSibling;
    if (!tooltip || !tooltip.classList.contains('slider-tooltip')) {
        tooltip = document.createElement('div');
        tooltip.className = 'slider-tooltip';
        slider.parentNode.insertBefore(tooltip, slider.nextSibling);
    }
    tooltip.textContent = slider.value;
}

/**
 * Utility function to show notifications
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

/**
 * Export functions for external use
 */
window.LLMTrainerInterface = {
    initializeInterface,
    updateModelStatus,
    showNotification
};