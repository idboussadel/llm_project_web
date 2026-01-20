/**
 * Main Application JavaScript
 * Base functionality for all pages
 */

// Utility: Format numbers
function formatNumber(num, decimals = 2) {
  return Number(num).toFixed(decimals);
}

// Utility: Format percentage
function formatPercent(num, decimals = 2) {
  return `${formatNumber(num * 100, decimals)}%`;
}

// Utility: Show toast notification (optional enhancement)
function showToast(message, type = "info") {
  console.log(`[${type.toUpperCase()}] ${message}`);
  // Could implement actual toast UI here
}

// Utility: Debounce function
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Export utilities
window.AppUtils = {
  formatNumber,
  formatPercent,
  showToast,
  debounce,
};
