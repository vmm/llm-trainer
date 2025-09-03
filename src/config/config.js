/**
 * Configuration settings for the application
 */

// Load environment variables
require('dotenv').config();

const config = {
  // JWT Configuration
  JWT_SECRET: process.env.JWT_SECRET || 'your-default-jwt-secret-key-change-in-production',
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '24h',
  
  // API Configuration
  PORT: process.env.PORT || 3000,
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  // Database Configuration (if needed)
  DATABASE_URL: process.env.DATABASE_URL,
  
  // Other configuration options can be added here
};

// Validate required configuration
if (config.NODE_ENV === 'production' && config.JWT_SECRET === 'your-default-jwt-secret-key-change-in-production') {
  console.warn('WARNING: Using default JWT secret in production. Please set JWT_SECRET environment variable.');
}

module.exports = config;