const jwt = require('jsonwebtoken');
const config = require('../config/config');

/**
 * JWT Authorization Middleware
 * Validates JWT tokens for API authentication
 */
const authorizationMiddleware = (req, res, next) => {
  try {
    // Extract token from headers - prioritize Authorization header
    let token = req.headers['authorization'] || req.headers['x-access-token'];
    
    // Check if token is provided
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Access denied. No token provided.'
      });
    }
    
    // Remove Bearer prefix if present
    if (token.startsWith('Bearer ')) {
      token = token.slice(7);
    } else if (token === 'Bearer') {
      token = '';  // Handle case where token is just "Bearer" without space
    }
    
    // Validate token format - check if empty after Bearer removal
    if (!token || token.trim() === '' || token.length === 0) {
      return res.status(401).json({
        success: false,
        message: 'Access denied. Invalid token format.'
      });
    }
    
    // Verify JWT token
    jwt.verify(token, config.JWT_SECRET, (err, decoded) => {
      if (err) {
        // Handle specific JWT errors
        let message = 'Token is not valid';
        let statusCode = 401;
        
        if (err.name === 'TokenExpiredError') {
          message = 'Token has expired';
        } else if (err.name === 'JsonWebTokenError') {
          message = 'Malformed token';
        } else if (err.name === 'NotBeforeError') {
          message = 'Token not active yet';
        }
        
        return res.status(statusCode).json({
          success: false,
          message: message,
          error: err.name
        });
      }
      
      // Token is valid, attach decoded payload to request
      req.decoded = decoded;
      req.user = decoded; // Also provide as req.user for convenience
      next();
    });
    
  } catch (error) {
    // Handle unexpected errors
    console.error('Authorization middleware error:', error);
    return res.status(500).json({
      success: false,
      message: 'Internal server error during authentication'
    });
  }
};

module.exports = authorizationMiddleware;