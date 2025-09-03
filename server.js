const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const jwt = require('jsonwebtoken');
const config = require('./src/config/config');
const authorizationMiddleware = require('./src/auth/authorizationMiddleware');

const app = express();

// Security middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Authentication endpoint (for login/getting tokens)
app.post('/api/authenticate', (req, res) => {
  // This is a demo endpoint - in production, you'd validate credentials
  const { username, password } = req.body;
  
  // Demo validation (replace with real authentication logic)
  if (!username || !password) {
    return res.status(400).json({
      success: false,
      message: 'Username and password are required'
    });
  }
  
  // In production, validate against your user database
  if (username === 'demo' && password === 'password') {
    // Create JWT token
    const payload = {
      userId: 1,
      username: username,
      iat: Math.floor(Date.now() / 1000)
    };
    
    const token = jwt.sign(payload, config.JWT_SECRET, {
      expiresIn: config.JWT_EXPIRES_IN
    });
    
    res.json({
      success: true,
      message: 'Authentication successful',
      token: token,
      user: {
        id: payload.userId,
        username: payload.username
      }
    });
  } else {
    res.status(401).json({
      success: false,
      message: 'Invalid credentials'
    });
  }
});

// Protected endpoints that require authentication
app.use('/api/protected', authorizationMiddleware);

// Example protected endpoint
app.get('/api/protected/user', (req, res) => {
  res.json({
    success: true,
    message: 'Access granted to protected resource',
    user: req.user
  });
});

// Example protected endpoint for LLM operations
app.get('/api/protected/models', (req, res) => {
  res.json({
    success: true,
    message: 'Available models',
    models: [
      { id: 1, name: 'llama3-8b', status: 'available' },
      { id: 2, name: 'llama3-reasoning', status: 'training' }
    ],
    user: req.user.username
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Something went wrong!'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

const PORT = config.PORT;
app.listen(PORT, () => {
  console.log(`LLM Trainer API Server running on port ${PORT}`);
  console.log(`Environment: ${config.NODE_ENV}`);
});

module.exports = app;