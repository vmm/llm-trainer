# JWT Authentication API

This document describes the JWT authentication system implemented for the LLM Trainer API.

## Overview

The authentication system provides secure access to the LLM Trainer API endpoints using JSON Web Tokens (JWT). It addresses the following security concerns:

- Missing authentication tokens
- Malformed tokens
- Expired tokens
- Invalid tokens

## File Structure

```
src/
├── auth/
│   └── authorizationMiddleware.js    # JWT authentication middleware
├── config/
│   └── config.js                     # Configuration settings
server.js                            # Main API server
tests/
└── auth.test.js                     # Authentication tests
```

## Authentication Middleware

The `authorizationMiddleware.js` file handles JWT token validation with comprehensive error handling:

### Features

1. **Token Extraction**: Supports both `Authorization: Bearer <token>` and `x-access-token` headers
2. **Token Validation**: Checks for missing, empty, or malformed tokens
3. **JWT Verification**: Validates token signature and expiration
4. **Error Handling**: Provides specific error messages for different failure scenarios
5. **User Context**: Attaches decoded user information to the request object

### Error Responses

| Scenario | Status Code | Message |
|----------|-------------|---------|
| No token provided | 401 | "Access denied. No token provided." |
| Empty token | 401 | "Access denied. Invalid token format." |
| Malformed token | 401 | "Malformed token" |
| Expired token | 401 | "Token has expired" |
| Invalid signature | 401 | "Token is not valid" |
| Server error | 500 | "Internal server error during authentication" |

## API Endpoints

### Authentication Endpoint

```
POST /api/authenticate
```

**Request Body:**
```json
{
  "username": "demo",
  "password": "password"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Authentication successful",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "demo"
  }
}
```

### Protected Endpoints

All endpoints under `/api/protected/*` require authentication.

**Example Request:**
```bash
curl -H "Authorization: Bearer <your-jwt-token>" \
     http://localhost:3000/api/protected/user
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Access granted to protected resource",
  "user": {
    "userId": 1,
    "username": "demo"
  }
}
```

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=24h
PORT=3000
NODE_ENV=development
```

### Security Notes

1. **JWT_SECRET**: Must be a strong, random string in production
2. **Token Expiration**: Configure appropriate expiration times for your use case
3. **HTTPS**: Always use HTTPS in production
4. **Token Storage**: Store tokens securely on the client side

## Usage

### Starting the Server

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Start production server
npm start
```

### Running Tests

```bash
npm test
```

## Example Usage

### 1. Get Authentication Token

```bash
curl -X POST http://localhost:3000/api/authenticate \
     -H "Content-Type: application/json" \
     -d '{"username": "demo", "password": "password"}'
```

### 2. Access Protected Resource

```bash
curl -H "Authorization: Bearer <your-token>" \
     http://localhost:3000/api/protected/models
```

## Integration with LLM Trainer

This authentication system can be integrated with the existing Python-based LLM Trainer framework to provide secure API access for:

- Model training management
- Dataset operations
- Evaluation results
- Model deployment

## Error Handling

The middleware provides comprehensive error handling for all JWT-related issues mentioned in the original Sentry error report:

- **UnauthorizedError: Invalid token** - Now properly categorized and handled
- **Missing tokens** - Clear error message provided
- **Malformed tokens** - Detected and reported
- **Expired tokens** - Specific error message for expired tokens