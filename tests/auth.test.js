const request = require('supertest');
const jwt = require('jsonwebtoken');
const app = require('../server');
const config = require('../src/config/config');

describe('JWT Authentication Middleware', () => {
  let validToken;
  let expiredToken;

  beforeAll(() => {
    // Create a valid token for testing
    validToken = jwt.sign(
      { userId: 1, username: 'testuser' },
      config.JWT_SECRET,
      { expiresIn: '1h' }
    );

    // Create an expired token for testing
    expiredToken = jwt.sign(
      { userId: 1, username: 'testuser' },
      config.JWT_SECRET,
      { expiresIn: '-1h' } // Already expired
    );
  });

  describe('GET /api/protected/user', () => {
    test('should return 401 when no token is provided', async () => {
      const response = await request(app)
        .get('/api/protected/user')
        .expect(401);

      expect(response.body).toEqual({
        success: false,
        message: 'Access denied. No token provided.'
      });
    });

    test('should return 401 when empty token is provided', async () => {
      const response = await request(app)
        .get('/api/protected/user')
        .set('Authorization', 'Bearer ')
        .expect(401);

      expect(response.body).toEqual({
        success: false,
        message: 'Access denied. Invalid token format.'
      });
    });

    test('should return 401 when malformed token is provided', async () => {
      const response = await request(app)
        .get('/api/protected/user')
        .set('Authorization', 'Bearer invalid-token')
        .expect(401);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toBe('Malformed token');
      expect(response.body.error).toBe('JsonWebTokenError');
    });

    test('should return 401 when expired token is provided', async () => {
      const response = await request(app)
        .get('/api/protected/user')
        .set('Authorization', `Bearer ${expiredToken}`)
        .expect(401);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toBe('Token has expired');
      expect(response.body.error).toBe('TokenExpiredError');
    });

    test('should return 200 when valid token is provided with Bearer prefix', async () => {
      const response = await request(app)
        .get('/api/protected/user')
        .set('Authorization', `Bearer ${validToken}`)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.user.username).toBe('testuser');
    });

    test('should return 200 when valid token is provided with x-access-token header', async () => {
      const response = await request(app)
        .get('/api/protected/user')
        .set('x-access-token', validToken)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.user.username).toBe('testuser');
    });

    test('should prioritize authorization header over x-access-token', async () => {
      const response = await request(app)
        .get('/api/protected/user')
        .set('Authorization', `Bearer ${validToken}`)
        .set('x-access-token', 'invalid-token')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.user.username).toBe('testuser');
    });
  });

  describe('POST /api/authenticate', () => {
    test('should return valid token for correct credentials', async () => {
      const response = await request(app)
        .post('/api/authenticate')
        .send({ username: 'demo', password: 'password' })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.token).toBeDefined();
      expect(response.body.user.username).toBe('demo');
    });

    test('should return 401 for invalid credentials', async () => {
      const response = await request(app)
        .post('/api/authenticate')
        .send({ username: 'demo', password: 'wrongpassword' })
        .expect(401);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toBe('Invalid credentials');
    });

    test('should return 400 when username is missing', async () => {
      const response = await request(app)
        .post('/api/authenticate')
        .send({ password: 'password' })
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toBe('Username and password are required');
    });
  });
});