require('dotenv').config();
const express = require('express');
const amqp = require('amqplib/callback_api');
const axios = require('axios');
const app = express();

const port = process.env.PORT || 3000;
const serviceName = process.env.SERVICE_NAME || 'service1';
const rabbitUrl =
  process.env.RABBITMQ_URL || 'amqp://guest:guest@rabbitmq:5672';
const queue = process.env.QUEUE_NAME || 'service_1-queue';
const AUTH_URL = process.env.AUTH_URL || 'http://auth:3000';

app.get('/', (req, res) =>
  res.send(`${serviceName}: root — Hello from ${serviceName}`)
);
app.get('/info', (req, res) =>
  res.json({service: serviceName, time: new Date().toISOString()})
);
app.get('/hello/:name?', (req, res) =>
  res.send(`Hello ${req.params.name || 'Guest'} from ${serviceName}`)
);

async function authMiddleware(req, res, next) {
  try {
    const auth = req.headers['authorization'] || '';
    if (!auth.startsWith('Bearer '))
      return res
        .status(401)
        .json({error: 'Missing or invalid Authorization header'});
    const token = auth.slice(7);
    const resp = await axios.post(
      `${AUTH_URL}/validate`,
      {token},
      {headers: {Authorization: `Bearer ${token}`}, timeout: 3000}
    );
    if (resp.data && resp.data.valid) {
      req.user = resp.data.user;
      return next();
    }
    return res.status(401).json({error: 'Invalid token'});
  } catch (err) {
    if (err.response && err.response.status === 401)
      return res.status(401).json({error: 'Invalid token'});
    console.error('Auth error:', err.message || err);
    return res.status(500).json({error: 'Auth service error'});
  }
}

app.get('/protected', authMiddleware, (req, res) => {
  res.json({
    ok: true,
    msg: `Hello ${req.user.username}, you accessed protected data!`,
    user: req.user,
  });
});

app.listen(port, () => {
  console.log(`${serviceName} listening on port ${port}`);
});

amqp.connect(rabbitUrl, function (error0, connection) {
  if (error0) {
    console.log('Rabbit connect error (service1):', error0.message || error0);
    return;
  }
  connection.createChannel(function (error1, channel) {
    if (error1) {
      console.log('Channel error (service1):', error1);
      return;
    }
    channel.assertQueue(queue, {durable: false});
    const msg = 'Hello from service1';
    channel.sendToQueue(queue, Buffer.from(msg));
    console.log('Sent %s', msg);
    setTimeout(() => {
      try {
        connection.close();
      } catch (e) {}
    }, 500);
  });
});
