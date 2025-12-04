require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const fs = require('fs-extra');
const path = require('path');
const cors = require('cors');

const USERS_FILE = path.join(__dirname, 'users.json');
const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret';
const TOKEN_TTL = process.env.TOKEN_TTL || '1h';

const app = express();
app.use(bodyParser.json());
app.use(cors());

async function ensureUsersFile() {
  if (!await fs.pathExists(USERS_FILE)) {
    await fs.writeJson(USERS_FILE, []);
  }
}
ensureUsersFile();

async function readUsers() {
  await ensureUsersFile();
  return fs.readJson(USERS_FILE);
}
async function writeUsers(data) {
  return fs.writeJson(USERS_FILE, data, { spaces: 2 });
}

app.post('/register', async (req, res) => {
  const { username, password, fullName } = req.body;
  if (!username || !password) return res.status(400).json({ error: 'username and password required' });

  const users = await readUsers();
  if (users.find(u => u.username === username)) return res.status(409).json({ error: 'user exists' });

  const hash = await bcrypt.hash(password, 10);
  const user = { id: Date.now(), username, passwordHash: hash, fullName: fullName || '', createdAt: new Date().toISOString() };
  users.push(user);
  await writeUsers(users);

  res.json({ ok: true, user: { id: user.id, username: user.username, fullName: user.fullName } });
});

app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  const users = await readUsers();
  const user = users.find(u => u.username === username);
  if (!user) return res.status(401).json({ error: 'invalid credentials' });

  const ok = await bcrypt.compare(password, user.passwordHash);
  if (!ok) return res.status(401).json({ error: 'invalid credentials' });

  const token = jwt.sign({ userId: user.id, username: user.username }, JWT_SECRET, { expiresIn: TOKEN_TTL });
  res.json({ token });
});

app.post('/validate', async (req, res) => {
  let token = req.body && req.body.token;
  if (!token) {
    const auth = req.headers['authorization'] || '';
    if (auth.startsWith('Bearer ')) token = auth.slice(7);
  }
  if (!token) return res.status(400).json({ valid: false, error: 'no token provided' });

  try {
    const payload = jwt.verify(token, JWT_SECRET);
    const users = await readUsers();
    const user = users.find(u => u.id === payload.userId);
    if (!user) return res.status(401).json({ valid: false, error: 'user not found' });
    res.json({ valid: true, user: { id: user.id, username: user.username, fullName: user.fullName } });
  } catch (err) {
    res.status(401).json({ valid: false, error: 'invalid token' });
  }
});

app.get('/', (req, res) => res.send('Auth service ready'));

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`auth service listening on ${port}`);
});
