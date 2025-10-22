import express from 'express';
import connection from './db.js';

const app = express();
app.use(express.json());

app.get('/students', async (req, res) => {
  const [rows] = await connection.query('SELECT * FROM students');
  res.json(rows);
});

app.post('/students', async (req, res) => {
  const {name, group_number} = req.body;
  await connection.query(
    'INSERT INTO students (name, group_number) VALUES (?, ?)',
    [name, group_number]
  );
  res.status(201).json({message: 'Student added successfully'});
});

app.listen(3000, () => {
  console.log('API service running on port 3000');
});
