const express = require('express');

const app = express();
const PORT = 3000;

app.get('/', (req, res) => {
  res.send(`Hello from service 1 on port ${PORT}`);
});

app.listen(PORT, () => console.log(`service 1 running on port ${PORT}`));
