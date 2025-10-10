const express = require('express');

const app = express();
const PORT = 3001;

app.get('/', (req, res) => {
  res.send(`Hello from service 2 on port ${PORT}`);
});

app.listen(PORT, () => console.log(`service 2 running on port ${PORT}`));
