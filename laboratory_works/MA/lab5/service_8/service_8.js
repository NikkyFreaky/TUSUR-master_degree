const express = require("express");

const app = express();
const PORT = 3001;

app.get("/", (req, res) => {
  res.send(`Hello from service 8 on port ${PORT}`);
});

app.listen(PORT, "0.0.0.0", () =>
  console.log(`service 8 running on port ${PORT}`)
);
