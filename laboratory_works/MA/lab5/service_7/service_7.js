const express = require("express");

const app = express();
const PORT = 3000;

app.get("/", (req, res) => {
  res.send(`Hello from service 7 on port ${PORT}`);
});

app.listen(PORT, "0.0.0.0", () =>
  console.log(`service 7 running on port ${PORT}`)
);
