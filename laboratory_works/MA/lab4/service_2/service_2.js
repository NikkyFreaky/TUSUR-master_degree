// service2/index.js
const express = require("express");
const pino = require("pino");
const pinoHttp = require("pino-http");
const { v4: uuidv4 } = require("uuid");

const app = express();
const PORT = 3001;
const logger = pino({ level: process.env.LOG_LEVEL || "info" });

app.use(
  pinoHttp({
    logger,
    genReqId: (req) => req.headers["x-request-id"] || uuidv4(),
    customProps: () => ({
      service: "service_2",
      env: process.env.NODE_ENV || "dev",
    }),
    customLogLevel: (req, res, err) => (err ? "error" : "info"),
    autoLogging: { ignorePaths: ["/health"] },
  })
);

app.use((req, res, next) => {
  res.setHeader("X-Request-Id", req.id);
  next();
});

app.get("/health", (req, res) => res.send("ok"));
app.get("/", (req, res) => {
  req.log.info({ path: "/", msg: "handling root" });
  res.send(`Hello from service 2 on port ${PORT}`);
});

app.listen(PORT, "0.0.0.0", () =>
  logger.info(`service_2 running on port ${PORT}`)
);
