// service1/index.js
const express = require("express");
const pino = require("pino");
const pinoHttp = require("pino-http");
const { v4: uuidv4 } = require("uuid");

const app = express();
const PORT = 3000;

// Базовый JSON-логгер (пишет в stdout)
const logger = pino({
  level: process.env.LOG_LEVEL || "info",
});

// HTTP-middleware: вытягиваем X-Request-Id или генерим UUID v4
app.use(
  pinoHttp({
    logger,
    genReqId: (req, res) => req.headers["x-request-id"] || uuidv4(),
    customProps: (req, res) => ({
      service: "service_1",
      env: process.env.NODE_ENV || "dev",
    }),
    customLogLevel: (req, res, err) => (err ? "error" : "info"),
    autoLogging: { ignorePaths: ["/health"] },
  })
);

// Отдаём ID обратно клиенту (удобно при дебаге)
app.use((req, res, next) => {
  res.setHeader("X-Request-Id", req.id);
  next();
});

app.get("/health", (req, res) => res.send("ok"));

app.get("/", (req, res) => {
  req.log.info({ path: "/", msg: "handling root" });
  res.send(`Hello from service 1 on port ${PORT}`);
});

// динамический import для CJS
const fetch = (...args) =>
  import("node-fetch").then(({ default: fetch }) => fetch(...args));

app.get("/call-svc2", async (req, res, next) => {
  try {
    const r = await fetch("http://service_2:3001/", {
      headers: { "X-Request-Id": req.id },
    });
    const text = await r.text();
    req.log.info({
      path: "/call-svc2",
      upstream: "service_2",
      status: r.status,
    });
    res.send(`svc1 -> svc2 OK: ${text}`);
  } catch (e) {
    next(e);
  }
});

app.listen(PORT, "0.0.0.0", () =>
  logger.info(`service_1 running on port ${PORT}`)
);
