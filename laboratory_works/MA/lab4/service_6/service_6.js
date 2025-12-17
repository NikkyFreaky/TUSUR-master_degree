const express = require("express");
const logger = require("./logger");

const app = express();
const PORT = 3001;

app.use((req, res, next) => {
  const requestId = req.get("X-Request-Id") || "missing";
  req.requestId = requestId;
  res.set("X-Request-Id", requestId);

  logger.info(`incoming ${req.method} ${req.path}`, { request_id: requestId });

  // "Выход из второго микросервиса" — логируем после отправки ответа
  res.on("finish", () => {
    logger.info(`outgoing ${req.method} ${req.path} ${res.statusCode}`, {
      request_id: requestId,
    });
  });

  next();
});

app.get("/", (req, res) => {
  res.json({
    ok: true,
    request_id: req.requestId,
    message: `Hello from service_6:${PORT}`,
  });
});

app.get("/work", (req, res) => {
  const requestId = req.requestId;

  if (req.query.fail === "1") {
    logger.error("forced error in service_6", { request_id: requestId });
    return res
      .status(500)
      .json({ ok: false, request_id: requestId, err: "forced error" });
  }

  logger.info("work done", { request_id: requestId });
  return res.json({ ok: true, request_id: requestId, result: "ok" });
});

app.listen(PORT, () => {
  logger.info(`service_6 started on ${PORT}`, { request_id: "bootstrap" });
});
