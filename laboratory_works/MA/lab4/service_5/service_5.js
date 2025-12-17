const express = require("express");
const axios = require("axios");
const logger = require("./logger");

const app = express();
const PORT = 3000;

app.use((req, res, next) => {
  const requestId = req.get("X-Request-Id") || "missing";
  req.requestId = requestId;
  res.set("X-Request-Id", requestId);

  logger.info(`incoming ${req.method} ${req.path}`, { request_id: requestId });
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
    message: `Hello from service_5:${PORT}`,
  });
});

app.get("/process", async (req, res) => {
  const requestId = req.requestId;
  try {
    const fail = req.query.fail === "1" ? "1" : "0";

    logger.info("calling service_6 /work", { request_id: requestId });

    const r = await axios.get(`http://service_6:3001/work?fail=${fail}`, {
      headers: { "X-Request-Id": requestId },
      timeout: 5000,
    });

    logger.info("service_6 replied", { request_id: requestId });
    res.json({ ok: true, request_id: requestId, service_6: r.data });
  } catch (e) {
    logger.error("error calling service_6", {
      request_id: requestId,
      err: e.message,
    });
    res.status(500).json({ ok: false, request_id: requestId, err: e.message });
  }
});

app.listen(PORT, () => {
  logger.info(`service_5 started on ${PORT}`, { request_id: "bootstrap" });
});
