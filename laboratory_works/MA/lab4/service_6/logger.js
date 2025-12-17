const fs = require("fs");
const path = require("path");
const winston = require("winston");

const logDir = process.env.LOG_DIR || "/var/log/service_6";
fs.mkdirSync(logDir, { recursive: true });

module.exports = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: path.join(logDir, "app.log") }),
  ],
});
