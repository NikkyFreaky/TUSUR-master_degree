set -e

echo "Build and start service_1"
(cd service_1 && docker compose up -d --build)

echo "Build and start service_2"
(cd service_2 && docker compose up -d --build)

echo "Build and start gateway"
(cd gateway && docker compose up -d --build)
