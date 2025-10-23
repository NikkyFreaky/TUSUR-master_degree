### Создание билда с передачей в него .env файла и запуск контейнера

```bash
docker compose --env-file ../.env up -d --build
```

### Создание бэкапа:

```bash
docker exec mysql-db mysqldump -u user -ppassword lab7_db > ./dump.sql
```

### Удаление volume с базой данных

```bash
docker compose down -v
```

### Восстановление базы данных из бэкапа

```bash
docker exec -i mysql-db mysql -u user -ppassword lab7_db < ./dump.sql
```
