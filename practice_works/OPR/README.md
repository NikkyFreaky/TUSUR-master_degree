# Cian E2E (Playwright + TypeScript)

Минимальный набор e2e и smoke тестов для `cian.ru`.

## Требования

- Node.js 18+
- npm

## Установка

```bash
npm install
npx playwright install
```

## Настройка

1. Создайте `.env` из шаблона:

```bash
copy .env.example .env
```

2. Укажите номер телефона в `.env`:

```env
CIAN_PHONE=79XXXXXXXXX
```

## Авторизация (OTP)

```bash
npm run test:setup
```

В открывшемся браузере введите код из SMS. После входа сессия сохранится в `playwright/.auth/user.json`.

## Запуск тестов

```bash
npm test
```

Полезные команды:

- `npm run test:ui` - UI режим Playwright
- `npm run test:live` - запуск с видимым браузером
- `npm run test:live:video` - видимый браузер + видео каждого теста
- `npm run report` - открыть HTML-отчет
