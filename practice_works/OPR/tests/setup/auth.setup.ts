import { expect, test } from '@playwright/test';
import { humanPause } from '../helpers/slow-actions.js';
import 'dotenv/config';

const phoneNumber = process.env.CIAN_PHONE;

if (!phoneNumber) {
  throw new Error('CIAN_PHONE is not set. Add it to .env file.');
}

async function isLoggedIn(page: import('@playwright/test').Page): Promise<boolean> {
  const byUrl = /my\.cian\.ru|profile|lk/i.test(page.url());
  if (byUrl) return true;

  for (let i = 0; i < 3; i++) {
    try {
      const visibleProfileLinks = page.locator('a[href*="my.cian.ru"]:visible');
      return (await visibleProfileLinks.count()) > 0;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!/Execution context was destroyed/i.test(message) || i === 2) throw error;
      await page.waitForTimeout(250);
    }
  }

  return false;
}

async function isRateLimited(page: import('@playwright/test').Page): Promise<boolean> {
  const lockTitle = page.getByText(/Подождите\s*15\s*минут/i).first();
  return lockTitle.isVisible().catch(() => false);
}

async function hasUnexpectedAuthError(page: import('@playwright/test').Page): Promise<boolean> {
  const err = page.getByText(/Произошла непредвиденная ошибка/i).first();
  return err.isVisible().catch(() => false);
}

test('ручная авторизация и сохранение storage state', async ({ page }) => {
  test.setTimeout(300_000);

  await page.goto('https://www.cian.ru/');
  await humanPause(1200);

  if (await isLoggedIn(page)) {
    await page.context().storageState({ path: 'playwright/.auth/user.json' });
    return;
  }

  const loginEntry = page
    .getByRole('link', { name: /Войти|Вход/i })
    .or(page.getByRole('button', { name: /Войти|Вход/i }))
    .first();

  await expect(loginEntry).toBeVisible({ timeout: 30_000 });
  await loginEntry.click({ timeout: 30_000 });
  await humanPause(1000);

  if (await isRateLimited(page)) {
    throw new Error(
      'Cian temporarily blocked OTP requests ("Подождите 15 минут"). Wait and rerun setup once, without repeated attempts.'
    );
  }

  if (await hasUnexpectedAuthError(page)) {
    throw new Error(
      'Cian auth modal returned "Произошла непредвиденная ошибка". Close modal, wait 1-2 minutes, and rerun setup.'
    );
  }

  const phoneInput = page
    .getByLabel(/Телефон/i)
    .or(page.getByPlaceholder(/Телефон/i))
    .first();

  await expect(phoneInput).toBeVisible({ timeout: 30_000 });
  await expect(phoneInput).toBeEnabled();
  await phoneInput.fill(phoneNumber);
  await humanPause(600);

  const sendCode = page
    .getByRole('button', { name: /Получить код|Продолжить/i })
    .first();

  if (await isRateLimited(page)) {
    throw new Error(
      'Cian temporarily blocked OTP requests ("Подождите 15 минут"). Wait and rerun setup once, without repeated attempts.'
    );
  }

  if (await hasUnexpectedAuthError(page)) {
    throw new Error(
      'Cian auth modal returned "Произошла непредвиденная ошибка". Close modal, wait 1-2 minutes, and rerun setup.'
    );
  }

  if (await sendCode.isVisible().catch(() => false)) {
    await expect(sendCode).toBeEnabled();
    await sendCode.click({ timeout: 20_000 });
  }

  console.log('Введите SMS-код в открытом браузере. Ожидание успешной авторизации до 3 минут...');

  await expect
    .poll(
      async () => {
        if (await isLoggedIn(page)) {
          return true;
        }

        const currentUrl = page.url();
        return /my\.cian\.ru|profile|lk/i.test(currentUrl);
      },
      {
        timeout: 180_000,
        intervals: [1000, 1500, 2000]
      }
    )
    .toBeTruthy();

  await page.context().storageState({ path: 'playwright/.auth/user.json' });
});
