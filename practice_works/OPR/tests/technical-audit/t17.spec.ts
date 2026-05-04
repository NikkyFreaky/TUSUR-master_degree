import { expect, test } from '@playwright/test';
import { expectStatusOk, openGuestPage } from './helpers.js';

test('T17 Техническая обработка авторизации по коду/телефону', async ({ browser }) => {
  const page = await openGuestPage(browser);
  await expectStatusOk(page, '/');
  if (/captcha|blocked|tmgrdfrend/i.test(page.url())) {
    test.skip(true, 'Auth flow is blocked by anti-bot/captcha on current run.');
  }

  const loginEntry = page
    .getByRole('link', { name: /Войти|Вход/i })
    .or(page.getByRole('button', { name: /Войти|Вход/i }))
    .first();

  if (!(await loginEntry.isVisible().catch(() => false))) {
    test.skip(true, 'Login entry is not visible in current regional/layout state.');
  }

  await loginEntry.click({ timeout: 20_000 });
  const phone = page.getByLabel(/Телефон/i).or(page.getByPlaceholder(/Телефон/i)).first();
  await expect(phone).toBeVisible({ timeout: 20_000 });
  await phone.fill('123');
  const sendCode = page.getByRole('button', { name: /Получить код|Продолжить/i }).first();
  if (await sendCode.isVisible().catch(() => false)) {
    await sendCode.click();
  }
  await expect(page.getByText(/некорректн|проверьте|ошибк/i).first()).toBeVisible({ timeout: 20_000 });
  await page.context().close();
});
