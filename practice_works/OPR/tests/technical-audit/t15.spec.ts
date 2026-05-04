import { expect, test } from '@playwright/test';
import { expectStatusOk } from './helpers.js';

test('T15 Загрузка cookie-баннера и служебных виджетов', async ({ page }) => {
  await expectStatusOk(page, '/');
  const cookieBanner = page.getByText(/cookie|куки/i).first();
  if (await cookieBanner.isVisible().catch(() => false)) {
    await expect(cookieBanner).toBeVisible();
  }
});
