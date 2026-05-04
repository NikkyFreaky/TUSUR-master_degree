import { expect, test } from '@playwright/test';
import { openBuyPage } from '../helpers/cian-actions.js';

test('T11 Поведение страницы после жесткой перезагрузки', async ({ page }) => {
  await openBuyPage(page);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page.locator('a[href*="/sale/flat/"]').first()).toBeVisible({ timeout: 30_000 });
});
