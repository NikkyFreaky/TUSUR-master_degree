import { expect, test } from '@playwright/test';
import { openBuyPage } from '../helpers/cian-actions.js';

test('T13 Работа ленивой загрукзи на сайте', async ({ page }) => {
  await openBuyPage(page);
  const beforeCount = await page.locator('a[href*="/sale/flat/"]').count();
  await page.mouse.wheel(0, 10_000);
  await page.waitForTimeout(1500);
  const afterCount = await page.locator('a[href*="/sale/flat/"]').count();
  expect(afterCount).toBeGreaterThanOrEqual(beforeCount);
});
