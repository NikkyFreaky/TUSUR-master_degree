import { expect, test } from '@playwright/test';
import { openBuyPage, searchByStreet } from '../helpers/cian-actions.js';

test('T07 Техническая обработка форм и действий', async ({ page }) => {
  await openBuyPage(page);
  await searchByStreet(page, 'Ленина');
  await expect(page.locator('a[href*="/sale/flat/"]').first()).toBeVisible({ timeout: 30_000 });
});
