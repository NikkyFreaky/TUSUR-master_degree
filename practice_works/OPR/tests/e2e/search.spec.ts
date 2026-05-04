import { expect, test } from '@playwright/test';
import { openBuyPage, searchByStreet } from './helpers/cian-actions.js';

test.describe('Поиск объявлений', () => {
  test('поиск по улице возвращает выдачу', async ({ page }) => {
    await openBuyPage(page);
    await searchByStreet(page, 'Ленина');

    const listingLinks = page.locator('a[href*="/sale/flat/"]');
    await expect(listingLinks.first()).toBeVisible({ timeout: 30_000 });
  });
});
