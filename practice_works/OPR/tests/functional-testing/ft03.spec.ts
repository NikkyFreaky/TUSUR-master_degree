import { expect, test } from '@playwright/test';
import { openBuyPage, searchByStreet } from '../helpers/cian-actions.js';

test.describe('Функциональное тестирование', () => {
  test('FT03 Поиск по локации или запросу', async ({ page }) => {
    test.setTimeout(90_000);

    await openBuyPage(page);
    await searchByStreet(page, 'Ленина');

    const listingLinks = page.locator('a[href*="/sale/flat/"]');
    await expect(listingLinks.first()).toBeVisible({ timeout: 45_000 });
  });
});
