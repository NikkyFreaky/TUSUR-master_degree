import { expect, test } from '@playwright/test';
import { applyBaseFilters, openBuyPage } from '../helpers/cian-actions.js';

test.describe('Функциональное тестирование', () => {
  test('FT04 Фильтры по цене, площади, комнатности и типу жилья', async ({ page }) => {
    await openBuyPage(page);
    await applyBaseFilters(page);

    await expect(page).toHaveURL(/price|maxprice|minprice|room|cat\.php|kupit/i);
    const listingLinks = page.locator('a[href*="/sale/flat/"]');
    await expect(listingLinks.first()).toBeVisible({ timeout: 30_000 });
  });
});
