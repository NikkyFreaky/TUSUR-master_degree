import { expect, test } from '@playwright/test';
import { openAnyListingCard, openBuyPage } from '../helpers/cian-actions.js';

test('T18 Стабильность после обновления страницы в карточке и каталоге', async ({ page }) => {
  await openBuyPage(page);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page.locator('a[href*="/sale/flat/"]').first()).toBeVisible({ timeout: 30_000 });
  await openAnyListingCard(page);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page.getByText(/₽|руб/i).first()).toBeVisible({ timeout: 20_000 });
});
