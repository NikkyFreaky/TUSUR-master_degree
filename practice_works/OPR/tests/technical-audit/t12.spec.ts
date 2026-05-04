import { expect, test } from '@playwright/test';
import { openAnyListingCard, openBuyPage } from '../helpers/cian-actions.js';

test('T12 Загрузка галереи фотографий в карточке', async ({ page }) => {
  await openBuyPage(page);
  await openAnyListingCard(page);
  const photo = page.locator('img').first();
  await expect(photo).toBeVisible({ timeout: 30_000 });
  await photo.click({ timeout: 20_000 });
  const nextButton = page.getByRole('button', { name: /Следующее|Next/i }).first();
  if (await nextButton.isVisible().catch(() => false)) {
    await nextButton.click({ timeout: 20_000 });
  }
  await expect(page.locator('[role="dialog"], [class*="gallery" i]').first()).toBeVisible({ timeout: 20_000 });
});
