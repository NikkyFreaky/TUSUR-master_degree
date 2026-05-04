import { expect, test } from '@playwright/test';
import { openAnyListingCard, openBuyPage } from './helpers/cian-actions.js';

test.describe('Карточка объявления', () => {
  test('карточка открывается и содержит ключевые блоки', async ({ page }) => {
    await openBuyPage(page);
    await openAnyListingCard(page);

    await expect(page.getByText(/₽|руб/i).first()).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText(/Описание|Характеристики|Об объекте/i).first()).toBeVisible({ timeout: 20_000 });
  });
});
