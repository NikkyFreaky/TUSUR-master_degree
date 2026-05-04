import { expect, test } from '@playwright/test';
import { openAnyListingCard, openBuyPage } from '../helpers/cian-actions.js';

test.describe('Функциональное тестирование', () => {
  test('FT02 Карточка объявления открывается по клику из каталога', async ({ page }) => {
    await openBuyPage(page);
    await openAnyListingCard(page);

    await expect(page.getByText(/₽|руб/i).first()).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText(/Описание|Характеристики|Об объекте/i).first()).toBeVisible({ timeout: 20_000 });
  });
});
