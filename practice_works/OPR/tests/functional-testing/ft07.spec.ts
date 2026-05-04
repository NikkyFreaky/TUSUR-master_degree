import { expect, test } from '@playwright/test';
import { hasAuthorizedSession, openBuyPage } from '../helpers/cian-actions.js';
import { expectFavoriteToggle, getFavoriteButton } from './favorites.helpers.js';

test.describe('Функциональное тестирование', () => {
  test('FT07 Добавление объекта в избранное', async ({ page }) => {
    const isAuthorized = await hasAuthorizedSession(page);
    expect(isAuthorized, 'Session is not authorized. Run setup project: npm run test:setup').toBeTruthy();

    await openBuyPage(page);
    const addToFavorites = await getFavoriteButton(page);
    await expect(addToFavorites).toBeVisible({ timeout: 30_000 });
    await expectFavoriteToggle(addToFavorites, page);
  });
});
