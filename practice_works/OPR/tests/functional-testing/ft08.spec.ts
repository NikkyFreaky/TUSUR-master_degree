import { expect, test } from '@playwright/test';
import { humanPause } from '../helpers/slow-actions.js';
import { hasAuthorizedSession, openBuyPage } from '../helpers/cian-actions.js';
import { expectFavoriteToggle, getFavoriteButton } from './favorites.helpers.js';

test.describe('Функциональное тестирование', () => {
  test('FT08 Удаление объекта из избранного', async ({ page }) => {
    const isAuthorized = await hasAuthorizedSession(page);
    expect(isAuthorized, 'Session is not authorized. Run setup project: npm run test:setup').toBeTruthy();

    await openBuyPage(page);
    const addToFavorites = await getFavoriteButton(page);
    await expect(addToFavorites).toBeVisible({ timeout: 30_000 });

    const currentState = await addToFavorites.getAttribute('aria-pressed');
    if (currentState !== 'true') {
      await addToFavorites.click({ timeout: 20_000 });
      await humanPause(1000);
    }

    await expectFavoriteToggle(addToFavorites, page);
  });
});
