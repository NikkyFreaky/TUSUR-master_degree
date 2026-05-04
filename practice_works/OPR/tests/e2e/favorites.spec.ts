import { expect, test } from '@playwright/test';
import { humanPause } from '../helpers/slow-actions.js';
import { hasAuthorizedSession, openBuyPage } from './helpers/cian-actions.js';

test.describe('Избранное', () => {
  test('можно добавить и убрать объявление в избранное из выдачи', async ({ page }) => {
    const isAuthorized = await hasAuthorizedSession(page);
    expect(isAuthorized, 'Сессия не авторизована. Выполните setup-проект: npm run test:setup').toBeTruthy();

    await openBuyPage(page);

    const firstListing = page.locator('article').first();
    const addToFavorites = firstListing
      .getByRole('button', { name: /Избран|Сохранить/i })
      .or(page.getByRole('button', { name: /Избран|Сохранить/i }).first());

    await expect(addToFavorites).toBeVisible({ timeout: 30_000 });
    const beforeFirstClick = await addToFavorites.getAttribute('aria-pressed');
    await addToFavorites.click({ timeout: 20_000 });
    await humanPause(1000);
    const afterFirstClick = await addToFavorites.getAttribute('aria-pressed');

    if (beforeFirstClick !== null && afterFirstClick !== null) {
      expect(afterFirstClick).not.toBe(beforeFirstClick);
    } else {
      const saveOrRemoveToast = page
        .getByText(/Добавлено в избранное|Сохранено в избранное|Удалено из избранного|Убрали из избранного/i)
        .first();
      await expect(saveOrRemoveToast).toBeVisible({ timeout: 20_000 });
    }

    const beforeSecondClick = await addToFavorites.getAttribute('aria-pressed');
    await addToFavorites.click({ timeout: 20_000 });
    await humanPause(1000);
    const afterSecondClick = await addToFavorites.getAttribute('aria-pressed');

    if (beforeSecondClick !== null && afterSecondClick !== null) {
      expect(afterSecondClick).not.toBe(beforeSecondClick);
    } else {
      const saveOrRemoveToast = page
        .getByText(/Добавлено в избранное|Сохранено в избранное|Удалено из избранного|Убрали из избранного/i)
        .first();
      await expect(saveOrRemoveToast).toBeVisible({ timeout: 20_000 });
    }
  });
});
