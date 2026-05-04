import { expect, type Locator, type Page } from '@playwright/test';
import { humanPause } from '../helpers/slow-actions.js';

export async function getFavoriteButton(page: Page) {
  const firstListing = page.locator('article').first();
  return firstListing
    .getByRole('button', { name: /Избран|Сохранить/i })
    .or(page.getByRole('button', { name: /Избран|Сохранить/i }).first());
}

export async function expectFavoriteToggle(addToFavorites: Locator, page: Page): Promise<void> {
  const beforeClick = await addToFavorites.getAttribute('aria-pressed');
  await addToFavorites.click({ timeout: 20_000 });
  await humanPause(1000);
  const afterClick = await addToFavorites.getAttribute('aria-pressed');

  if (beforeClick !== null && afterClick !== null) {
    expect(afterClick).not.toBe(beforeClick);
    return;
  }

  const saveOrRemoveToast = page
    .getByText(/Добавлено в избранное|Сохранено в избранное|Удалено из избранного|Убрали из избранного/i)
    .first();
  await expect(saveOrRemoveToast).toBeVisible({ timeout: 20_000 });
}
