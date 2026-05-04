import { expect, test } from '@playwright/test';

test('T10 Страница 404 и обработка несуществующих URL', async ({ page }) => {
  const response = await page.goto('/no-page-autotest-404', { waitUntil: 'domcontentloaded' });
  expect(response).toBeTruthy();
  expect(response!.status()).toBe(404);
  await expect(page.getByText(/404|Ошибка/i).first()).toBeVisible({ timeout: 20_000 });
});
