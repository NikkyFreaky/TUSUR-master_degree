import { expect, test } from '@playwright/test';

test('T19 Обработка пустой выдачи и пограничных параметров', async ({ page }) => {
  const response = await page.goto('/kupit/?minprice=9999999999', { waitUntil: 'domcontentloaded' });
  expect(response).toBeTruthy();
  expect(response!.status()).toBeLessThan(500);
  await expect(page.locator('body')).toBeVisible();
  await expect(page).toHaveURL(/kupit|cat\.php/i);
});
