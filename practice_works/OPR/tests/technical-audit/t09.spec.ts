import { expect, test } from '@playwright/test';

test('T09 Корректность редиректов на ключевых URL', async ({ page }) => {
  const kupitResp = await page.goto('https://www.cian.ru/kupit', { waitUntil: 'domcontentloaded' });
  expect(kupitResp).toBeTruthy();
  await expect(page).toHaveURL(/\/kupit\/?/i);
});
