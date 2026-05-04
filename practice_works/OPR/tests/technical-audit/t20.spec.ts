import { expect, test } from '@playwright/test';

test('T20 Корректность загрузки служебных файлов сайта', async ({ page }) => {
  const robots = await page.goto('https://tomsk.cian.ru/robots.txt', { waitUntil: 'domcontentloaded' });
  expect(robots).toBeTruthy();
  expect(robots!.status()).toBe(200);
  const sitemap = await page.goto('https://tomsk.cian.ru/sitemap.xml', { waitUntil: 'domcontentloaded' });
  expect(sitemap).toBeTruthy();
  expect(sitemap!.status()).toBe(200);
});
