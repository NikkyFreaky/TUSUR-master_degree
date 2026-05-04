import { expect, test } from '@playwright/test';
import { openAnyListingCard, openBuyPage } from '../helpers/cian-actions.js';

test('T14 Работа истории браузера', async ({ page }) => {
  await openBuyPage(page);
  const startUrl = page.url();
  await openAnyListingCard(page);
  const cardUrl = page.url();
  expect(cardUrl).not.toBe(startUrl);
  await page.goBack({ waitUntil: 'domcontentloaded' });
  await expect(page).toHaveURL(new RegExp(startUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').slice(0, 40)));
  await page.goForward({ waitUntil: 'domcontentloaded' });
  await expect(page).toHaveURL(/\/sale\/flat\//);
});
