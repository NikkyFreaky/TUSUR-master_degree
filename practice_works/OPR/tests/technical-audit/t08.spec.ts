import { expect, test } from '@playwright/test';
import { initAudit } from './helpers.js';

test('T08 Безопасное соединение и отсутствие смешанного контента', async ({ page }) => {
  const audit = initAudit(page);
  const httpResp = await page.goto('http://www.cian.ru', { waitUntil: 'domcontentloaded' });
  expect(httpResp).toBeTruthy();
  await expect(page).toHaveURL(/^https:\/\//i);
  const mixedContentErrors = audit.consoleErrors.filter((x) => /mixed content/i.test(x));
  expect(mixedContentErrors, `Mixed content errors:\n${mixedContentErrors.join('\n')}`).toHaveLength(0);
});
