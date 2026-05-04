import { expect, test } from '@playwright/test';
import { initAudit } from './helpers.js';

test('T01 Загрузка главной страницы без критических технических сбоев', async ({ page }) => {
  const audit = initAudit(page);
  const response = await page.goto('/', { waitUntil: 'domcontentloaded' });
  expect(response).toBeTruthy();
  expect(response!.status()).toBeLessThan(400);
  expect(audit.failed, `Критические failed requests:\n${audit.failed.join('\n')}`).toHaveLength(0);
  expect(audit.consoleErrors, `Критические console errors:\n${audit.consoleErrors.join('\n')}`).toHaveLength(0);
});
