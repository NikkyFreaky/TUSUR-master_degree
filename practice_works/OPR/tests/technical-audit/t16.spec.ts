import { expect, test } from '@playwright/test';
import { openBuyPage } from '../helpers/cian-actions.js';
import { initAudit } from './helpers.js';

test('T16 Загрузка сторонних скриптов и виджетов', async ({ page }) => {
  const audit = initAudit(page);
  await openBuyPage(page);
  await expect(page.locator('a[href*="/sale/flat/"]').first()).toBeVisible({ timeout: 30_000 });
  expect(audit.consoleErrors, `Критические console errors:\n${audit.consoleErrors.join('\n')}`).toHaveLength(0);
});
