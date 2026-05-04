import { expect, test } from '@playwright/test';
import { openBuyPage } from '../helpers/cian-actions.js';
import { initAudit } from './helpers.js';

test('T03 Загрузка основных ресурсов страницы', async ({ page }) => {
  const audit = initAudit(page);
  await openBuyPage(page);
  await expect(page.locator('a[href*="/sale/flat/"]').first()).toBeVisible({ timeout: 30_000 });
  expect(audit.failed, `Критические failed requests:\n${audit.failed.join('\n')}`).toHaveLength(0);
});
