import { expect, test } from '@playwright/test';
import { openAnyListingCard, openBuyPage } from '../helpers/cian-actions.js';
import { initAudit } from './helpers.js';

test('T04 Ошибки в консоли браузера', async ({ page }) => {
  const audit = initAudit(page);
  await openBuyPage(page);
  await openAnyListingCard(page);
  expect(audit.consoleErrors, `Критические console errors:\n${audit.consoleErrors.join('\n')}`).toHaveLength(0);
});
