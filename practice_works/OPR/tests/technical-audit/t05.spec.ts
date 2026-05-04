import { expect, test } from '@playwright/test';
import { openAnyListingCard, openBuyPage, searchByStreet } from '../helpers/cian-actions.js';
import { initAudit } from './helpers.js';

test('T05 Сетевые запросы и API на ключевых сценариях', async ({ page }) => {
  const audit = initAudit(page);
  await openBuyPage(page);
  await searchByStreet(page, 'Ленина');
  await openAnyListingCard(page);
  expect(audit.failed, `Критические failed requests:\n${audit.failed.join('\n')}`).toHaveLength(0);
});
