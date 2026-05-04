import { expect, test } from '@playwright/test';
import { expectStatusOk, initAudit } from './helpers.js';

test('T06 Инициализация и загрузка интерактивной карты', async ({ page }) => {
  const audit = initAudit(page);
  await expectStatusOk(page, '/map/');
  const mapCandidate = page.locator('[class*="map" i], [id*="map" i], canvas, ymaps').first();
  await expect(mapCandidate).toBeVisible({ timeout: 30_000 });
  await expect(page).toHaveURL(/map/i);
  expect(audit.failed, `Критические failed requests:\n${audit.failed.join('\n')}`).toHaveLength(0);
});
