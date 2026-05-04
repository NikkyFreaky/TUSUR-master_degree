import { test } from '@playwright/test';
import { expectStatusOk } from './helpers.js';

test('T02 Загрузка ключевых страниц и разделов', async ({ page }) => {
  await expectStatusOk(page, '/');
  await expectStatusOk(page, '/kupit/');
  await expectStatusOk(page, '/ipoteka-main/');
  await expectStatusOk(page, '/map/');
});
