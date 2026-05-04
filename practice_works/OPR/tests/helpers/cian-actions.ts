import { expect, type Page } from '@playwright/test';
import { humanPause } from './slow-actions.js';

export async function openBuyPage(page: Page): Promise<void> {
  await page.goto('/kupit/');
  await expect(page).toHaveURL(/cian\.ru\/kupit|cat\.php/i, { timeout: 30_000 });
  await humanPause(800);
}

export async function searchByStreet(page: Page, query: string): Promise<void> {
  const searchInput = page
    .getByRole('combobox', { name: /Регион|Адрес|Поиск|Локация/i })
    .or(page.getByPlaceholder(/Город|Адрес|Район|Населенный пункт|Адрес/i))
    .first();

  await expect(searchInput).toBeVisible({ timeout: 30_000 });
  await searchInput.click({ timeout: 20_000 });
  await searchInput.fill(query);
  await humanPause(400);
  await page.keyboard.press('Enter');

  const findButton = page.getByRole('link', { name: /Найти/i }).first();
  if (await findButton.isVisible().catch(() => false)) {
    await findButton.click();
  }

  await expect(page).toHaveURL(/kupit|cat\.php|deal_type=sale/i, { timeout: 30_000 });
}

export async function applyBaseFilters(page: Page): Promise<void> {
  const roomsFilter = page
    .getByRole('button', { name: /Комнат|Комнатность/i })
    .or(page.getByText(/Комнатность|Комнат/i).first())
    .first();
  await expect(roomsFilter).toBeVisible({ timeout: 20_000 });
  await roomsFilter.click({ timeout: 20_000 });
  await humanPause(600);

  let oneRoomOption = page.getByRole('button', { name: /^1$/ }).first();
  if (!(await oneRoomOption.isVisible().catch(() => false))) {
    oneRoomOption = page.getByRole('checkbox', { name: /^1$/ }).first();
  }
  if (!(await oneRoomOption.isVisible().catch(() => false))) {
    oneRoomOption = page.getByRole('radio', { name: /^1$/ }).first();
  }
  if (!(await oneRoomOption.isVisible().catch(() => false))) {
    oneRoomOption = page.getByRole('button', { name: /1-комн/i }).first();
  }
  if (!(await oneRoomOption.isVisible().catch(() => false))) {
    await roomsFilter.click({ timeout: 20_000 });
    await humanPause(500);
  }
  await expect(oneRoomOption).toBeVisible({ timeout: 20_000 });
  await oneRoomOption.click({ timeout: 20_000 });
  await humanPause(500);

  const priceFilter = page.getByRole('button', { name: /Цена/i }).first();
  await expect(priceFilter).toBeVisible({ timeout: 20_000 });
  await priceFilter.click({ timeout: 20_000 });
  await humanPause(500);

  const minPrice = page.getByPlaceholder(/от/i).first();
  const maxPrice = page.getByPlaceholder(/до/i).first();
  if (await minPrice.isVisible().catch(() => false)) {
    await minPrice.fill('1000000');
  }
  if (await maxPrice.isVisible().catch(() => false)) {
    await maxPrice.fill('7000000');
  }

  const applyPrice = page.getByRole('button', { name: /Показать|Применить|Готово/i }).first();
  if (await applyPrice.isVisible().catch(() => false)) {
    await applyPrice.click();
  } else {
    await page.keyboard.press('Enter');
  }

  await humanPause(1000);
}

export async function openAnyListingCard(page: Page): Promise<string> {
  const firstCardLink = page
    .locator('a[href*="/sale/flat/"]')
    .filter({ hasText: /кв\.|м²|этаж/i })
    .first();

  await expect(firstCardLink).toBeVisible({ timeout: 30_000 });
  const cardUrl = await firstCardLink.getAttribute('href');
  expect(cardUrl).toBeTruthy();
  await page.goto(cardUrl!, { waitUntil: 'domcontentloaded' });
  await expect(page).toHaveURL(/\/sale\/flat\//, { timeout: 30_000 });

  return cardUrl!;
}

export async function hasAuthorizedSession(page: Page): Promise<boolean> {
  await page.goto('https://www.cian.ru/', { waitUntil: 'domcontentloaded' });

  const currentUrl = page.url();
  if (/login|auth|signup|captcha|blocked/i.test(currentUrl)) {
    return false;
  }

  const profileLink = page.locator('a[href*="my.cian.ru"]:visible').first();
  if (await profileLink.isVisible().catch(() => false)) {
    return true;
  }

  const loginEntry = page
    .getByRole('link', { name: /Войти|Вход/i })
    .or(page.getByRole('button', { name: /Войти|Вход/i }))
    .first();

  const hasLoginEntry = await loginEntry.isVisible().catch(() => false);
  return !hasLoginEntry;
}
