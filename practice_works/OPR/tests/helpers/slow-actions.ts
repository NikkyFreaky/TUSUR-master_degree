import { Page } from '@playwright/test';

export async function humanPause(ms = 700): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

export async function safeClick(page: Page, selector: string, pauseMs = 700): Promise<void> {
  await page.locator(selector).first().click({ timeout: 20_000 });
  await humanPause(pauseMs);
}
