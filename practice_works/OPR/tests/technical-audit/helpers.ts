import { expect, type Browser, type Page, type Request } from '@playwright/test';

export type AuditState = {
  failed: string[];
  consoleErrors: string[];
};

const ignoredFailUrl = /googletagmanager|google-analytics|yandex|vk\.com|doubleclick|mail\.ru|cdn[-.]cian\.site/i;
const ignoredFailText = /aborted|cancelled|ERR_ABORTED|NS_BINDING_ABORTED/i;
const ignoredConsoleError =
  /ResizeObserver loop limit exceeded|Failed to load resource|TypeError: Load failed|_yasc.*invalid domain|wss\.cian\.ru\/chats|connection to .* was interrupted/i;

function isCriticalResource(req: Request): boolean {
  return ['document', 'xhr', 'fetch', 'script', 'stylesheet'].includes(req.resourceType());
}

export function initAudit(page: Page): AuditState {
  const state: AuditState = { failed: [], consoleErrors: [] };

  page.on('requestfailed', (req) => {
    const url = req.url();
    const errorText = req.failure()?.errorText ?? 'failed';
    if (!isCriticalResource(req)) return;
    if (ignoredFailUrl.test(url) || ignoredFailText.test(errorText)) return;
    state.failed.push(`${req.method()} ${url} -> ${errorText}`);
  });

  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    if (ignoredConsoleError.test(text)) return;
    state.consoleErrors.push(text);
  });

  return state;
}

export async function expectStatusOk(page: Page, url: string): Promise<void> {
  const response = await page.goto(url, { waitUntil: 'domcontentloaded' });
  expect(response, `Нет ответа для ${url}`).toBeTruthy();
  expect(response!.status(), `${url} -> ${response!.status()}`).toBeLessThan(400);
}

export async function openGuestPage(browser: Browser): Promise<Page> {
  const context = await browser.newContext();
  return context.newPage();
}
