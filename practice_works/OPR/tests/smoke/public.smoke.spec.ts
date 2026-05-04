import { expect, test } from '@playwright/test';

test.describe('Smoke тестирование', () => {
  test('главная и каталог открываются без критических сбоев', async ({ page }) => {
    const failedRequests: string[] = [];
    const severeConsole: string[] = [];

    const ignoredFailUrl = /googletagmanager|google-analytics|yandex|vk\.com|doubleclick|top-fwz1\.mail\.ru|cdn[-\.]cian\.site/i;
    const ignoredFailText = /aborted|cancelled|ERR_ABORTED|NS_BINDING_ABORTED/i;

    page.on('requestfailed', (req) => {
      const url = req.url();
      const errorText = req.failure()?.errorText ?? 'failed';
      const isCriticalResource = ['document', 'xhr', 'fetch', 'script'].includes(req.resourceType());
      if (!isCriticalResource) {
        return;
      }
      if (ignoredFailUrl.test(url) || ignoredFailText.test(errorText)) {
        return;
      }
      failedRequests.push(`${req.method()} ${url} -> ${errorText}`);
    });

    page.on('console', (msg) => {
      if (msg.type() !== 'error') return;
      const text = msg.text();
      if (/ResizeObserver loop limit exceeded|Failed to load resource|TypeError: Load failed|_yasc.*invalid domain|wss\.cian\.ru\/chats|connection to .* was interrupted/i.test(text)) return;
      severeConsole.push(text);
    });

    const home = await page.goto('/');
    expect(home?.status()).toBeGreaterThanOrEqual(200);
    expect(home?.status()).toBeLessThan(400);

    await expect(page).toHaveURL(/cian\.ru/);

    const catalog = await page.goto('/kupit/');
    expect(catalog?.status()).toBeGreaterThanOrEqual(200);
    expect(catalog?.status()).toBeLessThan(400);

    await expect(page).toHaveURL(/kupit/);

    expect(failedRequests, `Failed requests:\n${failedRequests.join('\n')}`).toHaveLength(0);
    expect(severeConsole, `Console errors:\n${severeConsole.join('\n')}`).toHaveLength(0);
  });
});
