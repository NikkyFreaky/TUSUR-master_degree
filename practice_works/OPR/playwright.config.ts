import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.BASE_URL ?? 'https://www.cian.ru';
const headed = process.env.HEADED === '1';
const videoMode = process.env.VIDEO_MODE ?? 'retain-on-failure';
const authStatePath = 'playwright/.auth/user.json';

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 1,
  workers: 1,
  reporter: [['html', { open: 'never' }], ['list']],
  use: {
    baseURL,
    headless: !headed,
    trace: 'on-first-retry',
    video: videoMode as 'off' | 'on' | 'retain-on-failure' | 'on-first-retry',
    screenshot: 'only-on-failure',
    actionTimeout: 20_000,
    navigationTimeout: 45_000
  },
  projects: [
    {
      name: 'setup',
      testMatch: /.*auth\.setup\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        headless: false
      }
    },
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        storageState: authStatePath
      }
    },
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        storageState: authStatePath
      }
    },
    {
      name: 'webkit',
      use: {
        ...devices['Desktop Safari'],
        storageState: authStatePath
      }
    }
  ]
});
