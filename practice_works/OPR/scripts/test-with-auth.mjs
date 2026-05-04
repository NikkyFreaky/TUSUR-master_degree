import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { chromium } from 'playwright';

const authStatePath = 'playwright/.auth/user.json';

function run(command, args) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    shell: true
  });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

async function hasLiveSession() {
  if (!existsSync(authStatePath)) {
    return false;
  }

  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({ storageState: authStatePath });
    const page = await context.newPage();
    await page.goto('https://www.cian.ru/', { waitUntil: 'domcontentloaded', timeout: 45_000 });

    const currentUrl = page.url();
    if (/captcha|blocked|tmgrdfrend/i.test(currentUrl)) {
      return true;
    }

    const profileLink = page.locator('a[href*="my.cian.ru"]:visible').first();
    if (await profileLink.isVisible().catch(() => false)) {
      return true;
    }

    const loginEntry = page
      .getByRole('link', { name: /Войти|Вход/i })
      .or(page.getByRole('button', { name: /Войти|Вход/i }))
      .first();

    return !(await loginEntry.isVisible().catch(() => false));
  } catch {
    return false;
  } finally {
    await browser.close();
  }
}

const forwardedArgs = process.argv.slice(2);
const sessionAlive = await hasLiveSession();
const hasProjectArg = forwardedArgs.some((arg) => arg === '--project' || arg.startsWith('--project='));

const defaultProjects = hasProjectArg
  ? []
  : ['--project=chromium', '--project=firefox', '--project=webkit'];

if (!sessionAlive) {
  console.log('Сессия невалидна или отсутствует. Запускаю setup авторизации...');
  run('npx', ['playwright', 'test', '--project=setup']);
} else {
  console.log('Сессия активна. Запускаю тесты без повторной авторизации.');
}

run('npx', ['playwright', 'test', ...defaultProjects, ...forwardedArgs]);
