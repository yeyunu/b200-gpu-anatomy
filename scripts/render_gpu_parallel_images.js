const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const pagePath = path.join(root, 'gpu-sm-warp-memory-explainer.html');

const shots = [
  ['mapping', 'gpu-parallel-01-mapping.png'],
  ['registers', 'gpu-parallel-02-registers.png'],
  ['shared', 'gpu-parallel-03-shared-storage.png'],
  ['barrier', 'gpu-parallel-04-barrier.png'],
  ['read-pairs', 'gpu-parallel-05-shared-read.png'],
  ['alu32', 'gpu-parallel-06-alu-parallel.png'],
  ['write32', 'gpu-parallel-07-shared-writeback.png'],
  ['rounds', 'gpu-parallel-08-rounds.png'],
  ['final', 'gpu-parallel-09-final-result.png'],
  ['storage', 'gpu-parallel-10-storage-map.png'],
  ['timeline', 'gpu-parallel-11-warp-timeline.png'],
  ['overview', 'gpu-parallel-12-overview.png'],
];

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({
    viewport: { width: 800, height: 1400 },
    deviceScaleFactor: 2,
    colorScheme: 'light',
  });
  const page = await context.newPage();
  await page.goto(pathToFileURL(pagePath).href, { waitUntil: 'load' });

  const iframe = page.locator('iframe');
  const frameHandle = await iframe.elementHandle();
  const frame = await frameHandle.contentFrame();
  await frame.waitForSelector('[data-global-step]');
  const visual = frame.locator('#gpu-parallel-global-zoom');

  for (const [step, filename] of shots) {
    await frame.locator(`[data-global-step="${step}"]`).click();
    await page.waitForTimeout(80);
    await visual.screenshot({ path: path.join(root, filename) });
    console.log(`Rendered ${filename}`);
  }

  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
