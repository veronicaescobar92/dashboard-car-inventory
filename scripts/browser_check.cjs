// Browser QA for the local application. Requires Playwright or Playwright Core.
const fs = require('node:fs');
const path = require('node:path');

const playwrightModule = process.env.PLAYWRIGHT_MODULE || 'playwright';
const { chromium } = require(playwrightModule);
const appUrl = process.env.REPUESTOS_UI || 'http://127.0.0.1:8501';
const evidenceDir = process.env.REPUESTOS_EVIDENCE_DIR || path.join('output', 'evidence');
const launchOptions = { headless: true };
if (process.env.BROWSER_EXECUTABLE) launchOptions.executablePath = process.env.BROWSER_EXECUTABLE;

(async () => {
  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(appUrl);
  await page.getByText('Una mirada al período', { exact: true }).waitFor();
  await page.waitForTimeout(2500);
  fs.mkdirSync(evidenceDir, { recursive: true });
  await page.screenshot({ path: path.join(evidenceDir, 'resumen.png'), fullPage: true });

  const runs = [];
  for (const [label, heading, file] of [
    ['Productos y compatibilidad', 'Qué se vende y qué necesita revisión', 'productos'],
    ['Evolución mensual', 'Reconoce los meses de mayor y menor venta', 'mensual'],
    ['Predicción', 'Prepara el próximo mes', 'prediccion'],
    ['Inventario', 'Inventario y reposición', 'inventario'],
  ]) {
    const start = Date.now();
    await page.getByText(label, { exact: true }).click();
    await page.getByText(heading, { exact: true }).waitFor();
    if (file === 'prediccion') {
      await page.getByRole('button', { name: 'Estimar ventas', exact: true }).click();
      await page.getByText(/Modelo: random-forest/).waitFor();
    } else {
      await page.locator('[data-testid="stDataFrame"]').first().waitFor();
    }
    await page.waitForTimeout(1800);
    await page.screenshot({ path: path.join(evidenceDir, `${file}.png`), fullPage: true });
    runs.push({ view: label, seconds: (Date.now() - start) / 1000, passed: true });
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(evidenceDir, 'mobile.png'), fullPage: true });
  fs.writeFileSync(
    path.join(evidenceDir, 'browser.json'),
    JSON.stringify({ runs, errors, method: 'Automated local browser walkthrough; no human participants' }, null, 2),
  );
  await browser.close();
  if (errors.length) throw new Error(errors.join('\n'));
  console.log(JSON.stringify(runs));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
