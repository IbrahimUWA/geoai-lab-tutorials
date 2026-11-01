"""Screenshot the web app with Playwright (uses the installed Edge), one image per site plus a click readout."""
import asyncio
import os
import sys

from playwright.async_api import async_playwright

OUT = r"E:\GeoAI-Lab-Tutorials\ep05-web-app\results"
URL = "http://127.0.0.1:8765/index.html"


async def main():
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        for site in range(3):
            await page.goto(f"{URL}?site={site}", wait_until="networkidle")
            await page.wait_for_timeout(4000)
            await page.screenshot(path=os.path.join(OUT, f"app_site{site}.png"))
            # click near the centre of the map to show the readout / popup
            await page.mouse.click(1130, 540)
            await page.wait_for_timeout(1500)
            await page.screenshot(path=os.path.join(OUT, f"app_site{site}_click.png"))
            readout = await page.inner_text("#readout")
            print(f"site {site}: readout -> {readout!r}")
        await browser.close()
        print("console errors:", errors[:5] if errors else "none")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
