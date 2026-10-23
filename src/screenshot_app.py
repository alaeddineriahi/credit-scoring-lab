"""Capture a clean screenshot of the Streamlit demo for the README / LinkedIn.

Requires the app running on http://localhost:8501 and Google Chrome installed.
Usage: python src/screenshot_app.py
"""
from playwright.sync_api import sync_playwright

from config import FIG_DIR

URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5,
                            color_scheme="light")
    page.goto(URL)
    page.get_by_text("Principaux facteurs de risque").wait_for(timeout=180_000)
    page.get_by_text("Probabilité de défaut").wait_for(timeout=60_000)
    page.get_by_text("Perte attendue").wait_for(timeout=60_000)
    page.wait_for_timeout(5000)  # let the last widgets settle
    page.screenshot(path=str(FIG_DIR / "app_demo.png"), full_page=True)
    browser.close()
print("Saved", FIG_DIR / "app_demo.png")
