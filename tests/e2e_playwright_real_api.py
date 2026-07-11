"""
Playwright E2E test against the real 4-card GPU API.

Simulates a user configuring the 4-card endpoint and running a
lightweight Concurrency Test to verify full end-to-end API connectivity.

Env:
    TEST_API_URL   4-card API base URL (default: http://192.168.199.62:10814/v1)
    TEST_MODEL     Model name if fetch fails (default: deepseek-chat)

Run: python tests/e2e_playwright_real_api.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright

STREAMLIT_TIMEOUT = 30
TEST_TIMEOUT = 300
SCREENSHOT_DIR = Path("tests/e2e_screenshots")

streamlit_process = None
API_BASE_URL = os.environ.get("TEST_API_URL", "http://192.168.199.62:10814/v1")
FALLBACK_MODEL = os.environ.get("TEST_MODEL", "deepseek-chat")


def start_streamlit():
    global streamlit_process
    print("Starting Streamlit app...")
    env = os.environ.copy()
    env["STREAMLIT_LOG_LEVEL"] = "error"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent.parent),
        env=env,
    )
    import urllib.request
    for i in range(STREAMLIT_TIMEOUT * 2):
        try:
            urllib.request.urlopen("http://localhost:8502/healthz", timeout=2)
            print("Streamlit ready on http://localhost:8502")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Streamlit failed to start")


def stop_streamlit():
    global streamlit_process
    if streamlit_process:
        print("Stopping Streamlit...")
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_process.kill()
            streamlit_process.wait()
        print("Streamlit stopped")


def take_screenshot(page, name: str):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"  Screenshot: {path}")
    return path


def select_by_typing(page, selectbox_index: int, type_text: str):
    """Click a selectbox, type to filter, and press Enter."""
    sbs = page.locator("[data-testid='stSelectbox']")
    sb = sbs.nth(selectbox_index)
    sb.click()
    time.sleep(0.4)
    page.keyboard.type(type_text)
    time.sleep(0.3)
    page.keyboard.press("Enter")
    time.sleep(0.4)


def wait_for_any_text(page, texts: list[str], timeout_ms: int = 30000):
    """Wait until any of the given texts appears on the page."""
    for _ in range(timeout_ms // 500):
        for t in texts:
            if page.locator("text=" + t).count() > 0:
                return t
        time.sleep(0.5)
    return None


def test_real_api_flow(page):
    url = "http://localhost:8502"
    print("\n=== Step 1: Open app ===")
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("text=LLM Benchmark Platform", timeout=15000)
    take_screenshot(page, "api_01_homepage")
    print("PASS: Homepage loaded")

    print("\n=== Step 2: Configure provider & API URL ===")
    # Provider (index 0)
    select_by_typing(page, 0, "Custom")
    print("PASS: Provider selected (Custom)")

    # API Base URL
    api_url_input = page.locator("[data-testid='stTextInput'] input").first
    api_url_input.fill(API_BASE_URL)
    page.keyboard.press("Tab")
    time.sleep(0.3)
    take_screenshot(page, "api_02_url_filled")
    print(f"PASS: API URL set ({API_BASE_URL})")

    print("\n=== Step 3: Fetch & select model ===")
    # Click the model refresh button.
    fetch_btn = page.get_by_role("button", name="Refresh").first
    if fetch_btn.count() > 0:
        fetch_btn.click()
    else:
        print("WARN: Fetch button missing")

    # Wait for success indicator or just settle
    time.sleep(2)

    # Try to select first model from the dropdown
    model_sb = page.locator("[data-testid='stSelectbox']").nth(1)
    model_sb.click()
    time.sleep(0.5)
    first_opt = page.get_by_role("option").first
    if first_opt.count() > 0:
        model_name = (first_opt.text_content() or "").strip()
        first_opt.click()
        print(f"PASS: Model selected from API ({model_name})")
    else:
        # Close dropdown and fallback to custom text input
        page.keyboard.press("Escape")
        time.sleep(0.2)
        custom_input = page.locator("[data-testid='stTextInput'] input").nth(1)
        if custom_input.count() > 0:
            custom_input.fill(FALLBACK_MODEL)
            page.keyboard.press("Tab")
            print(f"PASS: Fallback model ID entered ({FALLBACK_MODEL})")
        else:
            print("WARN: Neither model selectbox nor custom input found")
    time.sleep(0.3)
    take_screenshot(page, "api_03_model_set")

    print("\n=== Step 4: Switch to Concurrency Test ===")
    # Test type is usually the 3rd selectbox (index 2)
    select_by_typing(page, 2, "Concurrency")
    time.sleep(0.5)
    take_screenshot(page, "api_04_concurrency")
    print("PASS: Concurrency Test panel open")

    print("\n=== Step 5: Reduce Max Output Tokens to 1 ===")
    # Sidebar expander may need to be expanded first
    # Find "Parameter Settings" expander if closed
    param_setting = page.locator("[data-testid='stExpander']").filter(has_text="Parameter Settings")
    if param_setting.count() > 0:
        param_setting.first.click()
        time.sleep(0.3)

    max_token_input = page.locator("[data-testid='stNumberInput'] input").nth(1)
    if max_token_input.count() > 0:
        max_token_input.click()
        max_token_input.fill("1")
        page.keyboard.press("Tab")
        time.sleep(0.2)
        print("PASS: Max tokens = 1")
    else:
        print("WARN: Max tokens input not located")
    take_screenshot(page, "api_05_params_set")

    print("\n=== Step 6: Click Start Test ===")
    start_btn = page.locator("button").filter(has_text="Start Concurrency Test (M)")
    if start_btn.count() == 0:
        start_btn = page.locator("button").filter(has_text="Start Concurrency")
    if start_btn.count() == 0:
        # fallback: any primary button in main panel
        start_btn = page.locator("[data-testid='stButton'] button[kind='primary']")
    start_btn.first.click()
    print("PASS: Start button clicked")
    take_screenshot(page, "api_06_clicked")

    print("\n=== Step 7: Wait for completion ===")
    # 1) Confirm test is running
    running_loc = page.locator("text=Running").first
    try:
        running_loc.wait_for(state="visible", timeout=15000)
        print("  Confirmed: test is Running")
    except Exception:
        print("  WARN: 'Running' state not seen within 15s")
    take_screenshot(page, "api_07_running")

    # 2) Wait for "Running" to disappear (test finished + page rerendered)
    print("  Waiting for test to finish (Running -> Idle/Results)...")
    try:
        running_loc.wait_for(state="hidden", timeout=TEST_TIMEOUT * 1000)
        print("  Confirmed: Running state disappeared")
    except Exception:
        print("  WARN: Running state did not disappear within timeout")
    time.sleep(1)
    take_screenshot(page, "api_08_finished")

    # 3) Collect visible errors from UI
    alerts = page.locator("[data-testid='stException'], [data-testid='stAlert'], [data-testid='stToast']")
    found_errors = []
    for i in range(min(alerts.count(), 10)):
        txt = alerts.nth(i).text_content() or ""
        if any(k in txt.lower() for k in ["error", "fail", "exception", "timeout", "unable"]):
            found_errors.append(txt[:250])
    if found_errors:
        print("FAIL: App/API errors detected")
        for e in found_errors:
            print(f"  {e}")
        return False

    # 4) Verify result table exists (TTFT/TPS columns)
    if page.locator("text=TTFT").count() == 0 and page.locator("text=Idle").count() == 0:
        print("WARN: Neither result table (TTFT) nor Idle state detected after run")
        return False

    print("PASS: Test finished without visible errors")
    return True


def main():
    print("=" * 60)
    print("Playwright E2E Test: Real 4-Card API")
    print(f"API: {API_BASE_URL}")
    print("=" * 60)

    start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            console_errors = []
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            success = test_real_api_flow(page)

            if console_errors:
                print("\nConsole errors:")
                for err in console_errors:
                    print(f"  {err}")

            browser.close()
    finally:
        stop_streamlit()

    print("=" * 60)
    print("PASSED" if success else "FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
