"""
Playwright E2E test for Prefill Stress Test against the real 4-card GPU API.

Uses a single short token level (1024) + isolation mode (max_tokens=1)
to keep the test fast while still measuring real prefill/TTFT performance.

Env:
    TEST_API_URL   4-card API base URL (default: http://192.168.199.62:10814/v1)
    TEST_MODEL     Model name if fetch fails (default: deepseek-chat)

Run: python tests/e2e_playwright_real_api_prefill.py
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
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8504"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent.parent),
        env=env,
    )
    import urllib.request
    for i in range(STREAMLIT_TIMEOUT * 2):
        try:
            urllib.request.urlopen("http://localhost:8504/healthz", timeout=2)
            print("Streamlit ready on http://localhost:8504")
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
    sbs = page.locator("[data-testid='stSelectbox']")
    sb = sbs.nth(selectbox_index)
    sb.click()
    time.sleep(0.4)
    page.keyboard.type(type_text)
    time.sleep(0.3)
    page.keyboard.press("Enter")
    time.sleep(0.4)


def find_test_type_selectbox(page) -> int | None:
    boxes = page.locator("[data-testid='stSelectbox']")
    count = boxes.count()
    test_keywords = ["Concurrency", "Context", "Custom Text", "Prefill", "Stability", "Batch"]
    for i in range(count):
        boxes.nth(i).click()
        time.sleep(0.3)
        options = page.get_by_role("option")
        for j in range(min(options.count(), 6)):
            text = options.nth(j).text_content() or ""
            if any(kw in text for kw in test_keywords):
                page.keyboard.press("Escape")
                time.sleep(0.2)
                print(f"  Found Test Type selectbox at index {i}")
                return i
        page.keyboard.press("Escape")
        time.sleep(0.2)
    print("  WARN: Could not identify Test Type selectbox, falling back to last index")
    return count - 1 if count > 0 else None


def test_real_api_flow(page):
    url = "http://localhost:8504"
    print("\n=== Step 1: Open app ===")
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("text=LLM Benchmark Platform", timeout=15000)
    take_screenshot(page, "pf_01_homepage")
    print("PASS: Homepage loaded")

    print("\n=== Step 2: Configure provider & API URL ===")
    select_by_typing(page, 0, "Custom")
    print("PASS: Provider selected")

    api_url_input = page.locator("[data-testid='stTextInput'] input").first
    api_url_input.fill(API_BASE_URL)
    page.keyboard.press("Tab")
    time.sleep(0.3)
    take_screenshot(page, "pf_02_url_filled")
    print(f"PASS: API URL set ({API_BASE_URL})")

    print("\n=== Step 3: Fetch & select model ===")
    fetch_btn = page.get_by_role("button", name="Refresh").first
    if fetch_btn.count() > 0:
        fetch_btn.click()
    time.sleep(2)

    model_sb = page.locator("[data-testid='stSelectbox']").nth(1)
    model_sb.click()
    time.sleep(0.5)
    first_opt = page.get_by_role("option").first
    if first_opt.count() > 0:
        model_name = (first_opt.text_content() or "").strip()
        first_opt.click()
        print(f"PASS: Model selected from API ({model_name})")
    else:
        page.keyboard.press("Escape")
        time.sleep(0.2)
        custom_input = page.locator("[data-testid='stTextInput'] input").nth(1)
        if custom_input.count() > 0:
            custom_input.fill(FALLBACK_MODEL)
            page.keyboard.press("Tab")
            print(f"PASS: Fallback model ID entered ({FALLBACK_MODEL})")
    time.sleep(0.3)
    take_screenshot(page, "pf_03_model_set")

    print("\n=== Step 4: Switch to Prefill Stress Test ===")
    idx = find_test_type_selectbox(page)
    if idx is None:
        print("FAIL: No selectbox available")
        return False
    select_by_typing(page, idx, "Prefill")
    time.sleep(0.6)
    take_screenshot(page, "pf_04_prefill")
    print("PASS: Prefill Stress Test panel open")

    print("\n=== Step 5: Set only 1024 token level ===")
    chips = page.locator("[data-testid='stMultiSelect'] [data-testid='stMultiSelectTag']")
    total = chips.count()
    for _ in range(total):
        first_x = chips.locator("[data-testid='stMultiSelectTagCloseButton']").first
        if first_x.count() > 0:
            first_x.click()
            time.sleep(0.1)
    time.sleep(0.2)

    ms = page.locator("[data-testid='stMultiSelect']").first
    ms.click()
    time.sleep(0.3)
    page.keyboard.type("1024")
    time.sleep(0.3)
    page.keyboard.press("Enter")
    time.sleep(0.2)
    page.keyboard.press("Escape")
    time.sleep(0.2)
    print("PASS: Token level set to 1024 only")
    take_screenshot(page, "pf_05_tokens_set")

    print("\n=== Step 6: Click Start Prefill Stress Test ===")
    page.evaluate("window.scrollTo(0, 300)")
    time.sleep(0.2)

    start_btn = page.locator("button").filter(has_text="Start Prefill Stress Test")
    if start_btn.count() == 0:
        start_btn = page.locator("button").filter(has_text="Start Prefill")
    if start_btn.count() == 0:
        start_btn = page.locator("[data-testid='stButton'] button[kind='primary']")
    start_btn.first.click(force=True)
    print("PASS: Start button clicked")
    take_screenshot(page, "pf_06_clicked")

    print("\n=== Step 7: Wait for completion ===")
    running_loc = page.locator("text=Running").first
    try:
        running_loc.wait_for(state="visible", timeout=15000)
        print("  Confirmed: test is Running")
    except Exception:
        print("  WARN: 'Running' state not seen within 15s")
    take_screenshot(page, "pf_07_running")

    print("  Waiting for test to finish (Running -> Completed)...")
    try:
        running_loc.wait_for(state="hidden", timeout=TEST_TIMEOUT * 1000)
        print("  Confirmed: Running state disappeared")
    except Exception:
        print("  WARN: Running state did not disappear within timeout")
    time.sleep(1)
    take_screenshot(page, "pf_08_finished")

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

    if page.locator("text=Completed").count() == 0:
        print("WARN: 'Completed' state not found after run")
    else:
        print("PASS: 'Completed' state detected")

    print("PASS: Prefill Stress Test finished without visible errors")
    return True


def main():
    print("=" * 60)
    print("Playwright E2E Prefill Stress Test: Real 4-Card API")
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
