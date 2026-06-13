"""Playwright E2E tests for the RLM demo page."""

import time
from playwright.sync_api import sync_playwright, expect

BASE = "http://localhost:3000/demo"


def test_page_loads():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(BASE)
        
        # Header should be visible
        expect(page.locator("h1")).to_contain_text("Recursive Language Models")
        
        # Buttons should exist (wait for page to fully load)
        page.wait_for_selector("#btnRLM", timeout=10000)
        page.wait_for_selector("#btnBaseline", timeout=10000)
        
        # OOLONG button should exist
        expect(page.locator("button", has_text="Load OOLONG")).to_be_visible()
        
        browser.close()


def test_load_oolong_data():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(BASE)
        
        # Click "Load OOLONG Benchmark Data"
        page.locator("button", has_text="Load OOLONG").click()
        
        # Wait for data to load (status should update)
        page.wait_for_selector(".status-bar:not(.running)", timeout=15000)
        
        # Status should mention OOLONG
        status = page.locator("#status")
        status_text = status.text_content()
        assert "OOLONG" in status_text, f"Status doesn't mention OOLONG: {status_text}"
        assert "787" in status_text, f"Expected 787 questions: {status_text}"
        
        # Ground truth should be visible
        truth = page.locator("#groundTruth")
        truth_text = truth.text_content()
        assert "abbreviation" in truth_text, f"Ground truth missing: {truth_text}"
        
        # Textarea should be populated
        context = page.locator("#context")
        context_val = context.input_value()
        assert len(context_val) > 10000, f"Context too small: {len(context_val)} chars"
        
        browser.close()
        print("✅ test_load_oolong_data passed")


def test_algorithm_2_baseline():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(BASE)
        
        # Load OOLONG first
        page.locator("button", has_text="Load OOLONG").click()
        page.wait_for_selector(".status-bar:not(.running)", timeout=15000)
        
        # Click "Run Algorithm 2"
        page.locator("#btnBaseline").click()
        
        # Wait for result
        page.wait_for_selector(".status-bar:not(.running)", timeout=30000)
        
        # Check result card
        alg2_card = page.locator("#alg2Card")
        alg2_text = alg2_card.text_content()
        print(f"  Alg2 result: {alg2_text[:200]}")
        
        # Algorithm 2 should either be empty or have "length" finish reason
        # On OOLONG data (76K chars), it should choke
        assert "empty" in alg2_text.lower() or "length" in alg2_text.lower(), \
            f"Algorithm 2 didn't show expected failure: {alg2_text[:200]}"
        
        browser.close()
        print("✅ test_algorithm_2_baseline passed")


def test_rlm_run_small():
    """Test RLM with a small context (2+2)."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(BASE)
        
        # Type a small context
        page.locator("#context").fill("What is 2+2? Answer in one word.")
        
        # Click Run RLM
        page.locator("#btnRLM").click()
        
        # Wait for result (should be fast: ~8s)
        page.wait_for_selector(".status-bar:not(.running)", timeout=60000)
        
        # Check RLM result card
        rlm_card = page.locator("#rlmCard")
        rlm_text = rlm_card.text_content()
        print(f"  RLM result: {rlm_text[:200]}")
        
        assert "Four" in rlm_text or "four" in rlm_text.lower(), \
            f"RLM didn't return 'Four': {rlm_text[:200]}"
        
        # Take screenshot
        page.screenshot(path="tests/screenshots/rlm_small_pass.png")
        
        browser.close()
        print("✅ test_rlm_run_small passed")


def test_rlm_on_oolong():
    """Full E2E: OOLONG data + RLM. This may take 30-120s."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(BASE)
        
        # Load OOLONG
        page.locator("button", has_text="Load OOLONG").click()
        page.wait_for_selector(".status-bar:not(.running)", timeout=15000)
        
        # Run RLM (with 4 iterations to keep it fast)
        # We inject max_iterations into the request indirectly:
        # modify the context to be shorter for a faster test
        page.evaluate("""
            const orig = document.getElementById('context').value;
            const lines = orig.split('\\n');
            const short = lines.slice(0, 100).join('\\n');
            document.getElementById('context').value = short;
        """)
        
        page.locator("#btnRLM").click()
        
        # Wait for result (max 180s)
        try:
            page.wait_for_selector(".status-bar:not(.running)", timeout=180000)
        except Exception:
            # Take screenshot on timeout
            page.screenshot(path="tests/screenshots/rlm_oolong_timeout.png")
            raise
        
        rlm_card = page.locator("#rlmCard")
        rlm_text = rlm_card.text_content()
        print(f"  RLM on OOLONG: {rlm_text[:300]}")
        
        # Should have some result (not empty, not error)
        assert "Error" not in rlm_text, f"RLM errored: {rlm_text[:200]}"
        assert "TIMEOUT" not in rlm_text, f"RLM timed out: {rlm_text[:200]}"
        
        page.screenshot(path="tests/screenshots/rlm_oolong_pass.png")
        browser.close()
        print("✅ test_rlm_on_oolong passed")


def test_visualizer_loads():
    """Verify the visualizer page loads and shows traces."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:3000/")
        
        # Should show the RLM Visualizer header
        expect(page.locator("h1")).to_contain_text("RLM")
        
        # Should have upload area (text may vary after redesign)
        expect(page.locator("text=Upload trace").first).to_be_visible()
        
        page.screenshot(path="tests/screenshots/visualizer.png")
        browser.close()
        print("✅ test_visualizer_loads passed")


if __name__ == "__main__":
    import os
    os.makedirs("tests/screenshots", exist_ok=True)
    
    print("=" * 50)
    print("RLM Demo — Playwright E2E Tests")
    print("=" * 50)
    
    tests = [
        ("Page loads", test_page_loads),
        ("OOLONG data load", test_load_oolong_data),
        ("Algorithm 2 baseline", test_algorithm_2_baseline),
        ("RLM small context", test_rlm_run_small),
        ("Visualizer", test_visualizer_loads),
        ("RLM on OOLONG", test_rlm_on_oolong),
    ]
    
    passed = 0
    failed = 0
    
    for name, fn in tests:
        print(f"\n── {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
