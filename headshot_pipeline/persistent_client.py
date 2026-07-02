"""
Persistent Gemini Client — 浏览器常驻模式
==========================================

架构：
  1. Chrome 以 --remote-debugging-port=9222 常驻运行（手动或脚本启动）
  2. 脚本通过 CDP 连接到已打开的 Chrome，发送 prompt，取回图片
  3. 脚本退出后 Chrome 继续开着，session/cookies 全部保留
  4. 下次再跑脚本，直接连上去继续用

用法：
  # 第一步：启动 Chrome（只需一次）
  python persistent_client.py launch

  # 之后随时生成图片（浏览器保持开着）
  python persistent_client.py generate "A professional headshot of ..."
  python persistent_client.py generate --id bf_m_01
  python persistent_client.py batch --style business_formal
  python persistent_client.py batch --all
"""

import json
import time
import base64
import uuid
import argparse
import subprocess
import signal
import sys
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException
)

from watermark_remover import remove_gemini_watermark

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CDP_PORT = 9222
GEMINI_URL = "https://gemini.google.com/app"
CHROME_DEBUG_FLAGS = [
    "--remote-debugging-port={port}",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--user-data-dir={user_data_dir}",
]

# Where to find prompts.json
HERE = Path(__file__).resolve().parent


# ──────────────────────────────────────────────
# Chrome Launcher
# ──────────────────────────────────────────────
def find_chrome_binary():
    """Find Chrome or Chromium on macOS."""
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    # fallback: try which
    import shutil
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        return chrome
    raise FileNotFoundError(
        "Chrome not found. Install Google Chrome or set the path manually."
    )


def launch_chrome(port=CDP_PORT, user_data_dir=None):
    """Launch Chrome with remote debugging enabled. Returns process."""
    if user_data_dir is None:
        user_data_dir = str(HERE / ".chrome_profile")
    Path(user_data_dir).mkdir(parents=True, exist_ok=True)

    chrome_bin = find_chrome_binary()
    flags = [
        f.replace("{port}", str(port)).replace("{user_data_dir}", user_data_dir)
        for f in CHROME_DEBUG_FLAGS
    ]

    cmd = [chrome_bin] + flags
    print(f"🚀 Launching Chrome with CDP on port {port}...")
    print(f"   Binary:  {chrome_bin}")
    print(f"   Profile: {user_data_dir}")
    print(f"   (Close the terminal or Ctrl-C to stop Chrome)")

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    if proc.poll() is not None:
        raise RuntimeError(f"Chrome exited immediately with code {proc.returncode}")

    print(f"✓ Chrome running (PID {proc.pid})")
    print(f"  Open gemini.google.com and log in. Session will persist.")
    return proc


def check_chrome_running(port=CDP_PORT):
    """Check if Chrome with CDP is already running."""
    try:
        from selenium import webdriver
        opts = Options()
        opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
        driver = webdriver.Chrome(options=opts)
        driver.quit()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────
# Persistent Client
# ──────────────────────────────────────────────
class PersistentGeminiClient:
    """Connects to an already-running Chrome via CDP.

    Browser stays open after .close() — only the WebDriver session disconnects.
    """

    def __init__(self, port=CDP_PORT, output_dir="output", wait_timeout=60):
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wait_timeout = wait_timeout
        self.driver = None
        # Multi-turn conversation tracking
        self._in_conversation = False
        self._known_image_srcs: set[str] = set()  # blob URLs seen so far
        # Uploaded image fingerprints: list of (width, height) to detect re-rendered uploads
        self._uploaded_fingerprints: list[tuple[int, int]] = []

    def connect(self):
        """Connect to running Chrome via CDP."""
        opts = Options()
        opts.add_experimental_option("debuggerAddress", f"127.0.0.1:{self.port}")
        # NOTE: When connecting via CDP, we cannot use excludeSwitches,
        # useAutomationExtension, or disable-blink-features — those only
        # work when launching a new browser. The real Chrome session has
        # no automation flags anyway, which is exactly what we want.

        try:
            self.driver = webdriver.Chrome(options=opts)
        except WebDriverException as e:
            raise ConnectionError(
                f"Cannot connect to Chrome on port {self.port}. "
                f"Run 'python persistent_client.py launch' first.\n"
                f"Original error: {e}"
            )
        print(f"✓ Connected to Chrome (CDP port {self.port})")
        return self.driver

    def ensure_gemini_page(self):
        """Make sure we're on the Gemini page."""
        if not self.driver:
            self.connect()

        current_url = self.driver.current_url or ""
        if "gemini.google.com" not in current_url:
            print("Navigating to Gemini...")
            self.driver.get(GEMINI_URL)
            time.sleep(3)

        # Check if we need to log in
        if self._is_login_required():
            print("\n" + "=" * 50)
            print("⚠  Please log in to Gemini in the Chrome window.")
            print("   Waiting up to 120 seconds for you to log in...")
            print("=" * 50)
            if not self._wait_for_login(timeout=120):
                raise RuntimeError("Login timed out. Please log in and try again.")
            print("✓ Login detected!")

    def _is_login_required(self):
        """Check if the login page is shown."""
        login_indicators = [
            "//button[contains(text(), 'Sign in')]",
            "//a[contains(text(), 'Sign in')]",
        ]
        for sel in login_indicators:
            try:
                el = self.driver.find_element(By.XPATH, sel)
                if el.is_displayed():
                    return True
            except NoSuchElementException:
                continue
        return False

    def _wait_for_login(self, timeout=120):
        """Wait for the user to manually log in."""
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(3)
            try:
                # Look for chat input = logged in
                self._find_text_input()
                return True
            except NoSuchElementException:
                continue
        return False

    def _find_text_input(self):
        """Find Gemini's text input field."""
        selectors = [
            # Gemini 2026 uses a div with this aria-label
            (By.CSS_SELECTOR, "div[aria-label='Enter a prompt for Gemini']"),
            (By.CSS_SELECTOR, "div[aria-label*='Enter a prompt']"),
            (By.CSS_SELECTOR, "textarea[aria-label*='Enter a prompt']"),
            (By.CSS_SELECTOR, "textarea[placeholder*='Enter a prompt']"),
            (By.CSS_SELECTOR, "textarea"),
            (By.CSS_SELECTOR, "div[contenteditable='true']"),
            (By.CSS_SELECTOR, "[role='textbox']"),
        ]
        for by, sel in selectors:
            try:
                el = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((by, sel))
                )
                if el.is_displayed():
                    return el
            except (TimeoutException, NoSuchElementException):
                continue
        raise NoSuchElementException("Cannot find text input. Are you on Gemini?")

    def generate(self, prompt, title=None, photo_path=None):
        """Generate a single image (one-shot, opens new chat each time).

        Args:
            prompt: Text prompt
            title: Filename prefix (optional)
            photo_path: Reference photo to upload (optional)

        Returns:
            Path to saved image file
        """
        self.ensure_gemini_page()

        # Always start a fresh chat for each generation
        print("  Opening new chat...")
        self._new_chat()
        time.sleep(2)

        # Reset conversation tracking
        self._in_conversation = False
        self._known_image_srcs.clear()

        # Upload reference photo if provided
        if photo_path and Path(photo_path).exists():
            self._upload_photo(photo_path)
            time.sleep(1)

        # Snapshot existing images before generation
        self._snapshot_images()

        # Enter prompt
        text_input = self._find_text_input()
        self.driver.execute_script("arguments[0].focus();", text_input)
        text_input.click()
        time.sleep(0.3)
        text_input.clear()
        time.sleep(0.2)
        text_input.send_keys(prompt)
        time.sleep(0.3)

        # Submit
        self._submit_prompt(text_input)

        # Wait for image
        print("⏳ Waiting for image generation...")
        img_el = self._wait_for_image(max_wait=self.wait_timeout)

        # Download at original resolution
        filepath = self._download_image(img_el, title=title)
        print(f"✓ Saved: {filepath}")
        return filepath

    # ── Multi-turn conversation mode ────────────────────────

    def start_conversation(
        self,
        prompt,
        photo_paths=None,
        photo_path=None,
        title=None,
        template_path=None,
        editing_mode=False,
    ):
        """Start a multi-turn conversation. Opens new chat, generates first image.

        Unlike generate(), the conversation stays open so you can call
        converse() to iterate on the result.

        Args:
            prompt: Text prompt (short instruction when using template)
            photo_paths: List of user's selfie paths — all uploaded photos
            photo_path: Single photo path (legacy, converted to photo_paths=[...])
            title: Filename prefix
            template_path: Style template image — defines the VISUAL STYLE to match.
            editing_mode: If True, upload user photos FIRST and the template LAST
                ("编辑我的人物，参考这个风格"). If False, upload template FIRST and
                user photos AFTER ("生成这种风格，参考这个人的脸").

        Returns:
            Path to first saved image
        """
        # Backward compat: accept single photo_path
        if photo_paths is None and photo_path is not None:
            photo_paths = [photo_path]
        elif photo_paths is None:
            photo_paths = []

        self.ensure_gemini_page()

        print("  Opening new conversation...")
        self._new_chat()
        time.sleep(2)

        # Reset tracking
        self._known_image_srcs.clear()
        self._uploaded_fingerprints.clear()
        self._in_conversation = True

        # Upload order matters for the prompt:
        #   editing_mode=True:  user selfies first, template last
        #   editing_mode=False: template first, user selfies after
        if editing_mode:
            for pp in photo_paths:
                if pp and Path(pp).exists():
                    self._upload_photo(pp)
                    time.sleep(1)
                    print(f"  📷 Uploaded reference photo: {Path(pp).name}")
            if template_path and Path(template_path).exists():
                self._upload_photo(template_path)
                time.sleep(1)
                print(f"  🎨 Uploaded style template: {Path(template_path).name}")
        else:
            if template_path and Path(template_path).exists():
                self._upload_photo(template_path)
                time.sleep(1)
                print(f"  🎨 Uploaded style template: {Path(template_path).name}")
            for pp in photo_paths:
                if pp and Path(pp).exists():
                    self._upload_photo(pp)
                    time.sleep(1)
                    print(f"  📷 Uploaded reference photo: {Path(pp).name}")

        # Snapshot before generation & fingerprint uploaded images
        self._snapshot_images()
        self._fingerprint_uploaded_photos(*photo_paths, template_path)

        # Send prompt
        text_input = self._find_text_input()
        self.driver.execute_script("arguments[0].focus();", text_input)
        text_input.click()
        time.sleep(0.3)
        text_input.clear()
        time.sleep(0.2)
        text_input.send_keys(prompt)
        time.sleep(0.3)
        self._submit_prompt(text_input)

        print("⏳ Waiting for image generation...")
        img_el = self._wait_for_image(max_wait=self.wait_timeout)

        # Track this image so next converse() won't pick it up
        self._track_image(img_el)

        filepath = self._download_image(img_el, title=title or "turn_1")
        print(f"✓ Turn 1 saved: {filepath}")
        return filepath

    def converse(self, prompt, title=None, turn_number=None):
        """Send a follow-up prompt in the SAME conversation (multi-turn editing).

        Gemini sees the full context — previous images, the reference photo,
        and all prior prompts. So you can say things like "change the background
        to blue" or "make the expression more natural" and it will edit the
        existing result.

        Args:
            prompt: Follow-up instruction (e.g. "change background to blue")
            title: Filename prefix (optional)
            turn_number: Turn number for filename (optional)

        Returns:
            Path to saved image
        """
        if not self._in_conversation:
            raise RuntimeError("No active conversation. Call start_conversation() first.")

        turn = turn_number or 2
        print(f"\n🔄 Turn {turn}: {prompt[:80]}...")

        # Snapshot current images to distinguish old from new
        self._snapshot_images()

        # Find the input (Gemini keeps the same input box in conversation)
        text_input = self._find_text_input()
        self.driver.execute_script("arguments[0].focus();", text_input)
        text_input.click()
        time.sleep(0.3)
        text_input.clear()
        time.sleep(0.2)
        text_input.send_keys(prompt)
        time.sleep(0.3)
        self._submit_prompt(text_input)

        print(f"⏳ Waiting for edited image...")
        img_el = self._wait_for_image(max_wait=self.wait_timeout)

        self._track_image(img_el)

        filepath = self._download_image(img_el, title=title or f"turn_{turn}")
        print(f"✓ Turn {turn} saved: {filepath}")
        return filepath

    def converse_text(self, prompt, timeout=None) -> str:
        """Send a follow-up prompt and wait for a TEXT response (not an image).

        Used by the resemblance agent to ask Gemini to rate/compare faces.
        Gemini's text response appears in model-response/message-content elements.

        Args:
            prompt: The question to ask (e.g. resemblance rating prompt)
            timeout: Max seconds to wait for response text. None → use the
                client's configured wait_timeout (same budget as image gen).

        Returns:
            The text content of Gemini's response
        """
        if not self._in_conversation:
            raise RuntimeError("No active conversation. Call start_conversation() first.")
        if timeout is None:
            timeout = self.wait_timeout

        # Snapshot images so re-rendered images don't confuse future _wait_for_image calls
        self._snapshot_images()

        # Send prompt (same flow as converse())
        text_input = self._find_text_input()
        self.driver.execute_script("arguments[0].focus();", text_input)
        text_input.click()
        time.sleep(0.3)
        text_input.clear()
        time.sleep(0.2)
        text_input.send_keys(prompt)
        time.sleep(0.3)
        self._submit_prompt(text_input)

        # Wait for TEXT response instead of image. Pass the prompt so the waiter
        # can strip Gemini's echo of it and isolate just the model's reply.
        response_text = self._wait_for_text_response(prompt=prompt, timeout=timeout)
        return response_text

    def end_conversation(self):
        """End the current multi-turn conversation."""
        self._in_conversation = False
        self._known_image_srcs.clear()
        print("✓ Conversation ended.")

    # ── Text response helpers ──────────────────────────────
    #
    # DOM-agnostic by design. We deliberately do NOT walk response containers
    # with CSS selectors: Gemini 2026 renders model replies inside the same
    # conversation containers it uses for the user's prompt echo and uploaded
    # media, and the guessed selectors (model-response / message-content) match
    # nothing in the live DOM — so the previous element-based waiter spun until
    # its 60s timeout and the resemblance agent silently no-op'd (score=None,
    # every image "accepted"). This mirrors the proven _wait_for_image strategy:
    # snapshot document.body.innerText, then detect NEW text once streaming
    # settles. Completion is signalled two ways — (a) a NEW "评分：X/10" score
    # token appears (the judge's verdict landed), or (b) body length is stable
    # for ~6s with fresh content.

    def _wait_for_text_response(self, prompt: str = "", timeout: int = 60) -> str:
        """Wait for Gemini's TEXT reply and return its text.

        Args:
            prompt: The prompt we just sent, used to strip Gemini's echo of it
                so only the model's reply is returned.
            timeout: Max seconds to wait.
        """
        body_before = self._body_text()  # already includes the echoed prompt
        score_before = self._last_score_token(body_before)
        start = time.time()
        print(f"  ⏳ Waiting for text response (body_len={len(body_before)}, "
              f"timeout={timeout}s)...")
        time.sleep(2)  # let Gemini start processing

        last_len = len(body_before)
        stable = 0
        last_body = body_before

        while time.time() - start < timeout:
            elapsed = time.time() - start
            body_now = self._body_text()
            if len(body_now) == last_len:
                stable += 1
            else:
                stable = 0
                last_len = len(body_now)
            last_body = body_now

            if int(elapsed) % 10 == 0 and int(elapsed) != getattr(self, "_last_text_diag", -1):
                self._last_text_diag = int(elapsed)
                reply = self._extract_reply(body_now, body_before, prompt)
                print(f"    [{elapsed:.0f}s] body_len={len(body_now)} stable={stable} "
                      f"reply_len={len(reply)}")

            # Fast path: a NEW score token appeared → judge delivered its verdict.
            score_now = self._last_score_token(body_now)
            if score_now and score_now != score_before:
                reply = self._extract_reply(body_now, body_before, prompt)
                if reply:
                    print(f"  📝 Judge text reply received (score marker {score_now}) — "
                          f"{len(reply)} chars")
                    return reply
                return score_now

            # Slow path: body settled (~6s no growth) with fresh content.
            if stable >= 3 and len(body_now) > len(body_before) + 5:
                reply = self._extract_reply(body_now, body_before, prompt)
                if len(reply) > 5:
                    print(f"  📝 Text response received (stable) — {len(reply)} chars")
                    return reply
            time.sleep(2)

        # Timeout: return the model's reply only if one actually accumulated.
        reply = self._extract_reply(last_body, body_before, prompt)
        if len(reply) > 5:
            print(f"  ⚠  Text response timed out, returning partial ({len(reply)} chars)")
            return reply
        raise TimeoutException(f"Text response timed out after {timeout}s")

    def _body_text(self) -> str:
        """Snapshot the page's visible text. Wrapped so a transient driver
        hiccup never crashes the polling loop."""
        try:
            return self.driver.execute_script("return document.body.innerText || '';") or ""
        except Exception:
            return ""

    def _last_score_token(self, text: str) -> str:
        """Return the LAST score token in ``text`` (e.g. '8/10'), or '' if none.

        Only matches DIGIT scores, so the judge prompt's literal template
        '评分：X/10' (X is a letter) does not count. Used as a completion signal:
        a score that wasn't present before sending means this turn's verdict
        landed. Authoritative parsing lives in gemini_worker._parse_judge_response.
        """
        import re
        patterns = (
            r"评分[:：]\s*(\d{1,2})\s*/\s*10",   # 评分：8/10
            r"评分[:：]?\s*(\d{1,2})\s*分",       # 评分8分
            r"\b(\d{1,2})\s*/\s*10\b",            # bare 8/10
        )
        for pat in patterns:
            m = re.findall(pat, text)
            if m:
                return f"{m[-1]}/10"
        return ""

    def _extract_reply(self, body_now: str, body_before: str, prompt: str) -> str:
        """Isolate the model's reply as the NEW text added since ``body_before``.

        Locale- and DOM-agnostic: we take lines present in ``body_now`` but NOT in
        ``body_before`` (the snapshot captured right after sending our prompt),
        then drop (a) Gemini's echo of the prompt itself and (b) known UI/footer
        lines. What remains is the model's reply. Returns '' when Gemini added no
        new text (no reply), so the caller can distinguish a real (possibly
        partial) reply from silence and time out cleanly.
        """
        before_lines = {ln.strip() for ln in body_before.splitlines()}
        prompt_words = set()
        if prompt:
            prompt_words = {w for w in prompt.replace("\n", " ").split() if w}
        footer = {
            "Flash", "Gemini said", "You said", "Conversation with Gemini",
            "Gemini is AI and can make mistakes.", "Gemini is typing",
            "Gemini replied", "Your move", "What's the vibe",
        }

        reply_lines: list[str] = []
        for ln in body_now.splitlines():
            s = ln.strip()
            if not s or s in before_lines:
                continue
            # Drop the echoed prompt (whole line or any of its tokens-only lines).
            if prompt and (s in prompt or s in prompt_words):
                continue
            if s in footer:
                continue
            reply_lines.append(ln)
        return "\n".join(reply_lines).strip()



    # ── Image tracking helpers ──────────────────────────────

    def _snapshot_images(self):
        """Record all currently visible image blob URLs."""
        self._known_image_srcs.clear()
        try:
            all_imgs = self.driver.find_elements(By.CSS_SELECTOR, "img")
            for img in all_imgs:
                try:
                    src = img.get_attribute("src") or ""
                    if src and (src.startswith("blob:") or src.startswith("data:image")):
                        w = img.size["width"]
                        h = img.size["height"]
                        if w > 50 and h > 50:
                            self._known_image_srcs.add(src)
                except Exception:
                    continue
        except Exception:
            pass

    def _track_image(self, img_element):
        """Add an image to the known set so future turns skip it."""
        try:
            src = img_element.get_attribute("src") or ""
            if src:
                self._known_image_srcs.add(src)
        except Exception:
            pass

    def generate_batch(self, prompts_list, delay=5):
        """Generate multiple images in sequence.

        Args:
            prompts_list: List of dicts with keys: id, prompt, style, ...
            delay: Seconds between generations

        Returns:
            List of (entry, filepath_or_error)
        """
        self.ensure_gemini_page()
        results = []

        for i, entry in enumerate(prompts_list):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(prompts_list)}] {entry.get('id', '?')} "
                  f"| {entry.get('style_label', '?')} | {entry.get('gender', '?')}")
            print(f"{'='*60}")

            try:
                filepath = self.generate(
                    prompt=entry["prompt"],
                    title=entry.get("id"),
                )
                results.append((entry, filepath))
            except Exception as e:
                print(f"✗ Failed: {e}")
                results.append((entry, str(e)))

            # New chat before next
            if i < len(prompts_list) - 1:
                self._new_chat()
                time.sleep(delay)

        return results

    def _upload_photo(self, photo_path):
        """Upload a photo into Gemini's input."""
        photo_path = Path(photo_path)
        with open(photo_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        ext = photo_path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp"}.get(ext, "image/png")

        text_input = self._find_text_input()
        script = """
        var base64 = arguments[0], mime = arguments[1], name = arguments[2], input = arguments[3];
        input.focus(); input.click();
        var bin = atob(base64), bytes = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        var blob = new Blob([bytes], {type: mime});
        var file = new File([blob], name, {type: mime});
        var dt = new DataTransfer(); dt.items.add(file);
        input.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true}));
        input.dispatchEvent(new Event('input', {bubbles: true}));
        """
        self.driver.execute_script(script, img_data, mime, photo_path.name, text_input)
        time.sleep(1)
        print(f"  📷 Uploaded reference photo: {photo_path.name}")

    def _submit_prompt(self, text_input):
        """Submit the prompt via JS click (bypasses overlay interception)."""
        submit_selectors = [
            (By.CSS_SELECTOR, "button[aria-label*='Send']"),
            (By.CSS_SELECTOR, "button[aria-label*='Submit']"),
            (By.XPATH, "//button[contains(@aria-label, 'Send')]"),
        ]
        for by, sel in submit_selectors:
            try:
                btn = self.driver.find_element(by, sel)
                if btn.is_displayed() and btn.is_enabled():
                    # Use JS click to bypass overlay elements (e.g. upload container)
                    self.driver.execute_script("arguments[0].click();", btn)
                    return
            except NoSuchElementException:
                continue
        # Fallback: Enter key
        text_input.send_keys(Keys.RETURN)

    def _wait_for_image(self, max_wait=120):
        """Wait for a NEW generated image to appear.

        Strategy: record ALL img srcs present at the start (including uploaded
        photos).  A "new" image is one whose src was NOT present when we began
        waiting.  We no longer use DOM container walking or fingerprint matching
        because Gemini 2026 places generated images in the same containers as
        uploaded photos.
        """
        MIN_WAIT = 8  # seconds — ignore candidates before this
        start = time.time()
        time.sleep(3)  # Let Gemini start processing

        # Snapshot all blob/data srcs present at t=3s — these are uploads + UI
        initial_srcs: set[str] = set()
        try:
            for img in self.driver.find_elements(By.CSS_SELECTOR, "img"):
                src = img.get_attribute("src") or ""
                if src.startswith("blob:") or src.startswith("data:"):
                    initial_srcs.add(src)
        except Exception:
            pass
        initial_srcs.update(self._known_image_srcs)
        print(f"  📸 Initial src snapshot: {len(initial_srcs)} blob/data URLs")

        while time.time() - start < max_wait:
            elapsed = time.time() - start
            try:
                all_imgs = self.driver.find_elements(By.CSS_SELECTOR, "img")
                candidates = []
                for idx, img in enumerate(all_imgs):
                    try:
                        if not img.is_displayed():
                            continue
                        src = img.get_attribute("src") or ""
                        w = img.size["width"]
                        h = img.size["height"]
                        if w < 100 or h < 100:
                            continue

                        # CORE FILTER: only accept images with a NEW src
                        # (not in the initial snapshot and not from prior turns)
                        if src in initial_srcs:
                            continue

                        # Skip UI icons and avatars
                        if "gstatic.com" in src or "googleusercontent.com" in src:
                            continue

                        candidates.append((img, w * h, src))
                    except Exception:
                        continue

                if candidates and elapsed >= MIN_WAIT:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    best = candidates[0]
                    nw = self.driver.execute_script(
                        "return arguments[0].naturalWidth;", best[0]
                    )
                    nh = self.driver.execute_script(
                        "return arguments[0].naturalHeight;", best[0]
                    )
                    bw = best[0].size["width"]
                    bh = best[0].size["height"]
                    print(f"  🎯 Found new image: {bw}×{bh} nat={nw}×{nh}, "
                          f"elapsed={elapsed:.0f}s, src={best[2][:60]}...")
                    return best[0]

            except Exception:
                pass
            time.sleep(2)

        raise TimeoutException(f"Image generation timed out after {max_wait}s")

    def _fingerprint_uploaded_photos(self, *paths):
        """Record naturalWidth x naturalHeight of uploaded photos so we can
        detect when Gemini re-renders them in the response area."""
        self._uploaded_fingerprints.clear()
        for path in paths:
            if not path or not Path(path).exists():
                continue
            try:
                from PIL import Image
                img = Image.open(path)
                self._uploaded_fingerprints.append((img.width, img.height))
                print(f"  📋 Uploaded photo fingerprint: {img.width}×{img.height}")
            except Exception:
                pass

    def _matches_uploaded_fingerprint(self, img_element) -> bool:
        """Check if an image element has the same natural dimensions as an uploaded photo."""
        if not self._uploaded_fingerprints:
            return False
        try:
            nw = self.driver.execute_script(
                "return arguments[0].naturalWidth;", img_element
            )
            nh = self.driver.execute_script(
                "return arguments[0].naturalHeight;", img_element
            )
            for uw, uh in self._uploaded_fingerprints:
                # Allow 5% tolerance for potential resize/re-encode
                if abs(nw - uw) < uw * 0.05 and abs(nh - uh) < uh * 0.05:
                    return True
        except Exception:
            pass
        return False

    def _is_in_upload_container(self, img_element) -> bool:
        """Check if an image is inside an upload/preview/input container.
        These are the user-uploaded photos, not Gemini-generated ones."""
        try:
            # Walk up the DOM tree looking for upload-related containers
            result = self.driver.execute_script("""
                var el = arguments[0];
                var uploadClasses = ['upload', 'file-preview', 'attachment', 'input-area',
                                     'composer', 'prompt-area', 'user-content'];
                while (el && el !== document.body) {
                    el = el.parentElement;
                    if (!el) break;
                    var cls = (el.className || '').toLowerCase();
                    var role = (el.getAttribute('role') || '').toLowerCase();
                    for (var i = 0; i < uploadClasses.length; i++) {
                        if (cls.indexOf(uploadClasses[i]) !== -1) return true;
                    }
                    if (role === 'textbox') return true;
                }
                return false;
            """, img_element)
            return result
        except Exception:
            return False

    @staticmethod
    def _remove_watermark(filepath: Path):
        """Remove Gemini watermark from saved image file."""
        try:
            from PIL import Image as PILImage
            img = PILImage.open(filepath)
            cleaned = remove_gemini_watermark(img)
            cleaned.save(filepath, format="PNG")
        except Exception as e:
            print(f"  ⚠  Watermark removal failed: {e}")

    def _download_image(self, img_element, title=None):
        """Extract image at original resolution via JS canvas."""
        def _next_filename():
            # UUID suffix so two downloads in the same second never collide
            # (timestamp-only names silently overwrote each other).
            short = uuid.uuid4().hex[:8]
            base = title if title else "gemini"
            return f"{base}_{short}.png"

        # JS canvas extraction — full naturalWidth × naturalHeight
        canvas_script = """
        var img = arguments[0];
        if (img.complete && img.naturalWidth > 0) {
            var canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            canvas.getContext('2d').drawImage(img, 0, 0);
            return canvas.toDataURL('image/png');
        }
        return null;
        """
        data_url = self.driver.execute_script(canvas_script, img_element)

        if data_url and data_url.startswith("data:image"):
            _, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)

            filename = _next_filename()
            filepath = self.output_dir / filename
            with open(filepath, "wb") as f:
                f.write(image_bytes)

            w = self.driver.execute_script("return arguments[0].naturalWidth;", img_element)
            h = self.driver.execute_script("return arguments[0].naturalHeight;", img_element)
            print(f"  📐 Original resolution: {w}×{h}")

            # Remove Gemini watermark from bottom-right corner
            self._remove_watermark(filepath)
            return str(filepath)

        # Fallback: try blob fetch
        src = img_element.get_attribute("src") or ""
        if src.startswith("blob:"):
            fetch_script = """
            var img = arguments[0];
            var cb = arguments[arguments.length - 1];
            fetch(img.src).then(r => r.blob()).then(b => {
                var reader = new FileReader();
                reader.onload = () => cb(reader.result);
                reader.onerror = () => cb(null);
                reader.readAsDataURL(b);
            }).catch(() => cb(null));
            """
            data_url = self.driver.execute_async_script(fetch_script, img_element)
            if data_url and data_url.startswith("data:image"):
                _, encoded = data_url.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                filename = _next_filename()
                filepath = self.output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                self._remove_watermark(filepath)
                return str(filepath)

        # Last resort: screenshot
        filename = _next_filename()
        filepath = self.output_dir / filename
        img_element.screenshot(str(filepath))
        print(f"  ⚠  Used screenshot fallback (may be lower res)")
        self._remove_watermark(filepath)
        return str(filepath)

    def _new_chat(self):
        """Start a new Gemini chat."""
        self._in_conversation = False
        self._known_image_srcs.clear()
        selectors = [
            (By.XPATH, "//a[@aria-label='New chat']"),
            (By.XPATH, "//a[contains(@aria-label, 'New chat')]"),
            (By.XPATH, "//a[@href='/app']"),
        ]
        for by, sel in selectors:
            try:
                el = self.driver.find_element(by, sel)
                if el.is_displayed():
                    el.click()
                    time.sleep(2)
                    return
            except Exception:
                continue
        # Fallback: navigate directly
        self.driver.get(GEMINI_URL)
        time.sleep(2)

    def disconnect(self):
        """Disconnect from Chrome (keeps Chrome running)."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            print("✓ Disconnected from Chrome (browser stays open)")


# ──────────────────────────────────────────────
# Prompt loading helpers
# ──────────────────────────────────────────────
def load_prompts(prompts_file="prompts.json"):
    path = HERE / prompts_file
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_prompts(data, style_filter=None, gender_filter=None):
    results = []
    for style_key, style_data in data["styles"].items():
        if style_filter and style_key != style_filter:
            continue
        for p in style_data["prompts"]:
            if gender_filter and p.get("gender") != gender_filter:
                continue
            results.append({
                "id": p["id"],
                "prompt": p["prompt"],
                "style": style_key,
                "style_label": style_data["label"],
                "gender": p.get("gender", "unknown"),
            })
    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Persistent Gemini Headshot Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # launch
    p_launch = sub.add_parser("launch", help="Launch Chrome with CDP (run once)")
    p_launch.add_argument("--port", type=int, default=CDP_PORT)

    # generate (single)
    p_gen = sub.add_parser("generate", help="Generate a single image")
    p_gen.add_argument("prompt", nargs="?", help="Prompt text (or use --id)")
    p_gen.add_argument("--id", help="Prompt ID from prompts.json (e.g. bf_m_01)")
    p_gen.add_argument("--photo", help="Reference photo path")
    p_gen.add_argument("--output", default=None, help="Output directory")
    p_gen.add_argument("--timeout", type=int, default=120, help="Generation timeout (seconds)")

    # batch
    p_batch = sub.add_parser("batch", help="Batch generate from prompts.json")
    p_batch.add_argument("--all", action="store_true")
    p_batch.add_argument("--style", choices=["business_formal", "academic", "id_photo", "light_workplace"])
    p_batch.add_argument("--gender", choices=["male", "female"])
    p_batch.add_argument("--delay", type=int, default=5)
    p_batch.add_argument("--output", default=None)
    p_batch.add_argument("--timeout", type=int, default=120)

    # status
    sub.add_parser("status", help="Check if Chrome CDP is running")

    # chat (multi-turn interactive)
    p_chat = sub.add_parser("chat", help="Multi-turn conversation: iterate on a portrait")
    p_chat.add_argument("--photo", required=True, help="Reference photo to upload")
    p_chat.add_argument("--prompt", default=None,
                        help="Initial prompt (if omitted, enters interactive mode)")
    p_chat.add_argument("--output", default=None, help="Output directory")
    p_chat.add_argument("--timeout", type=int, default=120, help="Generation timeout (seconds)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # ── launch ──
    if args.command == "launch":
        launch_chrome(port=args.port)

    # ── status ──
    elif args.command == "status":
        if check_chrome_running():
            print(f"✓ Chrome is running with CDP on port {CDP_PORT}")
        else:
            print(f"✗ Chrome NOT detected on port {CDP_PORT}")
            print(f"  Run: python persistent_client.py launch")

    # ── generate ──
    elif args.command == "generate":
        prompt_text = args.prompt

        # If --id specified, load from prompts.json
        if args.id:
            data = load_prompts()
            all_prompts = flatten_prompts(data)
            match = [p for p in all_prompts if p["id"] == args.id]
            if not match:
                print(f"✗ Prompt ID '{args.id}' not found in prompts.json")
                print(f"  Available IDs: {', '.join(p['id'] for p in all_prompts)}")
                return
            prompt_text = match[0]["prompt"]
            title = match[0]["id"]
            print(f"📋 Loaded prompt: {args.id}")
        else:
            title = None

        if not prompt_text:
            print("✗ Provide a prompt text or --id <prompt_id>")
            return

        output_dir = args.output or str(HERE / "output")
        client = PersistentGeminiClient(output_dir=output_dir, wait_timeout=args.timeout)
        try:
            client.connect()
            filepath = client.generate(prompt_text, title=title, photo_path=args.photo)
            print(f"\n✅ Done: {filepath}")
        except Exception as e:
            print(f"\n✗ Error: {e}")
        finally:
            client.disconnect()

    # ── batch ──
    elif args.command == "batch":
        if not args.all and not args.style:
            print("✗ Specify --all or --style <style>")
            return

        data = load_prompts()
        prompts = flatten_prompts(data, style_filter=args.style, gender_filter=args.gender)
        if not prompts:
            print("No prompts matched the filters.")
            return

        print(f"📋 {len(prompts)} prompt(s) to generate:")
        for p in prompts:
            print(f"  • {p['id']}: {p['style_label']} / {p['gender']}")

        output_dir = args.output or str(HERE / "output")
        client = PersistentGeminiClient(output_dir=output_dir, wait_timeout=args.timeout)
        results = []
        try:
            client.connect()
            results = client.generate_batch(prompts, delay=args.delay)
        except Exception as e:
            print(f"\n✗ Fatal error: {e}")
        finally:
            client.disconnect()

        # Save log
        succeeded = sum(1 for _, r in results if not r.startswith("✗") and not r.startswith("Error"))
        failed = len(results) - succeeded
        log = {
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "results": [
                {"id": e.get("id"), "path": r, "style": e.get("style")}
                for e, r in results
            ],
        }
        log_path = Path(output_dir) / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"Batch done: {succeeded}✓ {failed}✗ | Log: {log_path}")
        print(f"{'='*60}")

    # ── chat (multi-turn) ──
    elif args.command == "chat":
        if not Path(args.photo).exists():
            print(f"✗ Photo not found: {args.photo}")
            return

        output_dir = args.output or str(HERE / "output" / "chat")
        client = PersistentGeminiClient(output_dir=output_dir, wait_timeout=args.timeout)
        try:
            client.connect()

            # Initial prompt
            if args.prompt:
                initial_prompt = args.prompt
            else:
                initial_prompt = (
                    "根据这张照片中人物的面部特征，生成一张专业人像照。"
                    "要求：严格保持五官、脸型和肤色不变，"
                    "专业影棚光线，浅灰色渐变背景，穿深色西装，"
                    "表情自然自信，自然皮肤质感，毛孔可见。"
                )

            print(f"\n{'='*60}")
            print("📸 Starting multi-turn portrait session")
            print(f"   Reference: {args.photo}")
            print(f"{'='*60}\n")

            filepath = client.start_conversation(
                prompt=initial_prompt,
                photo_path=args.photo,
            )
            turn = 2

            # Interactive loop
            print(f"\n{'='*60}")
            print("💬 Enter follow-up instructions to iterate.")
            print("   Examples: '背景换成蓝色', '表情再微笑一点', '换成白衬衫'")
            print("   Type 'quit' or 'done' to end, 'new' to start over.")
            print(f"{'='*60}\n")

            while True:
                try:
                    user_input = input(f"  Turn {turn} > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "done", "exit", "q"):
                    break
                if user_input.lower() == "new":
                    print("Starting new conversation...")
                    client.end_conversation()
                    filepath = client.start_conversation(
                        prompt=initial_prompt,
                        photo_path=args.photo,
                    )
                    turn = 2
                    continue

                try:
                    filepath = client.converse(
                        prompt=user_input,
                        turn_number=turn,
                    )
                    turn += 1
                except Exception as e:
                    print(f"✗ Error: {e}")
                    print("  You can continue or type 'new' to restart.")

            client.end_conversation()
            print(f"\n✅ Session ended. Images saved to: {output_dir}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
        finally:
            client.disconnect()


if __name__ == "__main__":
    main()
