#!/usr/bin/env python3
"""
URL Scraper — Extract design tokens from any URL.
Uses Playwright for rendering + CSS extraction, WhatFontIs for font detection.
"""

import base64
import re
import json
from collections import Counter

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

import requests


def scrape_url(url: str, whatfontis_key: str = None, viewport: dict = None) -> dict:
    """
    Load a URL, extract design tokens.

    Returns:
    {
        "url": str,
        "screenshot_b64": str or None,
        "dominant_fonts": [str, ...],
        "color_palette": [str, ...],
        "layout_pattern": str,
        "hero_structure": dict,
        "button_style": dict,
        "spacing_system": str,
        "texture": str,
        "errors": [str, ...]
    }
    """
    if not HAS_PLAYWRIGHT:
        return {"error": "Playwright not installed. Run: pip install playwright && playwright install chromium"}

    vp = viewport or {"width": 1440, "height": 900}
    result = {
        "url": url,
        "screenshot_b64": None,
        "dominant_fonts": [],
        "color_palette": [],
        "layout_pattern": "unknown",
        "hero_structure": {},
        "button_style": {},
        "spacing_system": "unknown",
        "texture": "none",
        "errors": []
    }

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport=vp)
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)  # Let fonts settle

            # ── Screenshot (full page, capped at 4000px to avoid huge files) ──
            try:
                screenshot = page.screenshot(full_page=False)
                result["screenshot_b64"] = f"data:image/png;base64,{base64.b64encode(screenshot).decode()}"
            except Exception as e:
                result["errors"].append(f"Screenshot failed: {e}")

            # ── Extract CSS tokens via JavaScript ──
            tokens = page.evaluate("""
            () => {
                const results = {
                    fonts: {},
                    colors: {},
                    buttons: [],
                    headings: [],
                    hero_sections: [],
                    body_bg: '',
                    body_font: '',
                    body_color: ''
                };

                // ── Body defaults ──
                const bodyStyle = getComputedStyle(document.body);
                results.body_bg = bodyStyle.backgroundColor;
                results.body_font = bodyStyle.fontFamily;
                results.body_color = bodyStyle.color;

                // ── Scan all text elements for fonts + colors ──
                const textElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, span, a, button, li, div');

                textElements.forEach(el => {
                    const style = getComputedStyle(el);
                    const tag = el.tagName.toLowerCase();
                    const text = el.textContent.trim().slice(0, 100);
                    if (!text || text.length < 3) return;

                    // Font
                    const font = style.fontFamily.split(',')[0].replace(/['"]/g, '').trim();
                    if (font && !font.includes('sans-serif') && !font.includes('serif') && !font.includes('monospace')) {
                        results.fonts[font] = (results.fonts[font] || 0) + 1;
                    }

                    // Color
                    const color = style.color;
                    if (color && color !== 'rgb(0, 0, 0)' && color !== 'rgba(0, 0, 0, 0)') {
                        results.colors[color] = (results.colors[color] || 0) + 1;
                    }

                    // Background color (non-transparent)
                    const bg = style.backgroundColor;
                    if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                        results.colors[bg] = (results.colors[bg] || 0) + 1;
                    }

                    // Headings
                    if (['h1', 'h2', 'h3'].includes(tag)) {
                        results.headings.push({
                            tag,
                            text: text.slice(0, 80),
                            fontSize: style.fontSize,
                            fontWeight: style.fontWeight,
                            color,
                            fontFamily: style.fontFamily.split(',')[0].replace(/['"]/g, '').trim()
                        });
                    }
                });

                // ── Scan buttons ──
                document.querySelectorAll('button, a[class*="btn"], a[class*="button"], [role="button"]').forEach(el => {
                    const style = getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    results.buttons.push({
                        text: el.textContent.trim().slice(0, 50),
                        bg: style.backgroundColor,
                        color: style.color,
                        borderRadius: style.borderRadius,
                        padding: style.padding,
                        fontSize: style.fontSize,
                        fontWeight: style.fontWeight,
                        textTransform: style.textTransform,
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    });
                });

                // ── Detect hero section ──
                // Look for the first large section near the top
                const sections = document.querySelectorAll('section, header, [class*="hero"], [class*="banner"], [id*="hero"]');
                if (sections.length > 0) {
                    const hero = sections[0];
                    const heroRect = hero.getBoundingClientRect();
                    const heroStyle = getComputedStyle(hero);
                    results.hero_sections.push({
                        tag: hero.tagName.toLowerCase(),
                        classes: hero.className,
                        width: Math.round(heroRect.width),
                        height: Math.round(heroRect.height),
                        bg: heroStyle.backgroundColor,
                        bgImage: heroStyle.backgroundImage !== 'none' ? heroStyle.backgroundImage : null,
                        padding: heroStyle.padding,
                        textAlign: heroStyle.textAlign
                    });
                }

                return results;
            }
            """)

            # ── Process fonts (top 5 by frequency) ──
            font_counts = Counter(tokens.get("fonts", {}))
            result["dominant_fonts"] = [f for f, _ in font_counts.most_common(5)]
            if tokens.get("body_font"):
                body_font = tokens["body_font"].split(",")[0].replace("'", "").replace('"', "").strip()
                if body_font and body_font not in result["dominant_fonts"][:3]:
                    result["dominant_fonts"].insert(0, body_font)

            # ── Process colors (top 10 by frequency, deduplicated by similarity) ──
            color_counts = Counter(tokens.get("colors", {}))
            result["color_palette"] = _deduplicate_colors([c for c, _ in color_counts.most_common(15)])

            # ── Button style (most common) ──
            buttons = tokens.get("buttons", [])
            if buttons:
                # Pick the most representative button (largest sample)
                primary = max(buttons, key=lambda b: len(b.get("text", "")))
                result["button_style"] = {
                    "bg": primary.get("bg"),
                    "color": primary.get("color"),
                    "border_radius": primary.get("borderRadius"),
                    "padding": primary.get("padding"),
                    "font_size": primary.get("fontSize"),
                    "font_weight": primary.get("fontWeight"),
                    "text_transform": primary.get("textTransform")
                }

            # ── Hero structure ──
            heroes = tokens.get("hero_sections", [])
            if heroes:
                h = heroes[0]
                result["hero_structure"] = {
                    "tag": h["tag"],
                    "width": h["width"],
                    "height": h["height"],
                    "bg_color": h["bg"],
                    "bg_image": h.get("bgImage"),
                    "padding": h["padding"],
                    "text_align": h["textAlign"]
                }
                # Classify layout pattern
                if h["width"] > 1000 and h["height"] > 400:
                    result["layout_pattern"] = "fullscreen-hero"
                elif h["textAlign"] == "center":
                    result["layout_pattern"] = "hero-centered"
                else:
                    result["layout_pattern"] = "hero-split"

            # ── Headings for font confirmation ──
            headings = tokens.get("headings", [])
            if headings:
                h1s = [h for h in headings if h["tag"] == "h1"]
                if h1s:
                    result["_heading_sample"] = h1s[0]

            # ── Body defaults ──
            if tokens.get("body_bg"):
                result["body_bg"] = tokens["body_bg"]

            # ── Spacing estimate ──
            if heroes and heroes[0]["height"] > 600:
                result["spacing_system"] = "generous"
            elif heroes and heroes[0]["height"] > 400:
                result["spacing_system"] = "standard"
            else:
                result["spacing_system"] = "compact"

            # ── Texture detection (crude) ──
            body_bg = tokens.get("body_bg", "")
            if "gradient" in str(heroes[0].get("bgImage", "")).lower() if heroes else False:
                result["texture"] = "gradient"
            elif body_bg and body_bg not in ("rgb(255, 255, 255)", "rgba(0,0,0,0)", "#fff", "#ffffff"):
                # Non-white background — could be textured
                result["texture"] = "solid-color"

            browser.close()

    except Exception as e:
        result["errors"].append(f"Browser scraping failed: {e}")

    # ── Optional: WhatFontIs font detection on screenshot ──
    if whatfontis_key and result["screenshot_b64"]:
        try:
            # Strip data URL prefix for API
            b64_data = result["screenshot_b64"].split(",", 1)[1] if "," in result["screenshot_b64"] else result["screenshot_b64"]
            wfi_resp = requests.post(
                "https://www.whatfontis.com/api2/",
                data={
                    "API_KEY": whatfontis_key,
                    "IMAGEBASE64": "1",
                    "urlimagebase64": b64_data,
                    "FREEFONTS": "1",
                    "limit": "10",
                    "NOTTEXTBOXSDETECTION": "0"
                },
                timeout=30
            )
            if wfi_resp.status_code == 200:
                wfi_fonts = wfi_resp.json()
                if isinstance(wfi_fonts, list):
                    wfi_names = [f["title"] for f in wfi_fonts[:5]]
                    # Merge: WFI fonts take priority for dominant position
                    existing = set(f.lower() for f in result["dominant_fonts"])
                    for wf in wfi_names:
                        if wf.lower() not in existing:
                            result["dominant_fonts"].append(wf)
                    result["_wfi_fonts"] = wfi_names
            else:
                result["errors"].append(f"WhatFontIs error: HTTP {wfi_resp.status_code}")
        except Exception as e:
            result["errors"].append(f"WhatFontIs failed: {e}")

    return result


def _rgb_to_hex(rgb_str: str) -> str:
    """Convert rgb(r, g, b) or rgba(r, g, b, a) to #rrggbb."""
    match = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)', rgb_str)
    if match:
        return "#{:02x}{:02x}{:02x}".format(*map(int, match.groups()))
    return rgb_str


def _color_similarity(c1: str, c2: str) -> bool:
    """Return True if two hex colors are visually similar."""
    c1 = _rgb_to_hex(c1).lstrip("#")
    c2 = _rgb_to_hex(c2).lstrip("#")
    if len(c1) != 6 or len(c2) != 6:
        return c1 == c2
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    # Simple Euclidean distance in RGB space
    dist = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    return dist < 30  # Very generous threshold


def _deduplicate_colors(colors: list) -> list:
    """Remove visually similar adjacent colors from palette."""
    result = []
    for c in colors:
        c_hex = _rgb_to_hex(c)
        if not any(_color_similarity(c_hex, _rgb_to_hex(existing)) for existing in result):
            result.append(c_hex)
        if len(result) >= 8:
            break
    return result
