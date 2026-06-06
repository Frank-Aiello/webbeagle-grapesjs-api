#!/usr/bin/env python3
"""
Spec Merger — Merge known (user-provided) design tokens with detected (scraped/AI) ones.
Known ALWAYS wins on conflicts. Flags gaps and generates warnings.
"""

import json
from copy import deepcopy
from typing import Dict, List, Any, Optional


def merge_spec(known: dict, detected: dict, sources: dict = None) -> dict:
    """
    Merge known (user-provided) with detected (automated) design tokens.

    Args:
        known:     User-provided design constants — NEVER overridden.
        detected:  Scraped/vision-extracted design tokens.
        sources:   Optional map of where each value came from (for audit trail).

    Returns:
        {
            "spec": dict,           # Merged design spec
            "warnings": [str],      # Things that were detected differently than known
            "gaps": [str],          # Things neither known nor detected
            "confidence": {
                "fonts": "high"|"medium"|"low",
                "colors": "high"|"medium"|"low",
                "layout": "high"|"medium"|"low"
            }
        }
    """
    sources = sources or {}
    known = deepcopy(known or {})
    detected = deepcopy(detected or {})

    warnings = []
    gaps = []
    spec = {}

    # ── Fonts ──
    known_fonts = _normalize_fonts(known.get("fonts", {}))
    detected_fonts = _normalize_fonts(detected.get("dominant_fonts", []))

    merged_fonts = {}
    font_sources = {}

    # Start with detected
    if isinstance(detected_fonts, dict):
        for role, font in detected_fonts.items():
            merged_fonts[role] = font
            font_sources[role] = "detected"
    elif isinstance(detected_fonts, list) and detected_fonts:
        # Assign first font as headline, second as body
        merged_fonts["headline"] = detected_fonts[0]
        font_sources["headline"] = "detected"
        if len(detected_fonts) > 1:
            merged_fonts["body"] = detected_fonts[1]
            font_sources["body"] = "detected"

    # Known overrides
    for role, font in known_fonts.items():
        if font:
            old = merged_fonts.get(role)
            if old and old.lower() != font.lower():
                warnings.append(f"Font conflict on '{role}': detected '{old}' → using known '{font}'")
            merged_fonts[role] = font
            font_sources[role] = "known"

    spec["fonts"] = merged_fonts
    spec["_font_sources"] = font_sources

    if not merged_fonts:
        gaps.append("No fonts specified or detected")
    elif "headline" not in merged_fonts and "body" not in merged_fonts:
        gaps.append("Body font not specified or detected")

    # ── Colors ──
    known_colors = known.get("colors") or known.get("brand_colors") or []
    detected_colors = detected.get("color_palette") or []

    merged_colors = []

    # Known first (always)
    if isinstance(known_colors, str):
        known_colors = [known_colors]
    for c in known_colors:
        if c and c not in merged_colors:
            merged_colors.append(_normalize_color(c))

    # Detected fills gaps
    for c in detected_colors:
        if c and c not in merged_colors:
            merged_colors.append(_normalize_color(c))
        if len(merged_colors) >= 8:
            break

    spec["colors"] = merged_colors

    if not merged_colors:
        gaps.append("No colors specified or detected")
    elif len(merged_colors) < 3:
        gaps.append(f"Only {len(merged_colors)} color(s) available — recommended 3+")

    # ── Logo ──
    if known.get("logo_url"):
        spec["logo_url"] = known["logo_url"]
    elif detected.get("logo_url"):
        spec["logo_url"] = detected["logo_url"]
        warnings.append("Logo auto-detected — verify it's correct")
    else:
        gaps.append("No logo provided or detected")

    # ── Layout ──
    spec["layout"] = {
        "pattern": known.get("layout") or detected.get("layout_pattern", "unknown"),
        "hero": known.get("hero_structure") or detected.get("hero_structure") or {},
        "spacing": known.get("spacing") or detected.get("spacing_system", "unknown"),
        "texture": known.get("texture") or detected.get("texture", "none")
    }

    if spec["layout"]["pattern"] == "unknown":
        gaps.append("Layout pattern unknown — will use default")

    # ── Button Style ──
    known_button = known.get("button_style") or {}
    detected_button = detected.get("button_style") or {}
    spec["button"] = {**detected_button, **known_button}  # Known keys override

    # ── Seed HTML (known only — always trusted) ──
    if known.get("seed_html"):
        spec["seed_html"] = known["seed_html"]

    # ── Site type ──
    spec["site_type"] = known.get("site_type") or detected.get("site_type") or "landing_page"

    # ── Pages ──
    spec["pages"] = known.get("pages") or ["home"]

    # ── Additional known fields ──
    for key in ("voice", "tone", "industry", "target_audience", "cta_text", "brand_name"):
        if known.get(key):
            spec[key] = known[key]
        elif detected.get(key):
            spec[key] = detected[key]

    # ── Sources (audit trail) ──
    spec["_sources"] = sources

    # ── Confidence scoring ──
    confidence = {}
    # Fonts: high if headline is known, medium if detected, low if gap
    if known_fonts.get("headline"):
        confidence["fonts"] = "high"
    elif detected_fonts and (isinstance(detected_fonts, list) and len(detected_fonts) > 0):
        confidence["fonts"] = "medium"
    else:
        confidence["fonts"] = "low"

    # Colors: high if 3+ known, medium if any known or 3+ detected, low otherwise
    if len(known_colors) >= 3:
        confidence["colors"] = "high"
    elif len(known_colors) >= 1 or len(detected_colors) >= 3:
        confidence["colors"] = "medium"
    else:
        confidence["colors"] = "low"

    # Layout: high if explicitly set, medium if detected, low if unknown
    if known.get("layout"):
        confidence["layout"] = "high"
    elif detected.get("layout_pattern") and detected["layout_pattern"] != "unknown":
        confidence["layout"] = "medium"
    else:
        confidence["layout"] = "low"

    return {
        "spec": spec,
        "warnings": warnings,
        "gaps": gaps,
        "confidence": confidence
    }


def _normalize_fonts(fonts) -> dict:
    """Normalize font input to dict {role: font_name}."""
    if isinstance(fonts, dict):
        return {k: v for k, v in fonts.items() if v}
    if isinstance(fonts, list):
        result = {}
        for i, f in enumerate(fonts):
            if i == 0:
                result["headline"] = f
            elif i == 1:
                result["body"] = f
            elif i == 2:
                result["ui"] = f
        return result
    if isinstance(fonts, str) and fonts:
        return {"headline": fonts}
    return {}


def _normalize_color(c: str) -> str:
    """Ensure color is a proper hex value."""
    c = c.strip()
    if c.startswith("#"):
        return c.lower()
    # Try to convert rgb/rgba
    import re
    match = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)', c)
    if match:
        return "#{:02x}{:02x}{:02x}".format(*map(int, match.groups()))
    return c


def validate_spec(spec: dict) -> dict:
    """
    Check a merged spec for completeness.
    Returns {"valid": bool, "issues": [str], "ready_to_build": bool}
    """
    issues = []

    if not spec.get("fonts", {}).get("headline"):
        issues.append("Missing headline font")
    if not spec.get("colors") or len(spec["colors"]) < 2:
        issues.append("Need at least 2 colors")

    ready = len(issues) == 0

    return {
        "valid": len([i for i in issues if "Missing" in i or "Need" in i]) == 0,
        "issues": issues,
        "ready_to_build": ready
    }
