#!/usr/bin/env python3
"""
GrapesJS Projects API — WebBeagle Builder
Serves: project CRUD, multi-page management, preview rendering,
        AI section generation (Claude Sonnet 4), AI redesign (vision + KIE assets),
        KIE image & video generation, closed-loop hero refinement (vision + screenshot + compare + fix)
"""
import json
import os
import re
import uuid
import shutil
import time
import yaml
import requests
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from anthropic import Anthropic

# ── Config ──────────────────────────────────────────────────
PROJECTS_DIR = Path("/opt/data/grapesjs-projects")
COMPONENTS_DIR = Path("/opt/data/grapesjs-components")
BUILDER_DIR = Path("/opt/data/grapesjs-demo")
ASSETS_DIR = Path("/opt/data/grapesjs-assets")
DOCS_DIR = Path("/opt/data/grapesjs-docs")
CONFIG_PATH = Path("/opt/data/config.yaml")
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

KIE_API_KEY = os.environ.get("KIE_API_KEY", "")
KIE_BASE = "https://api.kie.ai/api/v1"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Emailit ──────────────────────────────────────────────────
EMAILIT_API_KEY = os.environ.get("EMAILIT_API_KEY", "")
EMAILIT_FROM = "WebBeagle Forms <alerts@webbeagle.com>"
EMAILIT_BASE = "https://api.emailit.com/v2"

# ── Rate limiting for form submissions ──────────────────────
_submission_tracker: dict = defaultdict(list)

def _check_rate_limit(ip: str, max_per_minute: int = 5) -> bool:
    now = time.time()
    window = [t for t in _submission_tracker[ip] if now - t < 60]
    _submission_tracker[ip] = window
    if len(window) >= max_per_minute:
        return False
    _submission_tracker[ip].append(now)
    return True

# ── Load API clients ─────────────────────────────────────────
def _get_openai_client():
    """Load OpenAI key from Hermes config.yaml"""
    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text())
        for p in config.get("custom_providers", []):
            if p.get("name") == "WebBeagle Hermes":
                return OpenAI(
                    api_key=p["api_key"],
                    base_url=p.get("base_url", "https://api.openai.com/v1")
                )
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

ai_client = _get_openai_client()
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ── App ──────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(BUILDER_DIR), static_url_path="")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    # Aggressive no-cache for JS files to prevent Cloudflare stale cache
    if response.content_type and 'javascript' in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# ══════════════════════════════════════════════════════════════
#  KIE AI — Image & Video Generation
# ══════════════════════════════════════════════════════════════

def _kie_submit(model: str, input_params: dict, callBackUrl: str = None) -> str:
    """Submit a KIE generation task. Returns taskId."""
    if not KIE_API_KEY:
        raise Exception("KIE_API_KEY not configured")
    payload = {"model": model, "input": input_params}
    if callBackUrl:
        payload["callBackUrl"] = callBackUrl
    r = requests.post(
        f"{KIE_BASE}/jobs/createTask",
        headers={"Authorization": f"Bearer {KIE_API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=30
    )
    data = r.json()
    if data.get("code") == 200:
        return data["data"]["taskId"]
    raise Exception(f"KIE submit error: {data}")

def _kie_poll(taskId: str, model: str, timeout: int = 300) -> dict:
    """Poll for task completion. Returns {resultImageUrl, resultVideoUrl, ...}.
    Uses the UNIFIED query endpoint for all Market models."""
    query_url = f"{KIE_BASE}/jobs/recordInfo?taskId={taskId}"
    
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(
            query_url,
            headers={"Authorization": f"Bearer {KIE_API_KEY}"},
            timeout=10
        )
        data = r.json()
        task_data = data.get("data", {})
        state = task_data.get("state", "")
        
        if state == "success":
            # Results are in resultJson — a JSON-encoded string with resultUrls array
            result_json_str = task_data.get("resultJson", "{}")
            try:
                result_data = json.loads(result_json_str)
            except json.JSONDecodeError:
                result_data = {}
            result_urls = result_data.get("resultUrls", [])
            # First URL is typically the image, second might be video
            return {
                "resultImageUrl": result_urls[0] if len(result_urls) > 0 else None,
                "resultVideoUrl": result_urls[1] if len(result_urls) > 1 else None,
            }
        elif state == "fail":
            fail_msg = task_data.get("failMsg", "Unknown error")
            raise Exception(f"KIE task failed: {fail_msg}")
        # waiting, queuing, generating — keep polling
        time.sleep(3)
    raise Exception(f"KIE task {taskId} timed out after {timeout}s")


@app.route("/api/kie/record/<taskId>", methods=["GET"])
def kie_record(taskId):
    """Public endpoint to poll a KIE task status."""
    try:
        import requests as req_lib
        query_url = f"{KIE_BASE}/jobs/recordInfo?taskId={taskId}"
        r = req_lib.get(query_url, headers={"Authorization": f"Bearer {KIE_API_KEY}"}, timeout=10)
        data = r.json()
        task_data = data.get("data", {})
        return jsonify({
            "state": task_data.get("state", ""),
            "resultJson": task_data.get("resultJson", "{}"),
            "failMsg": task_data.get("failMsg", "")
        })
    except Exception as e:
        return jsonify({"error": str(e), "state": "error"}), 500


def _download_asset(url: str, filename: str) -> str:
    """Download to assets dir, return public URL path /assets/<filename>."""
    r = requests.get(url, timeout=60)
    dest = ASSETS_DIR / filename
    dest.write_bytes(r.content)
    return f"/assets/{filename}"

def _generate_image(prompt: str, aspect_ratio: str = "16:9", model: str = "grok-imagine/text-to-image") -> str:
    """Generate image via KIE, download to assets, return public URL path.
    Returns empty string on timeout (graceful degradation)."""
    try:
        taskId = _kie_submit(model, {"prompt": prompt, "aspect_ratio": aspect_ratio})
        info = _kie_poll(taskId, model, timeout=300)  # 5 min — Grok Imagine can be slow
        img_url = info.get("resultImageUrl")
        if not img_url:
            return ""
        filename = f"kie-img-{taskId[:8]}.png"
        return _download_asset(img_url, filename)
    except Exception as e:
        print(f"[KIE image] {e}")
        return ""

def _generate_video(prompt: str, aspect_ratio: str = "16:9", duration: str = "6",
                    model: str = "grok-imagine/text-to-video", resolution: str = "480p") -> str:
    """Generate video via KIE, download to assets, return public URL path."""
    taskId = _kie_submit(model, {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "mode": "normal",
        "duration": duration,
        "resolution": resolution
    })
    info = _kie_poll(taskId, model, timeout=300)
    vid_url = info.get("resultVideoUrl")
    if not vid_url:
        raise Exception(f"No resultVideoUrl in KIE response: {info}")
    filename = f"kie-vid-{taskId[:8]}.mp4"
    return _download_asset(vid_url, filename)

@app.route("/api/kie/callback", methods=["POST"])
def kie_callback():
    """Receive async KIE task results."""
    data = request.get_json() or {}
    taskId = data.get("data", {}).get("taskId", "unknown")
    result_url = data.get("data", {}).get("info", {}).get("resultImageUrl") or \
                 data.get("data", {}).get("info", {}).get("resultVideoUrl")
    print(f"[KIE callback] taskId={taskId} result={'OK' if result_url else 'FAIL'}")
    return jsonify({"status": "received"})

# ══════════════════════════════════════════════════════════════
#  Subdomain Routing
# ══════════════════════════════════════════════════════════════

@app.before_request
def route_by_subdomain():
    """Route *.wstdwork.webbeagle.com and *.preview.webbeagle.com to project previews."""
    host = request.host
    for suffix in ['.wstdwork.webbeagle.com', '.preview.webbeagle.com']:
        if host.endswith(suffix):
            subdomain = host[:-len(suffix)]
            if subdomain == 'grapesjs':
                return
            project_path = PROJECTS_DIR / subdomain
            if project_path.exists() and project_path.is_dir():
                path = request.path.strip('/')
                # Raw HTML bypass: serve from raw-{page}.html directly, no GrapesJS processing
                if path.startswith('raw/'):
                    raw_page = path[4:] or 'home'
                    raw_path = PROJECTS_DIR / subdomain / f"raw-{raw_page}.html"
                    if raw_path.exists():
                        raw_html = raw_path.read_text()
                        theme_css = ""
                        meta_path = PROJECTS_DIR / subdomain / "meta.json"
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                            theme_css = meta.get("theme_css", "")
                        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{subdomain} - {raw_page}</title>
  <style>{theme_css}</style>
</head>
<body style="margin:0;padding:0;background:#050505;">
{raw_html}
</body>
</html>"""
                page_id = path if path else 'home'
                # Prefer raw HTML if available (preserves inline styles exactly)
                raw_path_check = PROJECTS_DIR / subdomain / f"raw-{page_id}.html"
                if raw_path_check.exists():
                    raw_html = raw_path_check.read_text()
                    theme_css = ""
                    meta_path = PROJECTS_DIR / subdomain / "meta.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        theme_css = meta.get("theme_css", "")
                    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{subdomain} - {page_id}</title>
  <style>{theme_css}</style>
</head>
<body style="margin:0;padding:0;background:#050505;">
{raw_html}
</body>
</html>"""
                html = _render_preview(subdomain, page_id)
                if html:
                    return html
                if page_id != 'home':
                    html = _render_preview(subdomain, 'home')
                    if html:
                        return html
            break

# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _project_path(project_id):
    return PROJECTS_DIR / project_id

def _page_path(project_id, page_id):
    return PROJECTS_DIR / project_id / "pages" / f"{page_id}.json"

def _project_meta_path(project_id):
    return PROJECTS_DIR / project_id / "meta.json"

def _load_meta(project_id):
    p = _project_meta_path(project_id)
    if p.exists():
        return json.loads(p.read_text())
    return None

def _save_meta(project_id, meta):
    p = _project_meta_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))

def _backup_page(pp):
    """Create a timestamped backup of a page file before it gets overwritten.
    Keeps the last 20 backups, oldest are auto-purged."""
    if not pp.exists():
        return
    backup_dir = pp.parent / ".backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{pp.stem}-{ts}.json"
    shutil.copy2(pp, backup_path)
    # Purge old backups: keep only the 20 most recent
    existing = sorted(backup_dir.glob(f"{pp.stem}-*.json"))
    for old in existing[:-20]:
        old.unlink()

def _sanitize_slug(s):
    return re.sub(r"[^a-z0-9-]", "", s.lower().replace(" ", "-"))[:40]


def _ensure_form_attrs(components, depth=0):
    """Walk component tree and ensure every form has data-webbeagle-form attribute.
    Returns True if any form was touched (for logging)."""
    touched = False
    if isinstance(components, list):
        for comp in components:
            if _ensure_form_attrs(comp, depth + 1):
                touched = True
    elif isinstance(components, dict):
        node_type = components.get("type", "")
        if node_type == "form":
            attrs = components.setdefault("attributes", {})
            if "data-webbeagle-form" not in attrs:
                attrs["data-webbeagle-form"] = "true"
                touched = True
        if "components" in components:
            if _ensure_form_attrs(components["components"], depth + 1):
                touched = True
    return touched

# ── GrapesJS JSON → HTML converter ─────────────────────────────
VOID_ELEMENTS = {"br", "hr", "img", "input", "meta", "link", "source", "area",
                 "base", "col", "embed", "track", "wbr"}

def _gjs_to_html(node):
    """Convert a GrapesJS JSON component tree node to HTML string.
    Handles arrays (siblings), objects (elements/text), and raw strings.
    """
    if isinstance(node, list):
        return "".join(_gjs_to_html(n) for n in node)
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return str(node)

    tag = node.get("tagName", "div")
    node_type = node.get("type", "")

    # Textnode: just return content
    if node_type == "textnode":
        return node.get("content", "")

    # Type-based tag fallback
    if not node.get("tagName"):
        if node_type == "link":
            tag = "a"
        elif node_type == "image":
            tag = "img"
        elif node_type == "text":
            tag = "span"
        elif node_type == "video":
            tag = "video"
        elif node_type == "map":
            tag = "iframe"
        elif node_type in ("form", "input", "textarea", "button", "select", "option", "label"):
            tag = node_type

    # Build attributes
    attrs = {}
    # Inline styles from the style object
    if "style" in node and isinstance(node["style"], dict):
        style_parts = []
        for k, v in node["style"].items():
            css_key = k.replace("_", "-")
            style_parts.append(f"{css_key}:{v}")
        if style_parts:
            attrs["style"] = ";".join(style_parts)
    # Explicit attributes
    if "attributes" in node and isinstance(node["attributes"], dict):
        for k, v in node["attributes"].items():
            # style in attributes overrides the style object
            attrs[k] = str(v)
    # Classes
    if "classes" in node and isinstance(node["classes"], list) and node["classes"]:
        attrs["class"] = " ".join(node["classes"])

    attr_str = ""
    if attrs:
        parts = []
        for k, v in attrs.items():
            parts.append(f'{k}="{v}"')
        if parts:
            attr_str = " " + " ".join(parts)

    # Void elements
    if tag.lower() in VOID_ELEMENTS:
        return f"<{tag}{attr_str}>"

    # Children
    children_html = ""
    if "components" in node:
        children_html = _gjs_to_html(node["components"])
    elif "content" in node:
        children_html = node["content"]

    # Auto-wrap .btn-primary text in <span> for rotating gradient border CSS
    node_classes = node.get("classes") or []
    if "btn-primary" in node_classes and children_html and not children_html.strip().startswith("<span"):
        children_html = f"<span>{children_html}</span>"

    return f"<{tag}{attr_str}>{children_html}</{tag}>"


def _gjs_styles_to_css(styles):
    """Convert GrapesJS style array to CSS string.

    Handles:
    - Bare class names (adds '.' prefix)
    - Pseudo-class/element via 'state' field (hover, :before, :after)
    - @keyframes via atRuleType='keyframes'
    - @media queries via atRuleType='media'
    - selectorsAdd as the primary selector for at-rules
    """
    if isinstance(styles, str):
        return styles
    if not isinstance(styles, list):
        return ""

    # Group @keyframes and @media blocks (they span multiple items)
    at_rule_blocks = {}  # key -> {"type": "keyframes"|"media", "text": "...", "items": [...]}
    for item in styles:
        at_type = item.get("atRuleType")
        if at_type in ("keyframes", "media"):
            key = f"{at_type}:{item.get('mediaText', '')}"
            if key not in at_rule_blocks:
                at_rule_blocks[key] = {"type": at_type, "text": item.get("mediaText", ""), "items": []}
            at_rule_blocks[key]["items"].append(item)

    css_parts = []

    for item in styles:
        if not isinstance(item, dict):
            continue

        # Skip items that belong to an at-rule block (handled separately)
        at_type = item.get("atRuleType")
        if at_type in ("keyframes", "media"):
            continue

        style = item.get("style", {})
        if not style:
            continue

        # Get selectors — use selectorsAdd as primary for at-rule items, selectors for regular
        selectors = item.get("selectors", [])
        selectors_add = item.get("selectorsAdd", "")

        if isinstance(selectors, str):
            selectors = [selectors]

        # Build full selector list (combine selectors + selectorsAdd)
        all_sels = list(selectors)
        if selectors_add and selectors_add not in all_sels:
            all_sels.append(selectors_add)

        # Handle pseudo-class/element state (hover, :before, :after, etc.)
        state = item.get("state", "").strip()
        if state:
            if not state.startswith(":"):
                state = ":" + state

        # Prefix class selectors with '.' (GrapesJS stores them without dots)
        full_selectors = []
        for s in all_sels:
            s = s.strip()
            if not s:
                continue
            # Keyframe percentages (0%, 100%, etc.) — pass through as-is
            if s.endswith("%") and s[:-1].replace(".", "").isdigit():
                full_selectors.append(s)
                continue
            # Already has CSS prefix — pass through
            if s[0] in ('.', '#', ':', '[', '@', '*'):
                full_selectors.append(s)
            else:
                full_selectors.append('.' + s)

        if not full_selectors:
            continue

        # Append state to EACH selector (so .btn-primary:hover, nav-link:hover both work)
        if state:
            full_selectors = [s + state for s in full_selectors]

        props = "; ".join(f"{k}: {v}" for k, v in style.items() if v and not k.startswith('__'))
        if props:
            css_parts.append(f"{', '.join(full_selectors)} {{ {props}; }}")

    # Render @keyframes blocks
    for key, block in at_rule_blocks.items():
        if block["type"] == "keyframes":
            frames = []
            for fitem in block["items"]:
                frame_sel = fitem.get("selectorsAdd", "").strip()
                frame_style = fitem.get("style", {})
                frame_props = "; ".join(f"{k}: {v}" for k, v in frame_style.items() if v and not k.startswith('__'))
                if frame_sel and frame_props:
                    frames.append(f"  {frame_sel} {{ {frame_props}; }}")
            if frames:
                css_parts.append(f"@keyframes {block['text']} {{\n" + "\n".join(frames) + "\n}")

    # Render @media blocks
    for key, block in at_rule_blocks.items():
        if block["type"] == "media":
            media_rules = []
            for mitem in block["items"]:
                m_sel = mitem.get("selectorsAdd", "").strip()
                m_style = mitem.get("style", {})
                if not m_sel or not m_style:
                    continue
                # Prefix class selectors
                if m_sel and not m_sel[0] in ('.', '#', ':', '[', '@', '*') and not (m_sel.endswith("%") and m_sel[:-1].replace(".","").isdigit()):
                    m_sel = '.' + m_sel
                m_props = "; ".join(f"{k}: {v}" for k, v in m_style.items() if v and not k.startswith('__'))
                if m_props:
                    media_rules.append(f"  {m_sel} {{ {m_props}; }}")
            if media_rules:
                css_parts.append(f"@media {block['text']} {{\n" + "\n".join(media_rules) + "\n}")

    return "\n".join(css_parts)


def _render_preview(project_id, page_id):
    """Build a standalone HTML page from saved project data."""
    meta = _load_meta(project_id)
    if not meta:
        return None
    page_file = _page_path(project_id, page_id)
    if not page_file.exists():
        return None
    page_data = json.loads(page_file.read_text())
    raw_components = page_data.get("components", "")
    # Handle both JSON objects (GrapesJS component tree) and HTML strings
    if isinstance(raw_components, (list, dict)):
        components_html = _gjs_to_html(raw_components)
    else:
        components_html = str(raw_components)

    # Safety net: ensure every <form> has data-webbeagle-form attribute
    # (GrapesJS can drop empty attributes; this guarantees forms work on publish)
    components_html = re.sub(
        r'<form\b((?!data-webbeagle-form)[^>]*)>',
        r'<form data-webbeagle-form="true"\1>',
        components_html
    )
    styles_css = _gjs_styles_to_css(page_data.get("styles", ""))
    css_raw = meta.get("theme_css", "")

    # Collect CSS from all components (so library components render on published pages)
    components_css = ""
    if COMPONENTS_DIR.exists():
        for cat_dir in COMPONENTS_DIR.iterdir():
            if cat_dir.is_dir():
                for comp_file in cat_dir.glob("*.json"):
                    try:
                        comp = json.loads(comp_file.read_text())
                        if comp.get("css"):
                            components_css += f"/* {comp.get('name', comp_file.stem)} */\n{comp['css']}\n"
                    except Exception:
                        pass

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meta.get('name', 'Preview')} — {page_id}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@700;800&display=swap" rel="stylesheet">
  <style>{css_raw}</style>
  <style>{styles_css}</style>
  <style>{components_css}</style>
  <style>
/* ── WebBeagle Animation System v2 (published runtime) ── */
@keyframes wb-fade-up {{
  0%   {{ opacity: var(--wb-anim-opacity-start, 0); transform: translateY(var(--wb-anim-offset, 40px)); }}
  100% {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes wb-fade-in {{
  0%   {{ opacity: var(--wb-anim-opacity-start, 0); }}
  100% {{ opacity: 1; }}
}}
@keyframes wb-slide-left {{
  0%   {{ opacity: var(--wb-anim-opacity-start, 0); transform: translateX(calc(-1 * var(--wb-anim-offset, 40px))); }}
  100% {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes wb-slide-right {{
  0%   {{ opacity: var(--wb-anim-opacity-start, 0); transform: translateX(var(--wb-anim-offset, 40px)); }}
  100% {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes wb-zoom-in {{
  0%   {{ opacity: var(--wb-anim-opacity-start, 0); transform: scale(var(--wb-anim-scale-start, 0.92)); }}
  100% {{ opacity: 1; transform: scale(1); }}
}}
.wb-animated {{
  animation-fill-mode: both;
  animation-timing-function: var(--wb-anim-easing, cubic-bezier(0.16, 1, 0.3, 1));
  animation-duration: var(--wb-anim-duration, 800ms);
  animation-delay: var(--wb-anim-delay, 0ms);
  animation-play-state: paused;
  will-change: opacity, transform;
}}
.wb-animated[data-animate="fade-up"]     {{ animation-name: wb-fade-up; }}
.wb-animated[data-animate="fade-in"]     {{ animation-name: wb-fade-in; }}
.wb-animated[data-animate="slide-left"]  {{ animation-name: wb-slide-left; }}
.wb-animated[data-animate="slide-right"] {{ animation-name: wb-slide-right; }}
.wb-animated[data-animate="zoom-in"]     {{ animation-name: wb-zoom-in; }}
[data-trigger="load"] .wb-animated,
.wb-animated[style*="running"] {{ animation-play-state: running; }}
/* ── Form success message ── */
.wb-form-success {{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
}}
/* ── Form field defaults ── */
form[data-webbeagle-form] {{ max-width: 560px; margin: 0 auto; }}
form[data-webbeagle-form] > div {{ display: flex; flex-direction: column; gap: 16px; }}
form[data-webbeagle-form] input[type="text"],
form[data-webbeagle-form] input[type="email"],
form[data-webbeagle-form] textarea {{
  width: 100%; padding: 14px 18px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  color: #fff; font-size: 15px;
  font-family: var(--font-body, 'Inter', sans-serif);
  outline: none; transition: border-color .3s, box-shadow .3s;
}}
form[data-webbeagle-form] input:focus,
form[data-webbeagle-form] textarea:focus {{
  border-color: var(--accent, #A383E6);
  box-shadow: 0 0 0 3px rgba(163,131,230,.15);
}}
form[data-webbeagle-form] textarea {{ min-height: 120px; resize: vertical; }}
form[data-webbeagle-form] .btn-primary {{ align-self: flex-start; min-width: 160px; }}
  </style>
</head>
<body>
{components_html}
<script>
(function() {{
  'use strict';
  var scrollObserver = null;

  function processContainers() {{
    var containers = document.querySelectorAll('[data-animate]');
    containers.forEach(function(section) {{
      if (section._wbAnimated) return;
      var animType = section.getAttribute('data-animate');
      var stagger  = parseInt(section.getAttribute('data-stagger')) || 120;
      var duration = parseInt(section.getAttribute('data-duration')) || 800;
      var easing   = section.getAttribute('data-easing') || 'cubic-bezier(0.16,1,0.3,1)';
      var startOp  = section.getAttribute('data-initial-opacity') || '0';
      var trigger  = section.getAttribute('data-trigger') || 'scroll';
      var offset   = section.getAttribute('data-offset') || '40';
      var items = section.querySelectorAll('[data-animate-item]');
      if (items.length === 0) {{
        var grid = section.querySelector('.testimonials-grid, .services-grid, .steps-grid, [class*="-grid"]');
        if (grid) items = grid.children;
      }}
      if (items.length === 0) {{
        items = [];
        Array.from(section.children).forEach(function(c) {{
          if (c.nodeType === 1 && !c.classList.contains('section-header')) items.push(c);
        }});
      }}
      if (items.length === 0) {{ items = [section]; }}
      Array.from(items).forEach(function(item, i) {{
        if (item._wbAnimated) return;
        item.classList.add('wb-animated');
        item.setAttribute('data-animate', animType);
        item.style.setProperty('--wb-anim-delay', (i * stagger) + 'ms');
        item.style.setProperty('--wb-anim-duration', duration + 'ms');
        item.style.setProperty('--wb-anim-easing', easing);
        item.style.setProperty('--wb-anim-opacity-start', startOp);
        item.style.setProperty('--wb-anim-offset', offset + 'px');
        item._wbAnimated = true;
      }});
      section._wbAnimated = true;
      if (trigger === 'load') {{
        Array.from(items).forEach(function(item) {{ item.style.animationPlayState = 'running'; }});
      }}
    }});
    setupScrollObserver();
  }}

  function setupScrollObserver() {{
    if (!window.IntersectionObserver) return;
    if (scrollObserver) {{ try {{ scrollObserver.disconnect(); }} catch(e) {{}} }}
    var els = document.querySelectorAll('.wb-animated[data-animate]');
    var targets = [];
    els.forEach(function(el) {{
      if (el.style.animationPlayState === 'running') return;
      var s = el.closest('[data-trigger]');
      if (s && s.getAttribute('data-trigger') === 'load') return;
      targets.push(el);
    }});
    if (targets.length === 0) return;
    scrollObserver = new IntersectionObserver(function(entries) {{
      entries.forEach(function(entry) {{
        if (entry.isIntersecting) {{ entry.target.style.animationPlayState = 'running'; scrollObserver.unobserve(entry.target); }}
      }});
    }}, {{ threshold: 0.1 }});
    targets.forEach(function(el) {{ scrollObserver.observe(el); }});
  }}

  // Legacy .reveal support
  (function() {{
    var reveals = document.querySelectorAll('.reveal:not(.visible)');
    if (reveals.length === 0) return;
    var obs = new IntersectionObserver(function(entries) {{
      entries.forEach(function(entry) {{
        if (entry.isIntersecting) {{ entry.target.classList.add('visible'); obs.unobserve(entry.target); }}
      }});
    }}, {{ threshold: 0.15 }});
    reveals.forEach(function(el) {{ obs.observe(el); }});
  }})();

  // Hero scroll parallax
  var hero = document.querySelector('.hero');
  var heroWrap = document.querySelector('.hero-wrap');
  if (heroWrap && hero) {{
    window.addEventListener('scroll', function() {{
      var rect = heroWrap.getBoundingClientRect();
      var progress = 1 - Math.max(0, Math.min(1, -rect.top / (rect.height - window.innerHeight)));
      hero.style.opacity = 0.4 + (progress * 0.6);
    }}, {{ passive: true }});
  }}

  // Smooth anchor links
  document.querySelectorAll('a[href^="#"]').forEach(function(link) {{
    link.addEventListener('click', function(e) {{
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {{ e.preventDefault(); target.scrollIntoView({{ behavior: 'smooth' }}); }}
    }});
  }});

  // Glass card hover
  document.querySelectorAll('.glass').forEach(function(card) {{
    card.addEventListener('mousemove', function(e) {{
      var rect = card.getBoundingClientRect();
      card.style.setProperty('--mx', ((e.clientX - rect.left) / rect.width) * 100 + '%');
      card.style.setProperty('--my', ((e.clientY - rect.top) / rect.height) * 100 + '%');
    }});
  }});

  processContainers();
}})();
// ── WebBeagle Form Handler ──
(function() {{
  var forms = document.querySelectorAll('form[data-webbeagle-form]');
  if (forms.length === 0) return;

  forms.forEach(function(form) {{
    // Inject honeypot
    var hp = document.createElement('div');
    hp.style.cssText = 'position:absolute;left:-9999px;top:-9999px;opacity:0;height:0;width:0;overflow:hidden;';
    hp.innerHTML = '<input type="text" name="_wb_hp" tabindex="-1" autocomplete="off">';
    form.appendChild(hp);

    form.addEventListener('submit', function(e) {{
      e.preventDefault();
      var hpField = form.querySelector('input[name="_wb_hp"]');
      if (hpField && hpField.value) return;

      var btn = form.querySelector('button[type="submit"], input[type="submit"]');
      var btnOrig = btn ? btn.innerHTML : '';
      if (btn) {{ btn.disabled = true; btn.innerHTML = 'Sending...'; }}

      // Auto-assign name attributes to nameless fields
      // (FormData only collects fields with a name — GrapesJS often omits them)
      var fields = form.querySelectorAll('input:not([type=\"submit\"]):not([type=\"hidden\"]), textarea, select');
      var usedNames = {{}};
      fields.forEach(function(field) {{
        if (field.getAttribute('name')) {{
          usedNames[field.getAttribute('name')] = true;
        }}
      }});
      fields.forEach(function(field) {{
        if (!field.getAttribute('name')) {{
          var placeholder = (field.getAttribute('placeholder') || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
          var name = placeholder || field.type || 'field';
          if (usedNames[name]) {{
            var i = 2;
            while (usedNames[name + '_' + i]) i++;
            name = name + '_' + i;
          }}
          usedNames[name] = true;
          field.setAttribute('name', name);
        }}
      }});

      var fd = new FormData(form);
      var data = {{}};
      fd.forEach(function(v, k) {{ if (k !== '_wb_hp') data[k] = v; }});
      data['_project'] = '{project_id}';

      fetch('https://builder.webbeagle.com/api/submit-form', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(data)
      }})
      .then(function(r) {{ return r.json(); }})
      .then(function(result) {{
        if (result.status === 'ok') {{
          var msg = form.getAttribute('data-form-thankyou') || 'Thank you! We\\'ll be in touch soon.';
          var thankYou = document.createElement('div');
          thankYou.className = 'wb-form-success';
          thankYou.style.cssText = 'text-align:center;padding:32px 20px;';
          thankYou.innerHTML = '<div style="font-size:48px;margin-bottom:12px;">&#10003;</div><p style="font-family:var(--font-heading,inherit);font-size:1.1rem;color:#fff;">' + msg + '</p>';
          form.parentNode.replaceChild(thankYou, form);
        }} else {{
          throw new Error(result.error || 'Submission failed');
        }}
      }})
      .catch(function(err) {{
        if (btn) {{ btn.disabled = false; btn.innerHTML = btnOrig; }}
        alert('Something went wrong. Please try again.');
      }});
    }});
  }});
}})();
</script>
<script src="https://builder.webbeagle.com/assets/wp-interactions-full.js"></script>
<script>
(function() {{
  'use strict';
  function setupConfetti() {{
    InteractRunner.configure([{{
      key: 'cta_confetti',
      title: 'CTA Confetti',
      type: 'click',
      target: {{ type: 'selector', value: '.cta-section .btn-white' }},
      timelines: [{{
        loop: false,
        onceOnly: false,
        alternate: false,
        reset: false,
        reverse: false,
        actions: [{{
          type: 'confetti',
          key: 'confetti_1',
          target: {{ type: 'trigger' }},
          timing: {{
            isStartingState: false,
            start: 0,
            duration: 0.5,
            easing: 'outCirc'
          }},
          value: {{}}
        }}]
      }}],
      options: []
    }}]);
    InteractRunner.init();
  }}
  if (typeof InteractRunner !== 'undefined') {{
    setupConfetti();
  }} else {{
    window.addEventListener('load', function() {{
      if (typeof InteractRunner !== 'undefined') setupConfetti();
    }});
  }}
}})();
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════
#  Routes — Builder & Static
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(BUILDER_DIR), "index.html")

@app.route("/preview.html")
def standalone_preview():
    return send_from_directory(str(BUILDER_DIR), "preview.html")

@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(str(ASSETS_DIR), filename)

@app.route("/docs/")
@app.route("/docs/<path:filename>")
def serve_docs(filename="index.html"):
    return send_from_directory(str(DOCS_DIR), filename)


@app.route("/api/assets/upload", methods=["POST"])
def upload_asset():
    """Upload an image file. Returns the public URL."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "png"
    allowed = {"png", "jpg", "jpeg", "webp", "gif", "svg"}
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    name = f"{uuid.uuid4().hex}.{ext}"
    f.save(str(ASSETS_DIR / name))

    host = request.host
    forwarded = request.headers.get("X-Forwarded-Host", "")
    if forwarded:
        host = forwarded

    return jsonify({
        "url": f"https://{host}/assets/{name}",
        "filename": name,
        "size": os.path.getsize(str(ASSETS_DIR / name))
    }), 201

@app.route("/api/assets/upload-zip", methods=["POST"])
def upload_assets_zip():
    """Upload a ZIP file of asset images. Extracts them to the assets directory.
    These assets are then used by the pixel-perfect layout pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not f.filename.lower().endswith(".zip"):
        return jsonify({"error": "File must be a .zip"}), 400

    import zipfile, io
    extracted = []
    try:
        with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
            for name in zf.namelist():
                # Skip directories and hidden files
                if name.endswith("/") or name.startswith(".") or name.startswith("__"):
                    continue
                # Only extract image files
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext not in {"png", "jpg", "jpeg", "webp", "gif", "svg"}:
                    continue
                # Extract with a unique prefix to avoid collisions
                safe_name = f"{uuid.uuid4().hex[:8]}-{Path(name).name}"
                zf.extract(name, str(ASSETS_DIR))
                extracted_path = ASSETS_DIR / name
                if extracted_path.exists():
                    # Rename to safe name
                    new_path = ASSETS_DIR / safe_name
                    extracted_path.rename(new_path)
                    extracted.append({"original": name, "saved": safe_name, "url": f"/assets/{safe_name}"})
    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400

    host = request.host
    forwarded = request.headers.get("X-Forwarded-Host", "")
    if forwarded:
        host = forwarded

    return jsonify({
        "extracted": len(extracted),
        "files": extracted,
        "base_url": f"https://{host}/assets/"
    }), 201


# ══════════════════════════════════════════════════════════════
#  API — Health
# ══════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "claude": anthropic_client is not None,
        "kie": bool(KIE_API_KEY)
    })

# ══════════════════════════════════════════════════════════════
#  API — Projects CRUD
# ══════════════════════════════════════════════════════════════

@app.route("/api/projects", methods=["GET"])
def list_projects():
    projects = []
    for d in sorted(PROJECTS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            meta = _load_meta(d.name)
            if meta:
                projects.append({
                    "id": d.name,
                    "name": meta.get("name", d.name),
                    "created": meta.get("created", ""),
                    "updated": meta.get("updated", ""),
                    "pages": meta.get("pages", ["home"]),
                    "preview_url": f"/preview/{d.name}/home/"
                })
    return jsonify({"projects": projects})

@app.route("/api/projects", methods=["POST"])
def create_project():
    data = request.get_json() or {}
    name = data.get("name", "Untitled Project")
    slug = data.get("slug") or _sanitize_slug(name) or str(uuid.uuid4())[:8]
    project_id = slug
    counter = 1
    base_id = project_id
    while _project_path(project_id).exists():
        project_id = f"{base_id}-{counter}"
        counter += 1
    now = datetime.utcnow().isoformat()
    meta = {
        "name": name,
        "created": now,
        "updated": now,
        "pages": ["home"],
        "theme_css": data.get("theme_css", ""),
        "notification_email": data.get("notification_email", ""),
        "recaptcha_site_key": data.get("recaptcha_site_key", ""),
        "recaptcha_secret_key": data.get("recaptcha_secret_key", ""),
    }
    _save_meta(project_id, meta)
    page_dir = _page_path(project_id, "home").parent
    page_dir.mkdir(parents=True, exist_ok=True)
    _page_path(project_id, "home").write_text(json.dumps({
        "components": data.get("components", ""),
        "styles": data.get("styles", ""),
    }))
    return jsonify({"id": project_id, "name": name, "pages": ["home"]}), 201

@app.route("/api/projects/<project_id>", methods=["GET"])
def get_project(project_id):
    meta = _load_meta(project_id)
    if not meta:
        return jsonify({"error": "Project not found"}), 404
    pages_data = {}
    pages_dir = _project_path(project_id) / "pages"
    if pages_dir.exists():
        for pf in pages_dir.glob("*.json"):
            page_id = pf.stem
            pages_data[page_id] = json.loads(pf.read_text())
    return jsonify({
        "id": project_id,
        "name": meta.get("name"),
        "created": meta.get("created"),
        "updated": meta.get("updated"),
        "pages": meta.get("pages", list(pages_data.keys())),
        "pages_data": pages_data,
        "theme_css": meta.get("theme_css", ""),
        "notification_email": meta.get("notification_email", ""),
        "recaptcha_site_key": meta.get("recaptcha_site_key", ""),
        "recaptcha_secret_key": meta.get("recaptcha_secret_key", ""),
    })

@app.route("/api/projects/<project_id>", methods=["PUT"])
def save_project(project_id):
    meta = _load_meta(project_id)
    if not meta:
        return jsonify({"error": "Project not found"}), 404
    data = request.get_json() or {}
    now = datetime.utcnow().isoformat()
    if "name" in data:
        meta["name"] = data["name"]
    if "theme_css" in data:
        meta["theme_css"] = data["theme_css"]
    if "notification_email" in data:
        meta["notification_email"] = data["notification_email"]
    if "recaptcha_site_key" in data:
        meta["recaptcha_site_key"] = data["recaptcha_site_key"]
    if "recaptcha_secret_key" in data:
        meta["recaptcha_secret_key"] = data["recaptcha_secret_key"]
    meta["updated"] = now
    if "pages_data" in data:
        for page_id, page_data in data["pages_data"].items():
            # Auto-inject data-webbeagle-form on any form component
            components = page_data.get("components")
            if components:
                _ensure_form_attrs(components)
            pp = _page_path(project_id, page_id)
            pp.parent.mkdir(parents=True, exist_ok=True)
            _backup_page(pp)
            pp.write_text(json.dumps(page_data, indent=2))
        meta["pages"] = list(data["pages_data"].keys())
    elif "components" in data or "styles" in data:
        page_id = data.get("page_id", "home")
        components = data.get("components", "")
        if components:
            _ensure_form_attrs(components)
        pp = _page_path(project_id, page_id)
        pp.parent.mkdir(parents=True, exist_ok=True)
        _backup_page(pp)
        pp.write_text(json.dumps({
            "components": data.get("components", ""),
            "styles": data.get("styles", ""),
        }, indent=2))
        if page_id not in meta.get("pages", []):
            meta.setdefault("pages", []).append(page_id)
    _save_meta(project_id, meta)
    return jsonify({"status": "saved", "updated": now})

@app.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    pp = _project_path(project_id)
    if pp.exists():
        shutil.rmtree(pp)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Project not found"}), 404

@app.route("/api/projects/<project_id>/settings", methods=["GET"])
def get_project_settings(project_id):
    meta = _load_meta(project_id)
    if not meta:
        return jsonify({"error": "Project not found"}), 404
    return jsonify({
        "notification_email": meta.get("notification_email", ""),
        "recaptcha_site_key": meta.get("recaptcha_site_key", ""),
        "recaptcha_secret_key": meta.get("recaptcha_secret_key", ""),
    })

@app.route("/api/projects/<project_id>/settings", methods=["PUT"])
def update_project_settings(project_id):
    meta = _load_meta(project_id)
    if not meta:
        return jsonify({"error": "Project not found"}), 404
    data = request.get_json() or {}
    for key in ["notification_email", "recaptcha_site_key", "recaptcha_secret_key"]:
        if key in data:
            meta[key] = data[key]
    meta["updated"] = datetime.utcnow().isoformat()
    _save_meta(project_id, meta)
    return jsonify({"status": "saved"})

# ══════════════════════════════════════════════════════════════
#  Form Submissions → Emailit
# ══════════════════════════════════════════════════════════════

@app.route("/api/submit-form", methods=["POST"])
def submit_form():
    data = request.get_json() or {}

    # Resolve project: try Host header first, then _project field
    host = request.headers.get("Host", "")
    slug = data.pop("_project", None)

    if not slug:
        if ".preview.webbeagle.com" in host:
            slug = host.split(".")[0]
        else:
            for d in PROJECTS_DIR.iterdir():
                if d.is_dir():
                    m = _load_meta(d.name)
                    if m and m.get("live_domain") == host:
                        slug = d.name
                        break
    if not slug:
        return jsonify({"status": "error", "error": "Unknown project"}), 400

    meta = _load_meta(slug)
    if not meta:
        return jsonify({"status": "error", "error": "Project not found"}), 404

    # Honeypot check — silently accept bot submissions
    if data.get("_wb_hp"):
        return jsonify({"status": "ok"})

    # Rate limit
    if not _check_rate_limit(request.remote_addr):
        return jsonify({"status": "error", "error": "Too many submissions. Please wait."}), 429

    notification_email = meta.get("notification_email", "")
    if not notification_email:
        return jsonify({"status": "error", "error": "No notification email configured"}), 400

    # Optional reCAPTCHA
    recaptcha_secret = meta.get("recaptcha_secret_key", "")
    recaptcha_token = data.pop("_recaptcha_token", None)
    if recaptcha_secret and recaptcha_token:
        rr = requests.post("https://www.google.com/recaptcha/api/siteverify", data={
            "secret": recaptcha_secret,
            "response": recaptcha_token,
        }, timeout=10)
        if not rr.json().get("success"):
            return jsonify({"status": "error", "error": "reCAPTCHA validation failed"}), 400

    # Build email HTML
    fields_html = ""
    for key, value in data.items():
        if key.startswith("_"):
            continue
        fields_html += f'<tr><td style="padding:8px 12px;border-bottom:1px solid #eee;font-weight:600;color:#333;">{key}</td><td style="padding:8px 12px;border-bottom:1px solid #eee;">{value}</td></tr>'

    email_html = f"""<!DOCTYPE html>
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;">
  <h2 style="color:#A383E6;">New Form Submission</h2>
  <p style="color:#666;">Project: <strong>{meta.get('name', slug)}</strong></p>
  <table style="width:100%;border-collapse:collapse;margin:16px 0;">
    {fields_html}
  </table>
  <p style="color:#999;font-size:12px;">Sent via WebBeagle Forms</p>
</body></html>"""

    # Send via Emailit
    if not EMAILIT_API_KEY:
        app.logger.warning("EMAILIT_API_KEY not configured — email not sent")
    else:
        try:
            r = requests.post(
                f"{EMAILIT_BASE}/emails",
                headers={
                    "Authorization": f"Bearer {EMAILIT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": EMAILIT_FROM,
                    "to": [notification_email],
                    "subject": f"New form submission — {meta.get('name', slug)}",
                    "html": email_html,
                },
                timeout=15,
            )
            if r.status_code not in (200, 201, 202):
                app.logger.error(f"Emailit error: {r.status_code} {r.text}")
                return jsonify({"status": "error", "error": "Failed to send notification"}), 500
        except Exception as e:
            app.logger.error(f"Emailit exception: {e}")
            return jsonify({"status": "error", "error": "Failed to send notification"}), 500

    return jsonify({"status": "ok"})

@app.route("/api/projects/<project_id>/pages", methods=["POST"])
def add_page(project_id):
    meta = _load_meta(project_id)
    if not meta:
        return jsonify({"error": "Project not found"}), 404
    data = request.get_json() or {}
    page_id = _sanitize_slug(data.get("name", "new-page")) or str(uuid.uuid4())[:8]
    if page_id in meta.get("pages", []):
        return jsonify({"error": "Page already exists"}), 409
    pp = _page_path(project_id, page_id)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps({"components": "", "styles": ""}))
    meta.setdefault("pages", []).append(page_id)
    meta["updated"] = datetime.utcnow().isoformat()
    _save_meta(project_id, meta)
    return jsonify({"page_id": page_id, "pages": meta["pages"]}), 201

@app.route("/api/projects/<project_id>/pages/<page_id>", methods=["DELETE"])
def delete_page(project_id, page_id):
    meta = _load_meta(project_id)
    if not meta:
        return jsonify({"error": "Project not found"}), 404
    if page_id == "home":
        return jsonify({"error": "Cannot delete home page"}), 400
    pp = _page_path(project_id, page_id)
    if pp.exists():
        pp.unlink()
    if page_id in meta.get("pages", []):
        meta["pages"].remove(page_id)
    meta["updated"] = datetime.utcnow().isoformat()
    _save_meta(project_id, meta)
    return jsonify({"status": "deleted", "pages": meta["pages"]})

@app.route("/preview/<project_id>/<page_id>/")
@app.route("/preview/<project_id>/<page_id>")
def preview(project_id, page_id):
    html = _render_preview(project_id, page_id)
    if html is None:
        return jsonify({"error": "Project or page not found"}), 404
    return html

@app.route("/raw/<project_id>/<page_id>/")
@app.route("/raw/<project_id>/<page_id>")
def serve_raw(project_id, page_id):
    """Serve a page from raw HTML file, bypassing GrapesJS entirely."""
    raw_path = PROJECTS_DIR / project_id / f"raw-{page_id}.html"
    if not raw_path.exists():
        return "Raw page not found", 404
    html = raw_path.read_text()
    theme_css = ""
    meta_path = PROJECTS_DIR / project_id / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        theme_css = meta.get("theme_css", "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{project_id} - {page_id}</title>
  <style>{theme_css}</style>
</head>
<body style="margin:0;padding:0;background:#050505;">
{html}
</body>
</html>"""

# ══════════════════════════════════════════════════════════════
#  AI — Section Generation (Claude Sonnet 4)
# ══════════════════════════════════════════════════════════════

def _claude_generate(system_msg: str, user_msg, max_tokens: int = 4000) -> str:
    """Generate HTML using Claude Sonnet 4 — best for web design.
    user_msg can be a string or a list of content blocks (for images)."""
    if not anthropic_client:
        # Fallback to GPT-5.5 if Anthropic not configured
        if isinstance(user_msg, list):
            # Convert image blocks for OpenAI format
            openai_content = []
            for block in user_msg:
                if block.get("type") == "image":
                    openai_content.append({"type": "image_url", "image_url": block["source"]})
                else:
                    openai_content.append(block)
            user_msg = openai_content
        response = ai_client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    # Claude format
    if isinstance(user_msg, str):
        user_content = user_msg
    else:
        user_content = []
        for block in user_msg:
            if block.get("type") == "image":
                src = block["source"]
                if src.get("type") == "base64":
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": src.get("media_type", "image/png"),
                            "data": src["data"]
                        }
                    })
                else:
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": src.get("url", "")
                        }
                    })
            else:
                user_content.append(block)

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        temperature=0.7,
        system=system_msg,
        messages=[{"role": "user", "content": user_content}]
    )
    return response.content[0].text.strip()

def _extract_layout_constraints(existing_html: str, theme_css: str) -> str:
    """Parse theme CSS to find layout values for classes used in existing_html."""
    if not theme_css or not existing_html:
        return ""
    # Find all classes in the existing HTML
    classes = set(re.findall(r'class="([^"]*)"', existing_html))
    all_classes = set()
    for c in classes:
        all_classes.update(c.split())
    # Extract relevant CSS rules
    constraints = []
    css_lower = theme_css.lower()
    for cls in sorted(all_classes):
        # Look for rules targeting this class
        pattern = re.compile(rf'\.{re.escape(cls)}\s*\{{[^}}]+\}}', re.IGNORECASE)
        for match in pattern.finditer(theme_css):
            rule = match.group()
            props = {}
            for prop_match in re.finditer(r'(text-align|font-size|font-weight|padding|margin)\s*:\s*([^;]+);?', rule, re.IGNORECASE):
                props[prop_match.group(1).lower()] = prop_match.group(2).strip()
            if props:
                parts = [f".{cls} -> {k}: {v}" for k, v in sorted(props.items())]
                constraints.extend(parts)
    if constraints:
        return "## Layout Constraints (MUST PRESERVE)\n" + "\n".join(constraints)
    return ""

@app.route("/api/ai/generate", methods=["POST"])
def ai_generate():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    design_md = data.get("design_md", "")
    existing_html = data.get("existing_html", "")
    context_html = data.get("context_html", "")
    theme_css = data.get("theme_css", "")
    mode = data.get("mode", "generate")
    vision_url = data.get("vision_url", "")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    if mode == "edit":
        system_msg = """You are an expert web designer and frontend developer. You MODIFY existing HTML sections based on user requests, applying the full design system.

## Your Task
The user will provide an EXISTING HTML section and request changes. When asked to "match the design" or "apply the theme," REPLACE the entire look of the section with the design system — do NOT preserve old colors, backgrounds, or typography. The design system is authoritative.

## Output Rules
- Return ONLY the modified HTML (no explanations, no markdown wrappers)
- Completely replace old styling with design system classes (e.g., .glass, .gold-gloss, .wb-section)
- If the user says "match our design" or "apply the theme," rebuild the entire section using the design system — don't just tweak individual elements
- When applying glassmorphism, use the .glass class on panels/cards
- When applying gold text, use .gold-gloss on headlines
- Use the exact button classes from the design system (.btn-primary, .btn-secondary)
- CRITICAL: .btn-primary requires a <span> wrapper: <a class="btn-primary"><span>Text</span></a> — the span creates the solid button surface over a rotating gradient border. Without the span, the button is invisible.
- BUTTON TYPE MAPPING — when the user says "primary button" or "gradient button," use .btn-primary with <span>. When the user says "secondary button" or "outline button," use .btn-secondary (simple, no span). When in doubt, use .btn-primary for hero/CTA sections and .btn-secondary for less prominent actions.
- Keep all text content the same unless the user explicitly asks to change it
- Return the COMPLETE section, not a diff or partial update
- If adding items to a list/grid, replicate the exact pattern of existing items
"""
    else:
        system_msg = """You are an expert web designer and frontend developer. You generate production-quality HTML/CSS sections that match a given design system.

## Your Task
Generate a single self-contained HTML section based on the user's request. If the user provides a screenshot or design image, reproduce that design as faithfully as possible using the provided theme CSS classes.

## Output Rules
- Return ONLY the HTML (no explanations, no markdown wrappers)
- All styles must be inline or reference classes from the provided theme CSS
- Keep the HTML flat — no nested <style> or <script> tags
- Use the EXACT class names from the design system (e.g., .glass, .gold-gloss, .wb-section)
- CRITICAL: .btn-primary requires a <span> wrapper: <a class="btn-primary"><span>Text</span></a> — the span creates the solid button surface over a rotating gradient border. Without the span, the button is invisible.
- BUTTON TYPE MAPPING — when the user says "primary button" or "gradient button," use .btn-primary with <span>. When the user says "secondary button" or "outline button," use .btn-secondary (simple, no span). When in doubt, use .btn-primary for hero/CTA sections and .btn-secondary for less prominent actions.
- Make text content realistic and professional
- Ensure the section is complete and renders standalone
- If recreating from a screenshot, match layout, hierarchy, colors, and spacing as closely as possible
"""

    if design_md:
        system_msg += f"\n\n## Design System (DESIGN.md)\n{design_md}"

    # Extract and inject layout constraints for edit mode
    if mode == "edit" and existing_html and theme_css:
        constraints = _extract_layout_constraints(existing_html, theme_css)
        if constraints:
            system_msg += f"\n\n{constraints}"

    if existing_html:
        system_msg += f"\n\n## Existing HTML to Modify\n```html\n{existing_html[:4000]}\n```"

    if context_html and mode != "edit":
        system_msg += f"\n\n## Existing Page Context\nThe following HTML is already on the page. Make sure your section integrates well with it:\n```html\n{context_html[:3000]}\n```"

    try:
        if vision_url:
            # TWO-STAGE APPROACH: extract spec first, then build exact HTML
            # This prevents the garbage single-pass output

            # Stage 1: Extract rigid layout spec from the image
            vision_prompt = """You are a forensic UI analyst. Look at this design image and describe the EXACT pixel-level layout as a rigid specification. Be precise about every color, font size, spacing, and text.

Output ONLY this JSON structure — no markdown, no explanations, no backticks:
{
  "layout": {"columns": 1|2, "text_side": "left|right|center", "image_side": "left|right|none", "vertical_center": true|false, "gap": "Xpx", "padding": "Xpx Xpx"},
  "bg": {"color": "#hex", "has_image": true|false, "image_position": "right|left|full"},
  "elements": [
    {"type": "badge|headline|subtitle|text|button|input|pricing|card|image", "content": "exact text", "color": "#hex", "bg_color": "#hex", "font_size": "Xpx", "font_weight": "bold|normal", "text_transform": "uppercase|lowercase|none", "padding": "Xpx Xpx", "margin": "Xpx Xpx", "border_radius": "Xpx", "width": "Xpx or auto", "height": "Xpx or auto", "position": "left|center|right"}
  ]
}

CRITICAL RULES:
- Every color must be an exact #hex value from the image
- Every text string must be EXACTLY what's in the image — copy verbatim
- Every font size must be a specific pixel value
- Do NOT describe what you see — output ONLY the JSON
- If there's a background image, note its position (right/left/full)
- If text has strikethrough prices, mark them with "strikethrough": true"""

            spec_json = _claude_generate(vision_prompt, [
                {"type": "text", "text": "Extract exact layout spec from this design image."},
                {"type": "image", "source": {"type": "url", "url": vision_url}}
            ], max_tokens=2000)

            # Parse the JSON spec
            if spec_json.startswith("```"):
                spec_json = spec_json.split("\n", 1)[1].rsplit("```", 1)[0]
            spec = json.loads(spec_json)

            # Stage 2: Build exact HTML from the spec
            build_prompt = f"""You are a pixel-perfect web developer. Copy this EXACT layout specification into a single HTML section. Do NOT improvise, do NOT redesign, do NOT add anything not in the spec.

## RIGID SPECIFICATION
```json
{json.dumps(spec, indent=2)}
```

## EXISTING THEME CSS (reference for available classes)
```css
{theme_css[:2000]}
```

## ABSOLUTE RULES
1. EVERY color must be the EXACT hex from the spec — no substitutions
2. EVERY font size, padding, gap must match the spec values exactly
3. ALL text content must be EXACTLY the text strings from the spec — copy verbatim
4. Layout structure (columns, positions) must match the spec exactly
5. If spec has 2 columns: text on left, image/visual on right
6. For buttons: use inline styles matching the spec colors. DO NOT use .btn-primary unless the spec colors match the theme's accent
7. For the background image: use a CSS background-image with the right-side positioning if spec indicates
8. Do NOT add placeholder text like "Left Column" or "Right Column" — those are guides for YOU, not output
9. Use semantic HTML: <section>, <div>, <h1>, <h2>, <p>, <input>, <button>, <a>
10. ALL styles must be INLINE (style="...") — no CSS classes except what's needed for layout
11. Return ONLY the HTML. No markdown, no explanations, no backticks."""

            generated = _claude_generate(build_prompt, "Build the section matching this exact spec.", max_tokens=4000)
        else:
            generated = _claude_generate(system_msg, prompt)

        # Strip markdown code blocks if present
        if generated.startswith("```"):
            lines = generated.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            generated = "\n".join(lines)

        return jsonify({"html": generated, "model": "claude-sonnet-4", "spec": spec if vision_url else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ══════════════════════════════════════════════════════════════
#  AI — Pixel-Perfect Section (Rigid Template)
# ══════════════════════════════════════════════════════════════

@app.route("/api/ai/perfect-hero", methods=["POST"])
def ai_perfect_hero():
    """
    Generate a pixel-perfect hero section from a design image.
    Uses vision to extract the EXACT layout, then Claude copies it rigidly.
    Uses existing video/image assets (no regeneration).
    """
    data = request.get_json() or {}
    design_image_url = data.get("design_image_url", "")
    hero_video_url = data.get("hero_video_url", "")
    hero_image_url = data.get("hero_image_url", "")
    
    if not design_image_url:
        return jsonify({"error": "design_image_url is required"}), 400
    
    video_asset = hero_video_url or hero_image_url or ""
    
    # Step 1: Vision → extract EXACT layout spec
    vision_prompt = """You are a forensic UI analyst. Look at this design image and describe the EXACT pixel-level layout as a rigid specification. Output ONLY this JSON structure — no markdown, no explanations:

{
  "canvas": {"bg_color": "#hex", "texture": "description"},
  "badge": {"text": "exact text", "bg_color": "#hex", "text_color": "#hex", "position": "top-left|top-center", "font_size": "Xpx", "padding": "Xpx Ypx", "text_transform": "lowercase|uppercase", "underline_color": "#hex or null", "underline_word": "which word gets underline"},
  "headline": {"words": [{"text": "FOLLOW", "color": "#hex"}, {"text": "THE", "color": "#hex"}, {"text": "CODE.", "color": "#hex"}], "font_family": "name", "font_size": "Xpx", "font_weight": "bold|normal", "text_transform": "uppercase|lowercase", "letter_spacing": "Xpx", "line_height": "X", "position": "left|center"},
  "email_input": {"placeholder": "exact text", "border_color": "#hex", "border_width": "Xpx", "bg_color": "#hex", "text_color": "#hex", "width": "Xpx", "height": "Xpx", "font_size": "Xpx"},
  "cta_button": {"text": "JOIN WAITLIST", "bg_color": "#hex", "text_color": "#hex", "border_radius": "Xpx", "padding": "Xpx Ypx", "font_size": "Xpx", "font_weight": "bold|normal", "position": "right of input|below input"},
  "pricing": {"lines": [{"text": "exact text", "color": "#hex", "strikethrough": true|false}]},
  "hero_image_treatment": {"effect": "grayscale|halftone|normal", "contrast": "X", "position": "right|left", "width_percent": "X"},
  "layout": {"columns": 2, "text_side": "left", "image_side": "right", "vertical_center": true, "gap": "Xpx", "padding": "Xpx"}
}"""

    try:
        spec_json = _claude_generate(vision_prompt, [
            {"type": "text", "text": "Extract the exact layout from this design."},
            {"type": "image", "source": {"type": "url", "url": design_image_url}}
        ], max_tokens=2000)
        # Parse JSON
        if spec_json.startswith("```"):
            spec_json = spec_json.split("\n", 1)[1].rsplit("```", 1)[0]
        spec = json.loads(spec_json)
    except Exception as e:
        return jsonify({"error": f"Vision extraction failed: {e}"}), 500

    # Step 2: Build rigid HTML/CSS from the spec
    build_prompt = f"""You are a pixel-perfect web developer. Copy this EXACT layout specification into HTML/CSS. Do NOT improvise, do NOT redesign, do NOT "improve" anything. Copy it exactly.

## RIGID LAYOUT SPECIFICATION
```json
{json.dumps(spec, indent=2)}
```

## ASSETS TO USE
Hero video/image: {video_asset or 'NONE — use a dark gradient placeholder'}

## ABSOLUTE RULES
1. EVERY color must be the EXACT hex from the spec — no "similar" colors
2. EVERY font size, padding, gap must match the spec values exactly
3. The layout structure (columns, positions) must match exactly
4. The badge text, headline words, pricing text must be the EXACT text from the spec
5. If a video URL is provided, use it as the hero background: <video autoplay loop muted playsinline style="...the spec's image_treatment..."><source src="{video_asset}" type="video/mp4"></video>
6. If only an image URL, use <img> with the spec's image_treatment
7. Do NOT add any extra sections, text, or elements beyond what the spec defines
8. Use CSS Grid for the two-column layout with exact px gap/padding from spec
9. Apply the background texture/noise if specified in canvas.texture

Return ONLY the hero section HTML. No markdown, no explanations."""

    try:
        hero_html = _claude_generate(build_prompt, "Build the hero section matching this exact spec.", max_tokens=4000)
        if hero_html.startswith("```"):
            lines = hero_html.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            hero_html = "\n".join(lines)
    except Exception as e:
        return jsonify({"error": f"HTML generation failed: {e}"}), 500

    return jsonify({
        "html": hero_html,
        "spec": spec,
        "video_url": video_asset
    })

# ══════════════════════════════════════════════════════════════
#  AI — Closed-Loop Hero Refinement
# ══════════════════════════════════════════════════════════════

def _take_screenshot_b64(url: str, timeout: int = 15) -> str:
    """Take a screenshot of a URL and return as base64 data URL."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("playwright not installed. Run: pip install playwright && playwright install chromium")
    
    import base64
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
        # Wait for fonts to load
        page.wait_for_timeout(2000)
        screenshot = page.screenshot(full_page=False)
        browser.close()
    return f"data:image/png;base64,{base64.b64encode(screenshot).decode()}"


@app.route("/api/ai/refine-hero", methods=["POST"])
def ai_refine_hero():
    """
    Closed-loop refinement: design image → generate → deploy → screenshot → compare → fix → repeat.
    
    Body: {
        "design_image_url": "https://...",
        "seed_html": "<style>...</style><section>...</section>",  // optional — skip extraction+gen, start from this HTML
        "project_id": "hero-refine",          // optional, default auto-generated
        "page_id": "home",                     // optional
        "max_iterations": 3,                   // optional, default 3, max 5
        "hero_video_url": "https://...",       // optional
        "hero_image_url": "https://...",       // optional
        "whatfontis_api_key": "..."            // optional — enables font detection
    }
    """
    data = request.get_json() or {}
    design_image_url = data.get("design_image_url", "")
    seed_html = data.get("seed_html", "")
    project_id = data.get("project_id", f"refine-{uuid.uuid4().hex[:8]}")
    page_id = data.get("page_id", "home")
    max_iterations = min(data.get("max_iterations", 3), 5)
    video_asset = data.get("hero_video_url") or data.get("hero_image_url", "")
    wfi_key = data.get("whatfontis_api_key") or os.environ.get("WHATFONTIS_API_KEY", "")
    known = data.get("known", {})  # LOCKED values from onboarding

    if not design_image_url:
        return jsonify({"error": "design_image_url is required"}), 400
    
    iterations_log = []
    current_html = seed_html  # may be empty → will generate below
    
    # ── If seed_html provided, skip extraction + generation ──
    if seed_html:
        iterations_log.append({"step": "seeded", "status": "ok", "note": "Starting from provided seed HTML"})
    
    # ── Font detection via WhatFontIs (skip if fonts already locked) ──
    detected_fonts = {}
    if wfi_key and not known.get("fonts", {}).get("headline"):
        try:
            import requests as req_lib
            wfi_resp = req_lib.post("https://www.whatfontis.com/api2/", data={
                "API_KEY": wfi_key,
                "IMAGEBASE64": "0",
                "urlimage": design_image_url,
                "FREEFONTS": "1",
                "limit": "10",
                "NOTTEXTBOXSDETECTION": "0"
            }, timeout=30)
            if wfi_resp.status_code == 200:
                fonts = wfi_resp.json()
                if isinstance(fonts, list):
                    detected_fonts = {f["title"].lower(): f for f in fonts[:10]}
                iterations_log.append({"step": "font_detect", "status": "ok", "fonts_found": len(detected_fonts)})
            else:
                iterations_log.append({"step": "font_detect", "status": "error", "error": wfi_resp.text[:200]})
        except Exception as e:
            iterations_log.append({"step": "font_detect", "status": "error", "error": str(e)[:200]})
    
    # ── Step 1+2: Extract spec + generate HTML (only if no seed) ──
    if not seed_html:
        # Font hint for the spec prompt if we have detected fonts
        font_hint = ""
        if detected_fonts:
            font_names = list(detected_fonts.keys())[:5]
            font_hint = f"\nDETECTED FONTS (use exact names): {', '.join(font_names)}"
        
        vision_prompt = """You are a forensic UI analyst. Look at this design image and describe the EXACT pixel-level layout as a rigid specification. Output ONLY this JSON structure — no markdown, no explanations:

{
  "canvas": {"bg_color": "#hex", "texture": "description"},
  "badge": {"text_left": "exact text on left segment", "text_right": "exact text on right segment", "left_bg": "#hex", "right_bg": "#hex", "text_color": "#hex", "right_text_color": "#hex", "position": "top-left|top-center", "font_size": "Xpx", "padding": "Xpx Ypx", "torn_edge": true|false, "font_family": "name"},
  "headline": {"words": [{"text": "FOLLOW", "color": "#hex", "font_family": "name"}, {"text": "THE", "color": "#hex", "font_family": "name"}, {"text": "CODE.", "color": "#hex", "font_family": "name"}], "font_size": "Xpx", "text_transform": "uppercase", "letter_spacing": "Xpx", "line_height": "X"},
  "email_input": {"placeholder": "exact text", "border_color": "#hex", "border_width": "Xpx", "bg_color": "#hex", "text_color": "#hex", "width": "Xpx", "height": "Xpx", "font_size": "Xpx", "text_align": "left|center", "border_radius": "Xpx"},
  "cta_button": {"text": "exact text", "bg_color": "#hex", "text_color": "#hex", "border_radius": "Xpx", "padding": "Xpx Ypx", "font_size": "Xpx", "font_weight": "bold|normal", "gap_from_input": "Xpx"},
  "pricing": {"lines": [{"text": "exact text", "color": "#hex", "strikethrough": true|false, "opacity": "X"}]},
  "hero_image_treatment": {"effect": "grayscale|halftone|normal", "contrast": "X", "position": "right|left", "width_percent": "X", "proximity_to_text": "close|far"},
  "layout": {"columns": 2, "text_side": "left", "image_side": "right", "vertical_center": true, "gap": "Xpx", "padding": "Xpx"}
}""" + font_hint

        try:
            spec_json = _claude_generate(vision_prompt, [
                {"type": "text", "text": "Extract the exact layout from this design."},
                {"type": "image", "source": {"type": "url", "url": design_image_url}}
            ], max_tokens=2000)
            if spec_json.startswith("```"):
                spec_json = spec_json.split("\n", 1)[1].rsplit("```", 1)[0]
            spec = json.loads(spec_json)
            iterations_log.append({"step": "extract_spec", "status": "ok"})
        except Exception as e:
            return jsonify({"error": f"Vision extraction failed: {e}"}), 500

        # ── Override spec with LOCKED known values ──
        if known.get("colors") and len(known["colors"]) >= 2:
            if "canvas" not in spec: spec["canvas"] = {}
            spec["canvas"]["bg_color"] = known["colors"][0]
            if "cta_button" not in spec or spec.get("cta_button") is None: spec["cta_button"] = {}
            spec["cta_button"]["bg_color"] = known["colors"][1] if len(known["colors"]) > 1 else known["colors"][0]
            if "badge" in spec and spec["badge"]:
                spec["badge"]["right_bg"] = known["colors"][1] if len(known["colors"]) > 1 else known["colors"][0]
            iterations_log.append({"step": "locked_colors", "status": "ok", "colors": known["colors"]})
        if known.get("fonts", {}).get("headline"):
            for word in spec.get("headline", {}).get("words", []):
                word["font_family"] = known["fonts"]["headline"]
            iterations_log.append({"step": "locked_fonts", "status": "ok", "headline": known["fonts"]["headline"]})
        if known.get("brand_name"):
            words = spec.get("headline", {}).get("words", [])
            if words:
                words[0]["text"] = known["brand_name"]
                iterations_log.append({"step": "locked_brand", "status": "ok", "name": known["brand_name"]})
        if known.get("cta_text"):
            if "cta_button" not in spec or spec.get("cta_button") is None: spec["cta_button"] = {}
            spec["cta_button"]["text"] = known["cta_text"]
            iterations_log.append({"step": "locked_cta", "status": "ok", "cta": known["cta_text"]})

        # Override font_family in spec with detected fonts if available
        if detected_fonts:
            for word in spec.get("headline", {}).get("words", []):
                word_family = word.get("font_family", "").lower()
                for df_name in detected_fonts:
                    if df_name in word_family or word_family in df_name:
                        word["font_family"] = detected_fonts[df_name]["title"]
                        iterations_log.append({"step": "font_override", "word": word["text"], "font": detected_fonts[df_name]["title"]})

        # ── Inject locked values from known spec ──
        locked_text = ""
        if known:
            locked_lines = []
            kf = known.get("fonts", {})
            if kf.get("headline"): locked_lines.append(f"  - Headline font: {kf['headline']} (LOCKED — import from Google Fonts, use exact name)")
            if kf.get("body"): locked_lines.append(f"  - Body font: {kf['body']} (LOCKED — import from Google Fonts, use exact name)")
            if kf.get("ui"): locked_lines.append(f"  - UI font: {kf['ui']} (LOCKED — use exact name)")
            if known.get("colors"):
                locked_lines.append(f"  - Brand colors: {', '.join(known['colors'])} (LOCKED — use these EXACT hex values ONLY)")
            if known.get("brand_name"):
                locked_lines.append(f"  - Brand name: {known['brand_name']}")
            if known.get("cta_text"):
                locked_lines.append(f"  - CTA button text: {known['cta_text']}")
            if known.get("voice"):
                locked_lines.append(f"  - Visual voice: {known['voice']}")
            if locked_lines:
                locked_text = "\n## LOCKED VALUES — NEVER OVERRIDE\nThese come from the client's brand spec and override any vision extraction:\n" + "\n".join(locked_lines) + "\n"

        build_prompt = f"""You are a pixel-perfect web developer. Copy this EXACT layout specification into HTML/CSS. Do NOT improvise, do NOT redesign.

## RIGID LAYOUT SPECIFICATION
```json
{json.dumps(spec, indent=2)}
```

{locked_text}
## ASSETS TO USE
Hero video/image: {video_asset or 'NONE — use a dark gradient'}

## ABSOLUTE RULES
1. Every color must be the EXACT hex from the spec
2. Every font size, padding, gap must match exactly
3. If badge has torn_edge: true, use CSS clip-path for a jagged right edge on the green segment
4. Input text-align must match spec's text_align
5. CTA button must have gap_from_input px gap from the input
6. Use the EXACT font_family names from the spec for each text element — do NOT substitute
7. Add SVG white film grain texture over the background (feTurbulence + screen blend)
8. Add halftone dot overlay on the knight image if specified
9. Return ONLY <style> + <section> — NO DOCTYPE, NO html/head/body tags
10. Striped badge: use two <span> elements side by side — first with left_bg, second with right_bg"""

        try:
            hero_html = _claude_generate(build_prompt, "Build the hero section.", max_tokens=4000)
            if hero_html.startswith("```"):
                lines = hero_html.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                hero_html = "\n".join(lines)
            iterations_log.append({"step": "generate_initial", "status": "ok"})
        except Exception as e:
            return jsonify({"error": f"HTML generation failed: {e}"}), 500
        
        current_html = hero_html
    
    if not current_html:
        return jsonify({"error": "No HTML to refine. Provide seed_html or ensure design_image_url is valid."}), 400
    
    best_html = current_html
    best_score = 0
    
    for iteration in range(max_iterations):
        # Save current HTML to project
        try:
            meta = _load_meta(project_id)
            if not meta:
                meta = {"name": f"Refine {project_id}", "created": datetime.utcnow().isoformat(), "pages": [], "theme_css": ""}
            pp = _page_path(project_id, page_id)
            pp.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if pp.exists():
                existing = json.loads(pp.read_text())
            existing["components"] = current_html
            pp.write_text(json.dumps(existing, indent=2))
            if page_id not in meta.get("pages", []):
                meta.setdefault("pages", []).append(page_id)
            meta["updated"] = datetime.utcnow().isoformat()
            _save_meta(project_id, meta)
        except Exception as e:
            iterations_log.append({"step": f"save_iter{iteration}", "status": "error", "error": str(e)})
            break
        
        preview_url = f"https://{project_id}.preview.webbeagle.com/{page_id}/"
        
        # Take screenshot of preview
        try:
            screenshot_b64 = _take_screenshot_b64(preview_url)
            iterations_log.append({"step": f"screenshot_iter{iteration}", "status": "ok"})
        except Exception as e:
            iterations_log.append({"step": f"screenshot_iter{iteration}", "status": "error", "error": str(e)})
            # Continue with what we have — can't compare without screenshot
            break
        
        # Compare: original vs screenshot → list discrepancies
        known_compare = ""
        if known:
            kp = []
            if known.get("fonts", {}).get("headline"): kp.append(f"  - Headline font should be: {known['fonts']['headline']} (if rendered uses this, do NOT flag as discrepancy)")
            if known.get("colors"): kp.append(f"  - Brand colors should be: {', '.join(known['colors'])} (if rendered uses these, do NOT flag)")
            if known.get("brand_name"): kp.append(f"  - Brand name should be: {known['brand_name']}")
            if kp:
                known_compare = "## KNOWN SPEC (do NOT flag these as discrepancies if present)\n" + "\n".join(kp) + "\n\n"

        compare_prompt = f"""{known_compare}You are a pixel-level QA inspector. Compare the ORIGINAL design image with the RENDERED screenshot. List EVERY discrepancy, no matter how small.

Output ONLY a JSON object:
{{
  "match_score": 0-100,
  "discrepancies": [
    {{"element": "badge|headline|input|button|pricing|knight|spacing|texture|color|font", "description": "specific issue", "severity": "high|medium|low"}}
  ]
}}

If match_score >= 95, set discrepancies to empty array []."""

        try:
            compare_result = _claude_generate(compare_prompt, [
                {"type": "text", "text": "ORIGINAL DESIGN (reference):"},
                {"type": "image", "source": {"type": "url", "url": design_image_url}},
                {"type": "text", "text": "RENDERED SCREENSHOT (what we built):"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64.split(",", 1)[1] if "," in screenshot_b64 else screenshot_b64}}
            ], max_tokens=1500)
            
            if compare_result.startswith("```"):
                compare_result = compare_result.split("\n", 1)[1].rsplit("```", 1)[0]
            comparison = json.loads(compare_result)
            score = comparison.get("match_score", 0)
            discrepancies = comparison.get("discrepancies", [])
            
            iterations_log.append({
                "step": f"compare_iter{iteration}",
                "status": "ok",
                "score": score,
                "discrepancy_count": len(discrepancies)
            })
            
            if score > best_score:
                best_html = current_html
                best_score = score
            
            # If close enough or no discrepancies, stop
            if score >= 95 or not discrepancies:
                iterations_log.append({"step": "done", "reason": "match_threshold" if score >= 95 else "no_discrepancies"})
                break
                
        except Exception as e:
            iterations_log.append({"step": f"compare_iter{iteration}", "status": "error", "error": str(e)})
            break
        
        # Fix: apply discrepancies to generate corrected HTML
        fix_prompt = f"""Fix this hero section HTML to address ALL of these discrepancies. The original design is the reference.

## DISCREPANCIES TO FIX
{json.dumps(discrepancies, indent=2)}

## CURRENT HTML
```html
{current_html}
```

## RULES
1. Fix ONLY the listed discrepancies — don't change anything else
2. Match the original design exactly
3. Return ONLY the complete corrected <style> + <section> HTML
4. No markdown, no explanations"""

        try:
            fixed_html = _claude_generate(fix_prompt, [
                {"type": "text", "text": "Fix these specific discrepancies. Reference the original design:"},
                {"type": "image", "source": {"type": "url", "url": design_image_url}}
            ], max_tokens=4000)
            
            if fixed_html.startswith("```"):
                lines = fixed_html.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                fixed_html = "\n".join(lines)
            
            current_html = fixed_html
            iterations_log.append({"step": f"fix_iter{iteration}", "status": "ok"})
            
        except Exception as e:
            iterations_log.append({"step": f"fix_iter{iteration}", "status": "error", "error": str(e)})
            break
    
    # ── Return final result ──
    return jsonify({
        "html": best_html,
        "preview_url": f"https://{project_id}.preview.webbeagle.com/{page_id}/",
        "project_id": project_id,
        "page_id": page_id,
        "best_score": best_score,
        "iterations": iterations_log
    })

# ══════════════════════════════════════════════════════════════
#  AI — Pixel-Perfect Layout from Image + Assets
# ══════════════════════════════════════════════════════════════

@app.route("/api/ai/pixel-perfect", methods=["POST"])
def ai_pixel_perfect():
    """
    Convert a design mockup image + exported asset folder into pixel-perfect HTML.
    Three-stage pipeline:
    1. Run find_offsets.py to locate assets within the mockup
    2. Generate pixel-perfect HTML using cqw-based responsive methodology
    3. Closed-loop refinement: screenshot → compare → fix (up to 3 iterations)
    
    Body: {
        "mockup_url": "https://.../mockup.png",
        "prompt": "Build the hero section...",    // optional
        "refine": true                            // optional — enable closed-loop
    }
    """
    import subprocess, urllib.request, base64
    
    data = request.get_json() or {}
    mockup_url = data.get("mockup_url", "")
    prompt = data.get("prompt", "Build a pixel-perfect HTML section matching this mockup exactly.")
    do_refine = data.get("refine", False)
    uploaded_assets = data.get("assets", [])  # From frontend ZIP upload
    
    if not mockup_url:
        return jsonify({"error": "mockup_url is required"}), 400
    
    # ── Step 0: Download mockup ──
    mockup_filename = f"mockup-{uuid.uuid4().hex}.png"
    mockup_path = str(ASSETS_DIR / mockup_filename)
    try:
        urllib.request.urlretrieve(mockup_url, mockup_path)
    except Exception as e:
        return jsonify({"error": f"Failed to download mockup: {e}"}), 400
    
    # Get mockup dimensions
    from PIL import Image as PILImage
    mockup_img = PILImage.open(mockup_path)
    mockup_w, mockup_h = mockup_img.size
    
    # ── Step 1: Run offset finder ──
    offsets_data = []
    skill_dir = Path("/opt/data/skills/custom/converting-images-to-pixel-perfect-html")
    finder_script = skill_dir / "scripts" / "find_offsets.py"
    
    if finder_script.exists():
        try:
            result = subprocess.run(
                ["/opt/grapesjs-api/venv/bin/python3", str(finder_script),
                 "--mockup", mockup_path, "--assets", str(ASSETS_DIR)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                # Parse output: "Match file.png: offset=(374, 77) | size=(433, 543) | SAD=12.45"
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Match "):
                        parts = line.split("|")
                        name_part = parts[0].replace("Match ", "").strip().rstrip(":")
                        offset_str = parts[1].split("=")[1].strip() if len(parts) > 1 else "(0,0)"
                        size_str = parts[2].split("=")[1].strip() if len(parts) > 2 else "(0,0)"
                        # Parse (374, 77)
                        import re as re_mod
                        offset_match = re_mod.findall(r"[\d-]+", offset_str)
                        size_match = re_mod.findall(r"\d+", size_str)
                        offsets_data.append({
                            "name": name_part,
                            "x": int(offset_match[0]) if offset_match else 0,
                            "y": int(offset_match[1]) if len(offset_match) > 1 else 0,
                            "width": int(size_match[0]) if size_match else 0,
                            "height": int(size_match[1]) if len(size_match) > 1 else 0
                        })
                if offsets_data:
                    app.logger.info(f"Offset finder: found {len(offsets_data)} assets")
        except Exception as e:
            app.logger.warning(f"Offset finder failed: {e}")
    
    # Build asset information for the prompt
    # Match uploaded assets with offset finder results to get URLs
    asset_url_map = {}  # original_name → url
    for a in uploaded_assets:
        asset_url_map[a.get("original", "")] = a.get("url", "")
        # Also map by saved filename for flexibility
        saved = a.get("saved", "")
        if saved:
            asset_url_map[saved] = a.get("url", "")
    
    asset_info = ""
    if offsets_data:
        asset_list_lines = []
        for a in offsets_data:
            name = a['name']
            # Find matching asset URL
            asset_url = asset_url_map.get(name, "")
            if not asset_url:
                # Try fuzzy match
                for orig, url in asset_url_map.items():
                    if name in orig or orig in name:
                        asset_url = url
                        break
            asset_list_lines.append(
                f"  - {name}: position=({a['x']},{a['y']}), size={a['width']}x{a['height']}px, "
                f"url={asset_url or 'NOT_FOUND'}"
            )
        asset_list = "\n".join(asset_list_lines)
        asset_info = f"""
## ASSET IMAGES (use these EXACT <img> tags)
Mockup dimensions: {mockup_w}x{mockup_h}px

Each asset below has its exact position AND its public URL. Use <img src="URL" style="position:absolute; top:Ypx; left:Xpx; width:Wpx; height:Hpx;"> for every asset:
{asset_list}

CRITICAL: Use the URL provided for each asset. Do NOT skip any asset — every asset must appear in the HTML."""
    elif uploaded_assets:
        asset_url_list = "\n".join(
            f"  - {a.get('original', a.get('saved', 'unknown'))}: {a.get('url', '')}"
            for a in uploaded_assets
        )
        asset_info = f"""
## UPLOADED ASSETS (use these URLs in <img> tags)
{asset_url_list}

CRITICAL: Use these exact URLs for <img> tags. Position each asset by visually matching it to the mockup."""
    
    # ── Step 2: Generate pixel-perfect HTML ──
    generation_prompt = f"""You are a pixel-perfect web developer specializing in design-to-code conversion. 
Generate a SINGLE self-contained HTML section that reproduces this design mockup EXACTLY.

## METHODOLOGY — PX-BASED ABSOLUTE POSITIONING (GrapesJS Compatible)

### Why PX-Based (NOT cqw/container queries)
This HTML will be inserted into GrapesJS — a visual page builder. Container queries and cqw units BREAK inside GrapesJS's nested iframe/component structure. Use plain px values instead — GrapesJS preserves them exactly.

### Wrapper
Wrap everything in a <section> with:
- `position: relative; width: {mockup_w}px; height: {mockup_h}px; max-width: 100%; overflow: hidden;`
- Background color extracted from the mockup
- This wrapper establishes the coordinate system for all absolute children

### Positioning ALL Elements
1. EVERY element inside the wrapper gets `position: absolute;` with EXACT px values:
   - `top: Xpx; left: Xpx; width: Xpx; height: Xpx;`
2. Font sizes in px: `font-size: Xpx;`
3. All margins/padding in px
4. Use the offset data above for asset images
5. EVERY element gets inline styles — NO CSS classes for positioning

### Mobile Responsive (media query)
```css
@media (max-width: 767px) {{
  section {{ width: 100% !important; height: auto !important; }}
  section * {{ position: relative !important; top: auto !important; left: auto !important; width: 100% !important; height: auto !important; max-width: 100% !important; }}
}}
```
Wrap in a <style> block inside the section.

### Asset Images
For every asset listed above, use: `<img src="URL" style="position:absolute; top:Ypx; left:Xpx; width:Wpx; height:Hpx;">`
Use the EXACT URLs provided — they are publicly accessible.

### SVG Distress/Texture Filter
If the mockup has grunge/stamped/weathered typography, include this SVG filter:
```svg
<svg style="position:absolute;width:0;height:0;">
  <filter id="distress">
    <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="4" result="noise"/>
    <feDisplacementMap in="SourceGraphic" in2="noise" scale="3" xChannelSelector="R" yChannelSelector="G"/>
    <feBlend mode="multiply" in2="SourceGraphic"/>
  </filter>
</svg>
```
Apply to text: `filter:url(#distress);`

### Color Matching
- Extract EXACT #hex colors from the mockup for every element
- Use the mockup's exact background color for the wrapper
- Match text colors precisely — do NOT substitute

### Output Rules
- Return ONLY the complete HTML — no markdown, no explanations, no backticks
- One `<section>` containing everything (wrapper + all children)
- Inline styles on every element for positioning
- NO external CSS files, NO CSS classes for layout
- ALL text must be semantically correct (h1, h2, p, span, button, input)
- If the mockup has a CTA button, use a real `<button>` with the exact colors
- Include a <style> block inside the section for media queries and font-face declarations

{asset_info}"""
    
    try:
        generated_html = _claude_generate(generation_prompt, [
            {"type": "text", "text": prompt},
            {"type": "image", "source": {"type": "url", "url": mockup_url}}
        ], max_tokens=8000)
        
        if generated_html.startswith("```"):
            lines = generated_html.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            generated_html = "\n".join(lines)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500
    
    # ── Step 3: Closed-loop refinement (screenshot → compare → fix) ──
    iterations = []
    if do_refine:
        for iteration in range(3):
            try:
                # Write HTML to temp file and serve it
                temp_name = f"refine-{uuid.uuid4().hex}.html"
                temp_path = ASSETS_DIR / temp_name
                temp_path.write_text(f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{margin:0;padding:0;background:#111;display:flex;justify-content:center;align-items:center;min-height:100vh;}}</style>
</head><body>{generated_html}</body></html>""")
                
                host = request.host
                forwarded = request.headers.get("X-Forwarded-Host", "")
                if forwarded:
                    host = forwarded
                temp_url = f"https://{host}/assets/{temp_name}"
                
                # Screenshot the rendered output
                rendered_b64 = _take_screenshot_b64(temp_url)
                
                # Clean up
                temp_path.unlink(missing_ok=True)
                
                # Compare with mockup
                compare_prompt = f"""Compare the rendered output (IMAGE 1) with the original mockup (IMAGE 2).
Identify SPECIFIC differences in:
- Element positioning (offsets, alignment)
- Sizes (widths, heights, font sizes)
- Colors (exact hex mismatches)
- Missing or extra elements
- Spacing issues

Output a JSON diff object with actionable fixes:
{{"score": 0-100, "differences": ["specific fix 1", "specific fix 2", ...], "critical_fixes": ["must-fix 1", ...]}}

If score >= 90, add "PASS": true to the JSON."""
                
                diff_json = _claude_generate(compare_prompt, [
                    {"type": "text", "text": "Compare these two images and find differences."},
                    {"type": "image", "source": {"type": "url", "url": f"data:image/png;base64,{rendered_b64.split(',')[1] if ',' in rendered_b64 else rendered_b64}"}},
                    {"type": "image", "source": {"type": "url", "url": mockup_url}}
                ], max_tokens=1000)
                
                # Parse the diff
                if diff_json.startswith("```"):
                    diff_json = diff_json.split("\n", 1)[1].rsplit("```", 1)[0]
                import re as re_mod
                diff_json = re_mod.sub(r'^[^{]*', '', diff_json)
                diff = json.loads(diff_json)
                
                score = diff.get("score", 0)
                iterations.append({"iteration": iteration + 1, "score": score, "differences": diff.get("differences", [])})
                
                if score >= 90 or diff.get("PASS"):
                    break
                
                # Fix the differences
                fixes = "\n".join(f"- {d}" for d in diff.get("critical_fixes", diff.get("differences", [])))
                fix_prompt = f"""The rendered output has these differences from the mockup (score: {score}/100):
{fixes}

Rewrite the HTML to fix ALL of these issues. Keep everything that was already correct.
Return ONLY the complete fixed HTML. No markdown, no explanations."""
                
                fixed_html = _claude_generate(fix_prompt, [
                    {"type": "text", "text": "Fix these layout differences to match the mockup exactly."},
                    {"type": "image", "source": {"type": "url", "url": mockup_url}}
                ], max_tokens=8000)
                
                if fixed_html.startswith("```"):
                    lines = fixed_html.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    fixed_html = "\n".join(lines)
                
                generated_html = fixed_html
                
            except Exception as e:
                app.logger.warning(f"Refine iteration {iteration+1} failed: {e}")
                iterations.append({"iteration": iteration + 1, "error": str(e)})
                break
    
    return jsonify({
        "html": generated_html,
        "mockup": {"url": mockup_url, "width": mockup_w, "height": mockup_h},
        "offsets": offsets_data,
        "refinement": iterations,
        "mockup_path": f"/assets/{mockup_filename}"
    })

@app.route("/api/ai/pixel-perfect/refine", methods=["POST"])
def ai_pixel_perfect_refine():
    """
    Closed-loop refinement: take existing HTML, screenshot it, compare with mockup,
    and iterate until score >= 90 or 3 iterations used.
    
    Body: {
        "html": "<section>...</section>",
        "mockup_url": "https://.../mockup.png"
    }
    """
    import subprocess, urllib.request, base64
    
    data = request.get_json() or {}
    html = data.get("html", "")
    mockup_url = data.get("mockup_url", "")
    
    if not html or not mockup_url:
        return jsonify({"error": "html and mockup_url are required"}), 400
    
    iterations = []
    current_html = html
    
    for i in range(3):
        try:
            # Write HTML to temp file
            temp_name = f"refine-{uuid.uuid4().hex}.html"
            temp_path = ASSETS_DIR / temp_name
            temp_path.write_text(f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{margin:0;padding:0;background:#111;display:flex;justify-content:center;align-items:center;min-height:100vh;}}</style>
</head><body>{current_html}</body></html>""")
            
            host = request.host
            forwarded = request.headers.get("X-Forwarded-Host", "")
            if forwarded:
                host = forwarded
            temp_url = f"https://{host}/assets/{temp_name}"
            
            # Screenshot
            rendered_b64 = _take_screenshot_b64(temp_url)
            temp_path.unlink(missing_ok=True)
            
            # Compare
            compare_prompt = """Compare the rendered output (IMAGE 1) with the original mockup (IMAGE 2).
Identify SPECIFIC differences in positioning, sizes, colors, missing elements, and spacing.
Output a JSON diff: {"score": 0-100, "differences": ["fix 1", "fix 2"], "critical_fixes": ["must-fix"]}
If score >= 90, add "PASS": true."""
            
            diff_json = _claude_generate(compare_prompt, [
                {"type": "text", "text": "Compare these two images and find differences."},
                {"type": "image", "source": {"type": "url", "url": f"data:image/png;base64,{rendered_b64.split(',')[1] if ',' in rendered_b64 else rendered_b64}"}},
                {"type": "image", "source": {"type": "url", "url": mockup_url}}
            ], max_tokens=1000)
            
            if diff_json.startswith("```"):
                diff_json = diff_json.split("\n", 1)[1].rsplit("```", 1)[0]
            import re as re_mod
            diff_json = re_mod.sub(r'^[^{]*', '', diff_json)
            diff = json.loads(diff_json)
            
            score = diff.get("score", 0)
            iterations.append({"iteration": i + 1, "score": score, "differences": diff.get("differences", [])})
            
            if score >= 90 or diff.get("PASS"):
                break
            
            # Fix
            fixes = "\n".join(f"- {d}" for d in diff.get("critical_fixes", diff.get("differences", [])))
            fix_prompt = f"""Fix these differences from the mockup (score: {score}/100):
{fixes}

Rewrite the HTML to fix ALL issues. Return ONLY the complete fixed HTML. No markdown."""
            
            fixed_html = _claude_generate(fix_prompt, [
                {"type": "text", "text": "Fix these layout differences to match the mockup exactly."},
                {"type": "image", "source": {"type": "url", "url": mockup_url}}
            ], max_tokens=8000)
            
            if fixed_html.startswith("```"):
                lines = fixed_html.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                fixed_html = "\n".join(lines)
            
            current_html = fixed_html
            
        except Exception as e:
            app.logger.warning(f"Refine iteration {i+1} failed: {e}")
            iterations.append({"iteration": i + 1, "error": str(e)})
            break
    
    return jsonify({
        "html": current_html,
        "refinement": iterations,
        "final_score": iterations[-1].get("score", 0) if iterations else 0
    })

# ══════════════════════════════════════════════════════════════
#  AI — URL Scraper (Design Token Extraction)
# ══════════════════════════════════════════════════════════════

@app.route("/api/ai/scrape-url", methods=["POST"])
def ai_scrape_url():
    """
    Extract design tokens from a URL — fonts, colors, layout, button styles.

    Body: {
        "url": "https://site-to-scrape.com",
        "detect_fonts": true              // optional — run WhatFontIs
    }

    Returns: {
        "dominant_fonts": [...],
        "color_palette": [...],
        "layout_pattern": "hero-centered",
        "hero_structure": {...},
        "button_style": {...},
        "spacing_system": "generous",
        "screenshot_b64": "data:image/png;base64,...",
        "errors": [...]
    }
    """
    from url_scraper import scrape_url

    data = request.get_json(force=True)
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "url is required"}), 400

    wfi_key = data.get("whatfontis_api_key") or os.environ.get("WHATFONTIS_API_KEY", "")
    if not data.get("detect_fonts", True):
        wfi_key = ""

    result = scrape_url(url, whatfontis_key=wfi_key)

    return jsonify(result)


# ══════════════════════════════════════════════════════════════
#  AI — Concept Generation (Hero Image Concepts)
# ══════════════════════════════════════════════════════════════

@app.route("/api/ai/generate-concepts", methods=["POST"])
def ai_generate_concepts():
    """
    Generate 3 hero section concept images via KIE for client selection.

    Body: {
        "spec": {                           // merged spec from onboard or manual
            "fonts": {"headline": "Bebas Neue", "body": "Cinzel"},
            "colors": ["#000", "#22c55e", "#f5f0e8"],
            "site_type": "landing_page",
            "industry": "fitness",
            "brand_name": "SigmaCal"
        },
        "style_keywords": ["industrial", "gritty"],  // optional vibe hints
        "count": 3                          // optional, default 3
    }

    Returns: {
        "concepts": [
            {
                "id": "concept-1",
                "image_url": "https://...",
                "prompt": "...",
                "layout_hint": "hero-centered dark industrial with knight motif"
            }, ...
        ],
        "task_ids": [str, ...]              // KIE task IDs for polling
    }
    """
    data = request.get_json(force=True)
    spec = data.get("spec", {})
    style_keywords = data.get("style_keywords", [])
    count = min(data.get("count", 3), 5)

    if not spec:
        return jsonify({"error": "spec is required — provide at minimum colors and site_type"}), 400

    fonts = spec.get("fonts", {})
    colors = spec.get("colors", [])
    site_type = spec.get("site_type", "landing_page")
    brand_name = spec.get("brand_name", "")
    industry = spec.get("industry", "")

    # ── Build 3 distinct concept prompts ──
    concepts = [
        {
            "id": "concept-1",
            "style": "dark_gritty",
            "layout_hint": "bold centered hero with dramatic lighting",
            "prompt": _build_concept_prompt("dark_gritty", spec, style_keywords, brand_name, industry)
        },
        {
            "id": "concept-2",
            "style": "clean_modern",
            "layout_hint": "minimal hero with accent color pops and generous whitespace",
            "prompt": _build_concept_prompt("clean_modern", spec, style_keywords, brand_name, industry)
        },
        {
            "id": "concept-3",
            "style": "editorial",
            "layout_hint": "magazine-style hero with asymmetric layout and bold typography",
            "prompt": _build_concept_prompt("editorial", spec, style_keywords, brand_name, industry)
        }
    ]

    # Submit to KIE
    task_ids = []
    for concept in concepts[:count]:
        try:
            task_id = _kie_submit("grok-imagine/text-to-image", {
                "prompt": concept["prompt"],
                "aspect_ratio": "16:9"
            })
            concept["task_id"] = task_id
            task_ids.append(task_id)
        except Exception as e:
            concept["error"] = str(e)

    return jsonify({
        "concepts": concepts[:count],
        "task_ids": task_ids
    })


def _build_concept_prompt(style: str, spec: dict, keywords: list, brand_name: str, industry: str) -> str:
    """Build a KIE image prompt for a hero section concept."""
    colors = spec.get("colors", [])
    color_str = ", ".join(colors[:3]) if colors else "dark with accent green"

    style_prompts = {
        "dark_gritty": f"dark moody hero section, {color_str} color scheme, dramatic lighting, gritty texture, bold typography area, cinematic composition, 16:9, professional web design",
        "clean_modern": f"clean modern hero section, {color_str} color scheme, minimal design, generous whitespace, subtle shadows, professional UI, 16:9, sleek web design",
        "editorial": f"editorial magazine hero section, {color_str} color scheme, asymmetric layout, bold serif typography, high contrast, artistic composition, 16:9, premium web design"
    }

    prompt = style_prompts.get(style, style_prompts["dark_gritty"])

    if brand_name:
        prompt = f"{brand_name} website hero — {prompt}"
    if industry:
        prompt = f"{industry} brand — {prompt}"
    if keywords:
        prompt = f"{prompt}, {' '.join(keywords)}"

    return prompt


# ══════════════════════════════════════════════════════════════
#  AI — Project Onboarding (Full Pipeline)
# ══════════════════════════════════════════════════════════════

SPECS_DIR = Path("/opt/data/grapesjs-specs")
SPECS_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/api/ai/project-onboard", methods=["POST"])
def ai_project_onboard():
    """
    Full onboarding pipeline: ingest known + scrape URLs + detect fonts + merge + generate.

    Body: {
        "project_name": "sigmacal",
        "known": {
            "fonts": {"headline": "Bebas Neue", "body": "Cinzel"},
            "colors": ["#000000", "#22c55e", "#f5f0e8"],
            "logo_url": "https://...",
            "seed_html": "<style>...</style><section>...</section>",
            "site_type": "landing_page",
            "pages": ["home", "pricing"],
            "brand_name": "SigmaCal",
            "industry": "fitness tracking",
            "voice": "gritty, motivational"
        },
        "scrape_urls": ["https://existing-site.com", "https://competitor.com"],
        "screenshots": ["https://pub.img.url/design1.png", ...],
        "generate_concepts": false,
        "build_immediately": true
    }

    Returns: {
        "project_id": "sigmacal",
        "spec": {...},                      // merged design spec
        "warnings": [...],
        "gaps": [...],
        "confidence": {"fonts": "high", "colors": "medium", "layout": "high"},
        "preview_url": "https://..."       // if build_immediately
    }
    """
    from url_scraper import scrape_url
    from spec_merger import merge_spec, validate_spec

    data = request.get_json(force=True)
    project_name = data.get("project_name", f"onboard-{uuid.uuid4().hex[:8]}")
    known = data.get("known", {})
    scrape_urls = data.get("scrape_urls", [])
    screenshots = data.get("screenshots", [])
    generate_concepts = data.get("generate_concepts", False)
    build_immediately = data.get("build_immediately", True)

    wfi_key = data.get("whatfontis_api_key") or os.environ.get("WHATFONTIS_API_KEY", "")

    # ── Step 1: Scrape URLs ──
    scraped_tokens = {}
    scraped = []
    for url in scrape_urls:
        try:
            tokens = scrape_url(url, whatfontis_key=wfi_key)
            scraped.append({"url": url, "tokens": tokens})
        except Exception as e:
            scraped.append({"url": url, "error": str(e)})

    # Merge scraped results (last URL wins on conflicts, but we track all)
    all_scraped_fonts = []
    all_scraped_colors = []
    all_scraped_layouts = []
    for s in scraped:
        t = s.get("tokens", {})
        all_scraped_fonts.extend(t.get("dominant_fonts", []))
        all_scraped_colors.extend(t.get("color_palette", []))
        if t.get("layout_pattern"):
            all_scraped_layouts.append(t["layout_pattern"])

    detected = {
        "dominant_fonts": list(dict.fromkeys(all_scraped_fonts))[:5],  # dedupe preserve order
        "color_palette": list(dict.fromkeys(all_scraped_colors))[:8],
        "layout_pattern": all_scraped_layouts[0] if all_scraped_layouts else "unknown",
        "hero_structure": scraped[0]["tokens"].get("hero_structure", {}) if scraped else {},
        "button_style": scraped[0]["tokens"].get("button_style", {}) if scraped else {},
        "spacing_system": scraped[0]["tokens"].get("spacing_system", "unknown") if scraped else "unknown"
    }

    # ── Step 2: Run WhatFontIs on screenshots if provided ──
    if screenshots and wfi_key:
        import requests as req_lib
        screenshot_fonts = []
        for img_url in screenshots[:3]:
            try:
                wfi_resp = req_lib.post("https://www.whatfontis.com/api2/", data={
                    "API_KEY": wfi_key,
                    "IMAGEBASE64": "0",
                    "urlimage": img_url,
                    "FREEFONTS": "1",
                    "limit": "10",
                    "NOTTEXTBOXSDETECTION": "0"
                }, timeout=30)
                if wfi_resp.status_code == 200:
                    fonts = wfi_resp.json()
                    if isinstance(fonts, list):
                        for f in fonts[:5]:
                            screenshot_fonts.append(f["title"])
            except Exception:
                pass
        if screenshot_fonts:
            # Screenshot fonts take priority over scraped
            detected["dominant_fonts"] = list(dict.fromkeys(screenshot_fonts + detected.get("dominant_fonts", [])))[:5]

    # ── Step 3: Merge known + detected ──
    merged = merge_spec(known, detected)
    spec = merged["spec"]

    # Remove internal keys from spec before saving/returning
    spec.pop("_font_sources", None)
    spec.pop("_sources", None)

    # ── Step 4: Save spec ──
    spec_path = SPECS_DIR / f"{project_name}.json"
    spec_doc = {
        "project_name": project_name,
        "spec": spec,
        "known": known,
        "detected": detected,
        "scraped": scraped,
        "warnings": merged["warnings"],
        "gaps": merged["gaps"],
        "confidence": merged["confidence"],
        "created": datetime.utcnow().isoformat(),
        "updated": datetime.utcnow().isoformat()
    }
    spec_path.write_text(json.dumps(spec_doc, indent=2))

    response = {
        "project_id": project_name,
        "spec": spec,
        "warnings": merged["warnings"],
        "gaps": merged["gaps"],
        "confidence": merged["confidence"],
        "scraped_urls": len(scraped),
        "spec_url": f"/api/ai/spec/{project_name}"
    }

    # ── Step 5: Optionally generate concepts ──
    if generate_concepts:
        try:
            concept_task_ids = []
            for i, concept_style in enumerate(["dark_gritty", "clean_modern", "editorial"]):
                prompt = _build_concept_prompt(concept_style, spec, known.get("style_keywords", []),
                                               known.get("brand_name", ""), known.get("industry", ""))
                tid = _kie_submit("grok-imagine/text-to-image", {"prompt": prompt, "aspect_ratio": "16:9"})
                concept_task_ids.append({"style": concept_style, "task_id": tid})
            response["concept_task_ids"] = concept_task_ids
        except Exception as e:
            response["concept_error"] = str(e)

    # ── Step 6: Optionally build immediately ──
    if build_immediately and spec.get("seed_html"):
        try:
            project_id = project_name
            page_id = "home"
            seed = spec["seed_html"]

            # Deploy seed HTML as project
            project_dir = PROJECTS_DIR / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "pages").mkdir(exist_ok=True)
            (project_dir / f"pages/{page_id}.html").write_text(seed)
            _save_meta(project_id, {
                "name": project_name,
                "pages": [page_id],
                "created": datetime.utcnow().isoformat()
            })
            response["preview_url"] = f"https://{project_id}.preview.webbeagle.com/{page_id}/"
        except Exception as e:
            response["build_error"] = str(e)

    return jsonify(response)


# ══════════════════════════════════════════════════════════════
#  AI — Spec Management (Read / Update)
# ══════════════════════════════════════════════════════════════

@app.route("/api/ai/spec/<project_id>", methods=["GET"])
def ai_get_spec(project_id):
    """Read the current merged spec for a project."""
    spec_path = SPECS_DIR / f"{project_id}.json"
    if not spec_path.exists():
        return jsonify({"error": "Spec not found"}), 404
    return jsonify(json.loads(spec_path.read_text()))


@app.route("/api/ai/spec/<project_id>", methods=["PATCH"])
def ai_update_spec(project_id):
    """
    Update known values and re-merge the spec.

    Body: {
        "known": {
            "fonts": {"body": "Inter"},
            "colors": ["#000", "#fff", "#ff0000"],
            "cta_text": "GET STARTED"
        }
    }

    Returns the re-merged spec.
    """
    from spec_merger import merge_spec

    spec_path = SPECS_DIR / f"{project_id}.json"
    if not spec_path.exists():
        return jsonify({"error": "Spec not found. Run project-onboard first."}), 404

    doc = json.loads(spec_path.read_text())
    data = request.get_json(force=True)

    new_known = data.get("known", {})
    # Merge with existing known (incoming overrides)
    existing_known = doc.get("known", {})
    existing_known = _deep_merge(existing_known, new_known)

    # Re-merge
    merged = merge_spec(existing_known, doc.get("detected", {}))
    spec = merged["spec"]
    spec.pop("_font_sources", None)
    spec.pop("_sources", None)

    # Save
    doc["known"] = existing_known
    doc["spec"] = spec
    doc["warnings"] = merged["warnings"]
    doc["gaps"] = merged["gaps"]
    doc["confidence"] = merged["confidence"]
    doc["updated"] = datetime.utcnow().isoformat()
    spec_path.write_text(json.dumps(doc, indent=2))

    return jsonify(doc)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ══════════════════════════════════════════════════════════════
#  AI — Full Site Redesign (Legacy — use perfect-hero instead)
# ══════════════════════════════════════════════════════════════

@app.route("/api/ai/redesign", methods=["POST"])
def ai_redesign():
    """
    Full site redesign: design_image + site_url → new GrapesJS project.
    
    Steps:
    1. Vision-analyze design image → extract DESIGN.md tokens + theme CSS
    2. Scrape site content → sections, text, images, form HTML
    3. Generate hero background image via KIE
    4. Generate hero video (optional) via KIE
    5. Assemble all sections with new theme
    6. Create GrapesJS project and return slug
    """
    data = request.get_json() or {}
    design_image_url = data.get("design_image_url", "")
    site_url = data.get("site_url", "")
    generate_video = data.get("generate_video", False)
    project_name = data.get("name", "Redesigned Site")

    if not design_image_url or not site_url:
        return jsonify({"error": "design_image_url and site_url are required"}), 400

    steps_log = []

    # ── Step 1: Vision → CSS directly (NO JSON middleman) ──
    vision_css_prompt = """You are a world-class web designer. Look at this design image and generate COMPLETE, PIXEL-PERFECT theme CSS that recreates it exactly.

## CRITICAL — Extracted these EXACT elements from the image:
- The beige/paper strip banner at the very top with small text
- The massive bold uppercase headline with one word in bright green
- The dark background with subtle grit/noise texture
- The email input field with a dark outlined box
- The bright green CTA button with dark text
- The character/hero image on the right in a halftone dot pattern style, black and white, high contrast
- The strikethrough original pricing with green current pricing
- The two-column layout (text left, image right)

## CSS Requirements:
1. @import Google Fonts: Bebas Neue (headings), Inter (body)
2. :root variables for: --bg (#0a0a0a true black), --text (#ffffff), --accent (bright green from image), --paper (beige from badge), --muted (gray for secondary text)
3. body { background:#0a0a0a; color:#fff; font-family:'Inter',sans-serif; }
4. .badge { background:beige-paper-color; color:#1a1a1a; display:inline-block; padding:6px 16px; font-size:11px; text-transform:uppercase; letter-spacing:2px; font-weight:600; }
5. .headline { font-family:'Bebas Neue',sans-serif; font-size:clamp(3.5rem,8vw,7rem); font-weight:400; text-transform:uppercase; letter-spacing:2px; line-height:0.9; }
6. .accent { color: bright-green-from-image; }
7. .hero-section { display:grid; grid-template-columns:1fr 1fr; min-height:100vh; align-items:center; gap:60px; padding:120px 80px; }
8. .email-input { background:transparent; border:1px solid #555; color:#fff; padding:14px 20px; font-size:16px; width:100%; max-width:360px; }
9. .btn-accent { background: bright-green; color:#0a0a0a; border:none; padding:14px 32px; font-weight:700; text-transform:uppercase; letter-spacing:1px; cursor:pointer; }
10. .pricing-original { text-decoration:line-through; color:#888; }
11. .pricing-current { color: bright-green; font-size:1.5rem; font-weight:700; }
12. .hero-image img { width:100%; filter:grayscale(100%) contrast(1.4); }
13. .noise-overlay { position:fixed; inset:0; pointer-events:none; opacity:0.04; background: url noise pattern; }
14. .section { padding:100px 80px; max-width:1400px; margin:0 auto; }
15. .feature-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:40px; }
16. .feature-card { border:1px solid rgba(255,255,255,0.06); padding:30px; }
17. @media(max-width:768px){.hero-section{grid-template-columns:1fr;padding:60px 24px;}.section{padding:60px 24px;}}
18. @keyframes drift { 0%{transform:translate(0,0)} 100%{transform:translate(1px,1px)} }

Return ONLY the CSS. No markdown, no explanations, no HTML. Every hex color must be EXACTLY what you see in the image."""
    try:
        theme_css = _claude_generate(vision_css_prompt, [
            {"type": "text", "text": "Generate the CSS for this design."},
            {"type": "image", "source": {"type": "url", "url": design_image_url}}
        ], max_tokens=4000)
        # Strip markdown wrappers
        if theme_css.startswith("```"):
            lines = theme_css.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            theme_css = "\n".join(lines)
        steps_log.append(f"1/5: ✓ Theme CSS generated ({len(theme_css)} chars)")
    except Exception as e:
        return jsonify({"error": f"CSS generation failed: {e}", "steps": steps_log}), 500

    # ── Step 2: Content Extraction ──
    steps_log.append("2/5: Extracting site content...")
    try:
        import urllib.request
        from html.parser import HTMLParser
        
        req = urllib.request.Request(site_url, headers={"User-Agent": "WebBeagle/1.0"})
        html_content = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
        # Extract body content, strip scripts
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
        body_html = body_match.group(1) if body_match else html_content
        body_html = re.sub(r'<script[\s\S]*?</script>', '', body_html, flags=re.IGNORECASE)
        body_html = re.sub(r'<style[\s\S]*?</style>', '', body_html, flags=re.IGNORECASE)
        # Truncate for AI — need enough for all sections
        site_content = body_html[:30000]
        steps_log.append("2/5: ✓ Content extracted ({:,} chars)".format(len(site_content)))
    except Exception as e:
        return jsonify({"error": f"Content extraction failed: {e}", "steps": steps_log}), 500

    # ── Step 3: Generate Hero Image via KIE ──
    steps_log.append("3/5: Generating hero image...")
    hero_image_path = ""
    try:
        img_prompt = "Dark heroic character portrait. Knight in armor, high contrast black and white, halftone dot pattern effect, grunge texture overlay. No text, no UI elements. Cinematic side lighting. Dark gritty industrial aesthetic. 16:9. Professional quality website hero background."
        hero_image_path = _generate_image(img_prompt)
        steps_log.append(f"3/5: ✓ Hero image generated")
    except Exception as e:
        steps_log.append(f"3/5: ⚠ Image gen failed ({str(e)[:60]}), continuing without hero image")

    # ── Step 4: Generate Video (optional) ──
    hero_video_path = ""
    if generate_video:
        steps_log.append("4/5: Generating hero video...")
        try:
            vid_prompt = "Slow cinematic push-in on a dark armored knight figure. Dust particles floating in dramatic side lighting. High contrast black and white. Haunting gritty atmosphere. Seamless loop. 6 seconds. No text."
            hero_video_path = _generate_video(vid_prompt)
            steps_log.append("4/5: ✓ Hero video generated")
        except Exception as e:
            steps_log.append(f"4/5: ⚠ Video gen failed ({str(e)[:60]}), continuing without video")
    else:
        steps_log.append("4/5: Skipped (video not requested)")

    # ── Step 5: AI Assembly ──
    steps_log.append("5/5: Assembling site with new theme...")
    assembly_prompt = f"""You are rebuilding an entire website with a new visual design while preserving all the original CONTENT.

## THEME CSS (Authoritative — use these EXACT class names)
```css
{theme_css}
```

## HERO IMAGE
{hero_image_path or 'No hero image — use CSS background gradient matching the theme.'}

## HERO VIDEO
{hero_video_path or 'No video.'}

## ORIGINAL SITE CONTENT (Preserve ALL text, links, forms, and structure)
```html
{site_content}
```

## YOUR TASK — 10/10 QUALITY
Rebuild the entire site as a single HTML page. CRITICAL RULES:
1. Use THEME CSS classes for EVERY element: .badge, .headline, .accent (for green words), .subheadline, .email-input, .btn-accent, .pricing-original, .pricing-current, .hero-image, .hero-section, .noise-overlay, .section, .feature-grid, .feature-card, .section-dark
2. .headline text UPPERCASE with .accent on the highlight word
3. .badge creates beige paper strip effect — use for top tagline
4. Hero layout: two columns (.hero-section) — text left, hero image right
5. Include ALL sections from original: hero/waitlist, features (Photo Scan, Full Ingredient Logging, SigmaChef, Exercise Tracking, Rules), Your Plan, Aura Points, Log Anything, pricing CTA, footer
6. Preserve ALL form elements exactly — the waitlist form MUST work (method, action, inputs)
7. Remove broken image URLs (images/ paths from old site)
8. Include noise-overlay div for gritty texture
9. Every section has proper spacing, every text uses the right font class

Return ONLY the complete HTML. No markdown, no explanations. The page MUST be pixel-perfect to the theme CSS."""

    try:
        full_html = _claude_generate(assembly_prompt, "Rebuild this website with the new theme.", max_tokens=20000)
        # Strip markdown wrappers
        if full_html.startswith("```"):
            lines = full_html.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            full_html = "\n".join(lines)
        steps_log.append("5/5: ✓ Site assembled")
    except Exception as e:
        return jsonify({"error": f"Assembly failed: {e}", "steps": steps_log}), 500

    # ── Create Project ──
    slug = _sanitize_slug(project_name) or str(uuid.uuid4())[:8]
    counter = 1
    base_slug = slug
    while _project_path(slug).exists():
        slug = f"{base_slug}-{counter}"
        counter += 1

    now = datetime.utcnow().isoformat()
    meta = {
        "name": project_name,
        "created": now,
        "updated": now,
        "pages": ["home"],
        "theme_css": theme_css,
    }
    _save_meta(slug, meta)
    page_dir = _page_path(slug, "home").parent
    page_dir.mkdir(parents=True, exist_ok=True)
    _page_path(slug, "home").write_text(json.dumps({
        "components": full_html,
        "styles": "",
    }))

    return jsonify({
        "project_id": slug,
        "name": project_name,
        "preview_url": f"https://{slug}.preview.webbeagle.com/",
        "builder_url": f"https://builder.webbeagle.com/?project={slug}",
        "hero_image": hero_image_path,
        "hero_video": hero_video_path,
        "steps": steps_log,
    })

# ══════════════════════════════════════════════════════════════
#  Component Library Registry (Enhanced — Categories + Traits)
# ══════════════════════════════════════════════════════════════

@app.route("/api/components", methods=["GET"])
def list_components():
    """List components. Query: ?category=buttons (optional)"""
    category = request.args.get("category", "")
    components = []
    
    if category:
        cat_dir = COMPONENTS_DIR / category
        pattern = cat_dir.glob("*.json") if cat_dir.exists() else []
    else:
        cat_dirs = [d for d in COMPONENTS_DIR.iterdir() if d.is_dir()]
        pattern = []
        for d in cat_dirs:
            pattern.extend(d.glob("*.json"))
    
    for f in sorted(pattern):
        data = json.loads(f.read_text())
        data["id"] = f.stem
        data["category"] = f.parent.name if f.parent != COMPONENTS_DIR else ""
        components.append(data)
    
    return jsonify({"components": components})

@app.route("/api/components/categories", methods=["GET"])
def list_categories():
    cats = []
    for d in sorted(COMPONENTS_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*.json")))
            cats.append({"name": d.name, "count": count})
    return jsonify({"categories": cats})

@app.route("/api/components", methods=["POST"])
def save_component():
    """Save component. Body: {name, category, html, css, traits, tags, thumbnail}"""
    data = request.get_json() or {}
    name = data.get("name", "Untitled")
    category = data.get("category", "sections")
    html = data.get("html", "")
    css = data.get("css", "")
    traits = data.get("traits", [])
    tags = data.get("tags", [])
    thumbnail = data.get("thumbnail", "")
    icon = data.get("icon", "")
    
    slug = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-"))[:40]
    counter = 1
    base = slug or "component"
    
    cat_dir = COMPONENTS_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    while (cat_dir / f"{slug}.json").exists():
        slug = f"{base}-{counter}"
        counter += 1
    
    comp = {
        "name": name, "category": category, "html": html, "css": css,
        "traits": traits, "tags": tags, "thumbnail": thumbnail, "icon": icon,
        "created": datetime.utcnow().isoformat(),
        "updated": datetime.utcnow().isoformat()
    }
    
    (cat_dir / f"{slug}.json").write_text(json.dumps(comp, indent=2))
    return jsonify({"id": slug, "name": name, "category": category}), 201

@app.route("/api/components/<comp_id>", methods=["GET"])
def get_component(comp_id):
    for d in COMPONENTS_DIR.iterdir():
        if d.is_dir():
            path = d / f"{comp_id}.json"
            if path.exists():
                data = json.loads(path.read_text())
                data["id"] = comp_id
                data["category"] = d.name
                return jsonify(data)
    return jsonify({"error": "Not found"}), 404

@app.route("/api/components/<comp_id>", methods=["PUT"])
def update_component(comp_id):
    data = request.get_json() or {}
    for d in COMPONENTS_DIR.iterdir():
        if d.is_dir():
            path = d / f"{comp_id}.json"
            if path.exists():
                comp = json.loads(path.read_text())
                for key in ["name", "html", "css", "traits", "tags", "thumbnail", "icon"]:
                    if key in data:
                        comp[key] = data[key]
                if "category" in data and data["category"] != d.name:
                    new_dir = COMPONENTS_DIR / data["category"]
                    new_dir.mkdir(parents=True, exist_ok=True)
                    new_path = new_dir / f"{comp_id}.json"
                    new_path.write_text(json.dumps(comp, indent=2))
                    path.unlink()
                    return jsonify({"status": "updated", "id": comp_id, "category": data["category"]})
                comp["updated"] = datetime.utcnow().isoformat()
                path.write_text(json.dumps(comp, indent=2))
                return jsonify({"status": "updated", "id": comp_id})
    return jsonify({"error": "Not found"}), 404

@app.route("/api/components/<comp_id>", methods=["DELETE"])
def delete_component(comp_id):
    for d in COMPONENTS_DIR.iterdir():
        if d.is_dir():
            path = d / f"{comp_id}.json"
            if path.exists():
                path.unlink()
                return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8092, debug=False)
