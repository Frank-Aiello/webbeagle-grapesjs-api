#!/usr/bin/env python3
"""
GrapesJS Projects API — WebBeagle Builder
Serves: project CRUD, multi-page management, preview rendering,
        AI section generation (Claude Sonnet 4), AI redesign (vision + KIE assets),
        KIE image & video generation
"""
import json
import os
import re
import uuid
import shutil
import time
import yaml
import requests
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
CONFIG_PATH = Path("/opt/data/config.yaml")
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

KIE_API_KEY = os.environ.get("KIE_API_KEY", "")
KIE_BASE = "https://api.kie.ai/api/v1"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
    """Poll for task completion. Returns {resultImageUrl, resultVideoUrl, ...}."""
    # Determine query endpoint based on model family
    if "grok" in model:
        query_url = f"{KIE_BASE}/grok/record-info?taskId={taskId}"
    elif "flux" in model:
        query_url = f"{KIE_BASE}/flux/kontext/record-info?taskId={taskId}"
    else:
        query_url = f"{KIE_BASE}/jobs/taskStatus?taskId={taskId}"
    
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(
            query_url,
            headers={"Authorization": f"Bearer {KIE_API_KEY}"},
            timeout=10
        )
        data = r.json()
        # Check for success
        success_flag = data.get("data", {}).get("successFlag")
        if success_flag == 1:
            response_data = data.get("data", {}).get("response", {})
            return {
                "resultImageUrl": response_data.get("resultImageUrl"),
                "resultVideoUrl": response_data.get("resultVideoUrl"),
            }
        elif success_flag in (2, 3):
            raise Exception(f"KIE task failed: {data}")
        time.sleep(3)
    raise Exception(f"KIE task {taskId} timed out after {timeout}s")

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
        info = _kie_poll(taskId, model, timeout=120)  # 2 min timeout
        img_url = info.get("resultImageUrl")
        if not img_url:
            return ""
        filename = f"kie-img-{taskId[:8]}.png"
        return _download_asset(img_url, filename)
    except Exception as e:
        print(f"[KIE image] {e}")
        return ""

def _generate_video(prompt: str, aspect_ratio: str = "16:9", duration: str = "5",
                    model: str = "grok-imagine/text-to-video") -> str:
    """Generate video via KIE, download to assets, return public URL path."""
    taskId = _kie_submit(model, {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "resolution": "720p"
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
                page_id = path if path else 'home'
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

def _sanitize_slug(s):
    return re.sub(r"[^a-z0-9-]", "", s.lower().replace(" ", "-"))[:40]

def _render_preview(project_id, page_id):
    """Build a standalone HTML page from saved project data."""
    meta = _load_meta(project_id)
    if not meta:
        return None
    page_file = _page_path(project_id, page_id)
    if not page_file.exists():
        return None
    page_data = json.loads(page_file.read_text())
    components_html = page_data.get("components", "")
    styles_css = page_data.get("styles", "")
    css_raw = meta.get("theme_css", "")
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
</head>
<body>
{components_html}
<script>
(function(){{
  var observer = new IntersectionObserver(function(entries) {{
    entries.forEach(function(entry) {{
      if (entry.isIntersecting) entry.target.classList.add('visible');
    }});
  }}, {{ threshold: 0.15 }});
  document.querySelectorAll('.reveal').forEach(function(el) {{ observer.observe(el); }});
  document.querySelectorAll('.glass').forEach(function(card) {{
    card.addEventListener('mousemove', function(e) {{
      var rect = card.getBoundingClientRect();
      card.style.setProperty('--mx', ((e.clientX - rect.left) / rect.width) * 100 + '%');
      card.style.setProperty('--my', ((e.clientY - rect.top) / rect.height) * 100 + '%');
    }});
  }});
  var hero = document.querySelector('.hero');
  var heroWrap = document.querySelector('.hero-wrap');
  if (heroWrap && hero) {{
    window.addEventListener('scroll', function() {{
      var rect = heroWrap.getBoundingClientRect();
      var progress = 1 - Math.max(0, Math.min(1, -rect.top / (rect.height - window.innerHeight)));
      hero.style.opacity = 0.4 + (progress * 0.6);
    }}, {{ passive: true }});
  }}
  document.querySelectorAll('a[href^="#"]').forEach(function(link) {{
    link.addEventListener('click', function(e) {{
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {{ e.preventDefault(); target.scrollIntoView({{ behavior: 'smooth' }}); }}
    }});
  }});
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
    meta["updated"] = now
    if "pages_data" in data:
        for page_id, page_data in data["pages_data"].items():
            pp = _page_path(project_id, page_id)
            pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(json.dumps(page_data, indent=2))
        meta["pages"] = list(data["pages_data"].keys())
    elif "components" in data or "styles" in data:
        page_id = data.get("page_id", "home")
        existing = {}
        pp = _page_path(project_id, page_id)
        if pp.exists():
            existing = json.loads(pp.read_text())
        existing.update({
            "components": data.get("components", existing.get("components", "")),
            "styles": data.get("styles", existing.get("styles", "")),
        })
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps(existing, indent=2))
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
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": block["source"]["url"]
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
            generated = _claude_generate(system_msg, [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "url", "url": vision_url}}
            ])
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

        return jsonify({"html": generated, "model": "claude-sonnet-4"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ══════════════════════════════════════════════════════════════
#  AI — Full Site Redesign
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
        hero_style = design_data.get("hero_image_style", "")
        mood = design_data.get("mood", "dark gritty industrial")
        img_prompt = f"Dark heroic character portrait. {hero_style}. {mood}. High contrast black and white, halftone dot pattern effect, grunge texture overlay. No text, no UI elements. Cinematic lighting from side. 16:9. Professional quality suitable for website hero background."
        hero_image_path = _generate_image(img_prompt)
        steps_log.append(f"3/5: ✓ Hero image generated")
    except Exception as e:
        steps_log.append(f"3/5: ⚠ Image gen failed ({str(e)[:60]}), continuing without hero image")

    # ── Step 4: Generate Video (optional) ──
    hero_video_path = ""
    if generate_video:
        steps_log.append("4/5: Generating hero video...")
        try:
            vid_prompt = f"Slow cinematic push-in on a dark armored figure. {mood}. Dust particles floating in dramatic side lighting. High contrast black and white. No text, no faces visible clearly. Haunting atmosphere. Seamless loop feel. 6 seconds."
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
#  Component Library
# ══════════════════════════════════════════════════════════════

@app.route("/api/components", methods=["GET"])
def list_components():
    components = []
    for f in sorted(COMPONENTS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        data["id"] = f.stem
        components.append(data)
    return jsonify({"components": components})

@app.route("/api/components", methods=["POST"])
def save_component():
    data = request.get_json() or {}
    name = data.get("name", "Untitled")
    html = data.get("html", "")
    thumbnail = data.get("thumbnail", "")
    slug = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-"))[:30]
    counter = 1
    base = slug or "component"
    while (COMPONENTS_DIR / f"{slug}.json").exists():
        slug = f"{base}-{counter}"
        counter += 1
    comp = {
        "name": name,
        "html": html,
        "thumbnail": thumbnail,
        "created": datetime.utcnow().isoformat()
    }
    (COMPONENTS_DIR / f"{slug}.json").write_text(json.dumps(comp, indent=2))
    return jsonify({"id": slug, "name": name}), 201

@app.route("/api/components/<comp_id>", methods=["DELETE"])
def delete_component(comp_id):
    path = COMPONENTS_DIR / f"{comp_id}.json"
    if path.exists():
        path.unlink()
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8092, debug=False)
