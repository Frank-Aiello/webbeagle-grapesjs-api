#!/usr/bin/env python3
"""
GrapesJS Projects API — WebBeagle Builder
Serves: project CRUD, multi-page management, preview rendering
"""
import json
import os
import re
import uuid
import shutil
import yaml
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────
PROJECTS_DIR = Path("/opt/data/grapesjs-projects")
COMPONENTS_DIR = Path("/opt/data/grapesjs-components")
BUILDER_DIR = Path("/opt/data/grapesjs-demo")
CONFIG_PATH = Path("/opt/data/config.yaml")
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load API key from config ─────────────────────────────────
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
    # Fallback to env var
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

ai_client = _get_openai_client()

app = Flask(__name__, static_folder=str(BUILDER_DIR), static_url_path="")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

# ── Subdomain Routing ──────────────────────────────────────
@app.before_request
def route_by_subdomain():
    """Route *.wstdwork.webbeagle.com and *.preview.webbeagle.com to project previews."""
    host = request.host
    # Check both preview domains
    for suffix in ['.wstdwork.webbeagle.com', '.preview.webbeagle.com']:
        if host.endswith(suffix):
            subdomain = host[:-len(suffix)]  # strip the suffix (which includes leading dot)
            if subdomain == 'grapesjs':
                return  # builder subdomain — serve builder normally
            # Check if this subdomain matches a project
            project_path = PROJECTS_DIR / subdomain
            if project_path.exists() and project_path.is_dir():
                path = request.path.strip('/')
                page_id = path if path else 'home'
                html = _render_preview(subdomain, page_id)
                if html:
                    return html
                # Page not found — try home
                if page_id != 'home':
                    html = _render_preview(subdomain, 'home')
                    if html:
                        return html
            # Project doesn't exist — fall through to builder
            break

# ── Helpers ─────────────────────────────────────────────────
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

    # Build standalone HTML
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
// Scroll reveal
(function(){{
  var observer = new IntersectionObserver(function(entries) {{
    entries.forEach(function(entry) {{
      if (entry.isIntersecting) entry.target.classList.add('visible');
    }});
  }}, {{ threshold: 0.15 }});
  document.querySelectorAll('.reveal').forEach(function(el) {{ observer.observe(el); }});

  // Glass shimmer
  document.querySelectorAll('.glass').forEach(function(card) {{
    card.addEventListener('mousemove', function(e) {{
      var rect = card.getBoundingClientRect();
      card.style.setProperty('--mx', ((e.clientX - rect.left) / rect.width) * 100 + '%');
      card.style.setProperty('--my', ((e.clientY - rect.top) / rect.height) * 100 + '%');
    }});
  }});

  // Sticky hero opacity
  var hero = document.querySelector('.hero');
  var heroWrap = document.querySelector('.hero-wrap');
  if (heroWrap && hero) {{
    window.addEventListener('scroll', function() {{
      var rect = heroWrap.getBoundingClientRect();
      var progress = 1 - Math.max(0, Math.min(1, -rect.top / (rect.height - window.innerHeight)));
      hero.style.opacity = 0.4 + (progress * 0.6);
    }}, {{ passive: true }});
  }}

  // Smooth scroll
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

# ── Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BUILDER_DIR), "index.html")

@app.route("/preview.html")
def standalone_preview():
    return send_from_directory(str(BUILDER_DIR), "preview.html")

# ── API: Health ─────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

# ── API: List Projects ──────────────────────────────────────
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

# ── API: Create Project ─────────────────────────────────────
@app.route("/api/projects", methods=["POST"])
def create_project():
    data = request.get_json() or {}
    name = data.get("name", "Untitled Project")
    slug = data.get("slug") or _sanitize_slug(name) or str(uuid.uuid4())[:8]
    project_id = slug

    # Ensure unique
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

    # Create default home page
    page_dir = _page_path(project_id, "home").parent
    page_dir.mkdir(parents=True, exist_ok=True)
    _page_path(project_id, "home").write_text(json.dumps({
        "components": data.get("components", ""),
        "styles": data.get("styles", ""),
    }))

    return jsonify({"id": project_id, "name": name, "pages": ["home"]}), 201

# ── API: Get Project ────────────────────────────────────────
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

# ── API: Save Project ───────────────────────────────────────
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

    # Save pages
    if "pages_data" in data:
        for page_id, page_data in data["pages_data"].items():
            pp = _page_path(project_id, page_id)
            pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(json.dumps(page_data, indent=2))
        meta["pages"] = list(data["pages_data"].keys())

    # Save current page
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

# ── API: Delete Project ─────────────────────────────────────
@app.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    pp = _project_path(project_id)
    if pp.exists():
        shutil.rmtree(pp)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Project not found"}), 404

# ── API: Add Page ───────────────────────────────────────────
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

# ── API: Delete Page ────────────────────────────────────────
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

# ── Preview ─────────────────────────────────────────────────
@app.route("/preview/<project_id>/<page_id>/")
@app.route("/preview/<project_id>/<page_id>")
def preview(project_id, page_id):
    html = _render_preview(project_id, page_id)
    if html is None:
        return jsonify({"error": "Project or page not found"}), 404
    return html

# ── AI: Generate Section ────────────────────────────────────
@app.route("/api/ai/generate", methods=["POST"])
def ai_generate():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    design_md = data.get("design_md", "")
    existing_html = data.get("existing_html", "")
    context_html = data.get("context_html", "")
    mode = data.get("mode", "generate")  # "generate" or "edit"
    vision_url = data.get("vision_url", "")  # base64 data URL from image upload

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Build system message
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

    if existing_html:
        system_msg += f"\n\n## Existing HTML to Modify\n```html\n{existing_html[:4000]}\n```"

    if context_html and mode != "edit":
        system_msg += f"\n\n## Existing Page Context\nThe following HTML is already on the page. Make sure your section integrates well with it:\n```html\n{context_html[:3000]}\n```"

    try:
        # Build messages — include image if provided
        messages = [{"role": "system", "content": system_msg}]

        if vision_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": vision_url, "detail": "high"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )
        generated = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if generated.startswith("```"):
            lines = generated.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            generated = "\n".join(lines)

        return jsonify({
            "html": generated,
            "tokens_used": response.usage.total_tokens if response.usage else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Component Library ───────────────────────────────────────
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
    thumbnail = data.get("thumbnail", "")  # base64 data URL

    # Generate slug from name
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
    p = COMPONENTS_DIR / f"{comp_id}.json"
    if p.exists():
        p.unlink()
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8092, debug=False)
