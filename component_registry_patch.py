# ══════════════════════════════════════════════════════════════
#  Component Library Registry (Enhanced — Categories + Traits)
# ══════════════════════════════════════════════════════════════

@app.route("/api/components", methods=["GET"])
def list_components():
    """List components. Query: ?category=buttons (optional)"""
    category = request.args.get("category", "")
    components = []
    
    if category:
        # Search specific category directory
        cat_dir = COMPONENTS_DIR / category
        pattern = cat_dir.glob("*.json") if cat_dir.exists() else []
    else:
        # Search all category directories
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
    """List all component categories with counts."""
    cats = []
    for d in sorted(COMPONENTS_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*.json")))
            cats.append({"name": d.name, "count": count})
    return jsonify({"categories": cats})

@app.route("/api/components", methods=["POST"])
def save_component():
    """Save a component. Body: {name, category, html, css, traits, tags, thumbnail}"""
    data = request.get_json() or {}
    name = data.get("name", "Untitled")
    category = data.get("category", "sections")
    html = data.get("html", "")
    css = data.get("css", "")
    traits = data.get("traits", [])
    tags = data.get("tags", [])
    thumbnail = data.get("thumbnail", "")
    icon = data.get("icon", "")
    
    # Generate slug
    slug = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-"))[:40]
    counter = 1
    base = slug or "component"
    
    # Ensure category directory exists
    cat_dir = COMPONENTS_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    while (cat_dir / f"{slug}.json").exists():
        slug = f"{base}-{counter}"
        counter += 1
    
    comp = {
        "name": name,
        "category": category,
        "html": html,
        "css": css,
        "traits": traits,
        "tags": tags,
        "thumbnail": thumbnail,
        "icon": icon,
        "created": datetime.utcnow().isoformat(),
        "updated": datetime.utcnow().isoformat()
    }
    
    (cat_dir / f"{slug}.json").write_text(json.dumps(comp, indent=2))
    return jsonify({"id": slug, "name": name, "category": category}), 201

@app.route("/api/components/<comp_id>", methods=["GET"])
def get_component(comp_id):
    """Get a single component by ID."""
    # Search all categories
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
    """Update a component's fields."""
    data = request.get_json() or {}
    
    # Find the component
    for d in COMPONENTS_DIR.iterdir():
        if d.is_dir():
            path = d / f"{comp_id}.json"
            if path.exists():
                comp = json.loads(path.read_text())
                for key in ["name", "html", "css", "traits", "tags", "thumbnail", "icon"]:
                    if key in data:
                        comp[key] = data[key]
                if "category" in data and data["category"] != d.name:
                    # Move to new category
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
    """Delete a component."""
    for d in COMPONENTS_DIR.iterdir():
        if d.is_dir():
            path = d / f"{comp_id}.json"
            if path.exists():
                path.unlink()
                return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404
