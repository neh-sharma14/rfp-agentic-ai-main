### File: src/compliance_crew/crew.py

import os
from ..tools import LlamaIndexQueryTool

def extract_json_requirements(response_text: str):
    import json, re, ast
    try:
        match = re.search(r'(\[.*?\])', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return []
    except Exception:
        try:
            return ast.literal_eval(response_text)
        except Exception:
            return []

def normalize_requirement(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".,;:")

async def kickoff_compliance_requirements_crew(knowledge_dir: str = "./knowledge_extract_text", inputs: dict = {}):
    if not os.path.exists(knowledge_dir) or not os.path.isdir(knowledge_dir):
        raise FileNotFoundError(f"Folder path '{knowledge_dir}' does not exist or is not a directory.")

    # ✅ Use LlamaIndexQueryTool instead of agents
    llama_tool = LlamaIndexQueryTool(knowledge_dir=knowledge_dir)
    query_prompt = (
        "Extract all compliance requirements from the knowledge base in the format:\n"
        "[{\"sectionNo\": \"<section number>\", \"requirement\": \"<requirement text>\"}, ...]\n"
        "Each requirement must have a section number. Use '0' if unavailable."
    )
    
    try:
        print("🔍 Querying compliance requirements with LlamaIndex...")
        response = llama_tool.query(query_prompt)
        raw_requirements = extract_json_requirements(response)
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {"requirements": []}

    # ✅ Deduplication
    seen = set()
    final = []
    for req in raw_requirements:
        if not isinstance(req, dict):
            continue
        section_no = str(req.get("sectionNo", "0")).strip()
        requirement = req.get("requirement", "").strip()
        if not requirement:
            continue
        norm = normalize_requirement(requirement)
        if norm not in seen:
            final.append({"sectionNo": section_no, "requirement": requirement})
            seen.add(norm)

    print(f"✅ Extracted {len(final)} unique compliance requirements.")
    return {"requirements": final}
