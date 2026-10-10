"""Classify business-card activity using the supplied category workbook first,
augmented with web search and LLM domain reasoning.

Search/LLM failures never block OCR processing.
"""
from __future__ import annotations

import json
import logging
import re
import difflib
import zipfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

from config import settings

log = logging.getLogger(__name__)

_MAX_DESC_LEN = 200
_TIMEOUT = 8
_CATEGORY_FILE = Path(__file__).resolve().parents[1] / "CATEGORY FILE (09-10-2025).xlsx"
_XLS_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
}

STANDARD_MAIN_DOMAINS = [
    "Agriculture & Food",
    "Apparel, Textiles & Accessories",
    "Auto & Transportation",
    "Bags, Shoes & Accessories",
    "Banking & Financial Services",
    "Construction & Real Estate",
    "Electrical Equipment, Components & Telecoms",
    "Electronics",
    "Energy & Petroleum",
    "Gifts, Sports & Toys",
    "Health & Beauty",
    "Home, Lights & Construction",
    "IT & Technology",
    "Logistics & Supply Chain",
    "Machinery, Industrial Parts & Tools",
    "Media, Marketing & Advertising",
    "Metallurgy, Chemicals, Rubber & Plastics",
    "Packaging, Advertising & Office",
    "Other Services",
]


@dataclass(frozen=True)
class Classification:
    business_type: str
    confidence: int


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


@lru_cache(maxsize=1)
def _category_rows() -> list[dict[str, str]]:
    """Load FIELD1/FIELD2/MAINCLASS/SUBSCLASS from the supplied workbook."""
    if not _CATEGORY_FILE.is_file():
        log.warning("Business category workbook not found: %s", _CATEGORY_FILE.name)
        return []
    try:
        with zipfile.ZipFile(_CATEGORY_FILE) as workbook:
            names = set(workbook.namelist())
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
                shared = [
                    "".join(t.text or "" for t in item.findall(".//m:t", _XLS_NS))
                    for item in root.findall("m:si", _XLS_NS)
                ]

            sheet_path = "xl/worksheets/sheet1.xml"
            sheet = ET.fromstring(workbook.read(sheet_path))
            parsed: list[dict[str, str]] = []
            for row in sheet.findall(".//m:sheetData/m:row", _XLS_NS):
                values: dict[int, str] = {}
                for cell in row.findall("m:c", _XLS_NS):
                    ref = cell.attrib.get("r", "")
                    col = re.match(r"[A-Z]+", ref)
                    if not col:
                        continue
                    col_number = 0
                    for char in col.group(0):
                        col_number = col_number * 26 + ord(char) - 64
                    value = cell.find("m:v", _XLS_NS)
                    inline = cell.find("m:is", _XLS_NS)
                    text = (
                        "".join(t.text or "" for t in inline.findall(".//m:t", _XLS_NS))
                        if inline is not None else (value.text if value is not None else "")
                    )
                    if cell.attrib.get("t") == "s" and text:
                        try:
                            text = shared[int(text)]
                        except (IndexError, ValueError):
                            text = ""
                    values[col_number] = str(text or "").strip()
                if values.get(1, "").upper() == "FIELD1":
                    continue
                entry = {
                    "field1": values.get(1, ""),
                    "field2": values.get(2, ""),
                    "main_class": values.get(3, ""),
                    "sub_class": values.get(4, ""),
                }
                if entry["field1"]:
                    parsed.append(entry)

            # Rows containing "(See ...)" redirect to another category entry.
            by_field1 = {_normalize(row["field1"]): row for row in parsed}
            for row in parsed:
                if row["main_class"] and row["sub_class"]:
                    continue
                target = re.search(r"\bsee\s+(?:also\s+)?(.+?)[).]?$", row["field2"], re.I)
                if target:
                    destination = by_field1.get(_normalize(target.group(1)))
                    if destination:
                        row["main_class"] = destination.get("main_class") or ""
                        row["sub_class"] = destination.get("sub_class") or ""
            return [r for r in parsed if r["main_class"] or r["sub_class"]]
    except Exception as exc:
        log.error("Could not load business category workbook: %s", exc)
        return []


def _excel_category(
    company_name: str,
    business_hint: str,
    raw_text: str,
) -> tuple[dict[str, str], int] | None:
    """Find a card's business activity in the supplied workbook."""
    rows = _category_rows()
    if not rows:
        return None
    aliases: list[tuple[str, dict[str, str]]] = []
    for row in rows:
        for candidate in (row["field1"], row["field2"]):
            normalized = _normalize(candidate)
            if normalized and not normalized.startswith("see also"):
                aliases.append((normalized, row))

    candidates = [company_name, business_hint]
    candidates.extend(
        line.strip(" •-|,;")
        for line in raw_text.splitlines()
        if 3 <= len(line.strip()) <= 100
    )
    normalized_candidates = list(dict.fromkeys(
        _normalize(candidate) for candidate in candidates if _normalize(candidate)
    ))

    # Exact matches and explicit activity phrases in OCR text take priority.
    for query in normalized_candidates:
        if len(query) < 3:
            continue
        for alias, row in aliases:
            if query == alias:
                return row, 100
            if len(alias) >= 5 and f" {alias} " in f" {query} ":
                return row, 96

    # High-confidence fuzzy match (0.92+)
    best: tuple[float, dict[str, str]] | None = None
    for query in normalized_candidates:
        if len(query) < 5:
            continue
        for alias, row in aliases:
            score = difflib.SequenceMatcher(None, query, alias).ratio()
            if best is None or score > best[0]:
                best = (score, row)
    if best and best[0] >= 0.92:
        return best[1], round(best[0] * 100)
    return None


_INDUSTRY_MAP: list[tuple[list[str], str]] = [
    (["software", "saas", "app development", "mobile app", "cloud solutions", "cybersecurity",
      "artificial intelligence", "data science", "web solution", "web solutions", "web development",
      "web design", "website development", "digital solution", "digital solutions", "software house",
      "it services", "it solutions", "information technology", "tech solutions"],
     "IT & Technology"),
    (["petroleum", "oil & gas", "oil and gas", "refinery", "lng", "lpg", "drilling",
      "fuel", "pipeline", "solar energy", "renewable energy", "power generation"],
     "Energy & Petroleum"),
    (["bank", "banking", "finance", "financial", "insurance", "investment", "capital",
      "securities", "wealth management", "microfinance", "leasing", "fintech", "chartered accountant"],
     "Banking & Financial Services"),
    (["hospital", "clinic", "healthcare", "pharma", "pharmaceutical", "medical", "medicine",
      "diagnostic", "biotech", "laboratory", "surgical"],
     "Health & Beauty"),
    (["construction", "builder", "architect", "real estate", "property", "housing",
      "infrastructure", "civil engineering", "contractor"],
     "Home, Lights & Construction"),
    (["textile", "garments", "apparel", "fabrics", "spinning", "weaving", "cotton mills",
      "fashion", "yarn", "hosiery"],
     "Apparel, Textiles & Accessories"),
    (["machinery", "industrial equipment", "tools", "automation", "bearings", "compressors",
      "pumps", "hydraulics", "pneumatics"],
     "Machinery, Industrial Parts & Tools"),
    (["chemicals", "metallurgy", "steel", "iron", "plastics", "polymers", "rubber",
      "minerals", "coatings", "paints"],
     "Metallurgy, Chemicals, Rubber & Plastics"),
    (["electronics", "security systems", "cctv", "telecom", "telecommunications", "networking"],
     "Electronics"),
    (["auto", "automotive", "vehicles", "transport", "logistics", "freight", "cargo",
      "courier", "shipping", "supply chain"],
     "Auto & Transportation"),
    (["agriculture", "agri", "farming", "fertilizers", "pesticides", "seeds", "food products",
      "beverages", "confectionery", "dairy"],
     "Agriculture & Food"),
    (["packaging", "printing", "corrugated boxes", "advertising", "marketing agency", "stationery"],
     "Packaging, Advertising & Office"),
]


def _clean_website(website: str) -> str:
    """Strip protocol/www and extract the bare domain."""
    website = str(website or "").strip().lower()
    website = re.sub(r"^https?://", "", website)
    website = re.sub(r"^www\.", "", website)
    return website.split("/")[0].strip()


def _ddg_abstract(query: str) -> tuple[str, str]:
    """Query DuckDuckGo Instant Answer API. Returns (abstract_text, entity_type)."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        abstract = str(data.get("AbstractText") or data.get("Answer") or "").strip()
        entity_type = str(data.get("Type") or "").strip()
        return abstract, entity_type
    except Exception as exc:
        log.debug("DDG lookup failed for %r: %s", query, exc)
        return "", ""


def _website_description(website: str) -> str:
    """Read public title/meta description from the printed website when available."""
    domain = _clean_website(website)
    if not domain:
        return ""
    for scheme in ("https://", "http://"):
        try:
            req = urllib.request.Request(
                scheme + domain,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                html = resp.read(250_000).decode("utf-8", errors="replace")
            title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            description = re.search(
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
                html, re.I | re.S,
            )
            og_desc = re.search(
                r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
                html, re.I | re.S,
            )
            text_parts = []
            for item in (title, description, og_desc):
                if item:
                    cleaned = re.sub(r"<[^>]+>", " ", item.group(1)).strip()
                    if cleaned:
                        text_parts.append(cleaned)
            text = " | ".join(text_parts)
            if text:
                return re.sub(r"\s+", " ", text)[:400]
        except Exception as exc:
            log.debug("Website lookup failed for %s: %s", domain, exc)
    return ""


def _classify_from_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").lower())
    for keywords, label in _INDUSTRY_MAP:
        for kw in keywords:
            pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
            if re.search(pattern, normalized):
                return label
    return ""


def _llm_classify(
    company_name: str,
    person_name: str,
    job_title: str,
    address: str,
    website: str,
    web_context: str,
    raw_text: str,
) -> Classification | None:
    """Use configured LLMs (Gemini / NVIDIA / OpenRouter) to determine the exact main domain."""
    prompt = (
        "You are an expert business classifier. Analyze the business card details and web context below "
        "to determine the MAIN BUSINESS DOMAIN and sub-activity for this company.\n\n"
        f"Company Name: {company_name or 'N/A'}\n"
        f"Contact Person: {person_name or 'N/A'}\n"
        f"Designation / Title: {job_title or 'N/A'}\n"
        f"Address / Location: {address or 'N/A'}\n"
        f"Website: {website or 'N/A'}\n"
        f"Web Search / Site Summary: {web_context or 'N/A'}\n"
        f"Raw Card OCR Text:\n{raw_text[:300] if raw_text else 'N/A'}\n\n"
        "Available Standard Main Domains:\n"
        "- Agriculture & Food\n"
        "- Apparel, Textiles & Accessories\n"
        "- Auto & Transportation\n"
        "- Bags, Shoes & Accessories\n"
        "- Banking & Financial Services\n"
        "- Construction & Real Estate\n"
        "- Electrical Equipment, Components & Telecoms\n"
        "- Electronics\n"
        "- Energy & Petroleum\n"
        "- Gifts, Sports & Toys\n"
        "- Health & Beauty\n"
        "- Home, Lights & Construction\n"
        "- IT & Technology\n"
        "- Logistics & Supply Chain\n"
        "- Machinery, Industrial Parts & Tools\n"
        "- Media, Marketing & Advertising\n"
        "- Metallurgy, Chemicals, Rubber & Plastics\n"
        "- Packaging, Advertising & Office\n"
        "- Other Services\n\n"
        "RULES:\n"
        "1. Strictly base your categorization on factual evidence from company name, website, title, location, or search summary. DO NOT guess or hallucinate.\n"
        "2. Choose the most accurate Main Domain from the list above or provide a precise industry name.\n"
        "3. Provide the specific sub-class / activity.\n"
        "4. Return ONLY a valid JSON object with keys: 'main_domain', 'sub_class', 'confidence' (integer 0-100), 'reason'."
    )

    # Strategy 1: Gemini API
    if settings.gemini_api_key:
        try:
            from services import gemini_verifier
            res = gemini_verifier._call_gemini(prompt)
            main_domain = str(res.get("main_domain") or "").strip()
            sub_class = str(res.get("sub_class") or "").strip()
            conf = int(res.get("confidence") or 88)
            if main_domain and main_domain.lower() != "unknown":
                label = f"Main Class: {main_domain}"
                if sub_class:
                    label += f" | Sub Class: {sub_class}"
                return Classification(label, min(100, max(50, conf)))
        except Exception as exc:
            log.debug("Gemini LLM classification failed: %s", exc)

    # Strategy 2: NVIDIA API
    if settings.nvidia_api_key:
        try:
            payload = {
                "model": settings.nvidia_model,
                "temperature": 0.1,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            }
            req = urllib.request.Request(
                settings.nvidia_base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + settings.nvidia_api_key,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "Visitors-OCR/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            from services.qwen_extractor import _parse_object
            data = _parse_object(content)
            main_domain = str(data.get("main_domain") or "").strip()
            sub_class = str(data.get("sub_class") or "").strip()
            conf = int(data.get("confidence") or 85)
            if main_domain and main_domain.lower() != "unknown":
                label = f"Main Class: {main_domain}"
                if sub_class:
                    label += f" | Sub Class: {sub_class}"
                return Classification(label, min(100, max(50, conf)))
        except Exception as exc:
            log.debug("NVIDIA LLM classification failed: %s", exc)

    return None


def classify_with_confidence(
    company_name: str = "",
    website: str = "",
    address: str = "",
    business_hint: str = "",
    raw_text: str = "",
    person_name: str = "",
) -> Classification:
    """Classify business domain using:
    1. Excel taxonomy match (fast, direct)
    2. Global web search / website metadata
    3. LLM synthesis & reasoning (Gemini / NVIDIA / Qwen)
    """
    comp = (company_name or "").strip()
    hint = (business_hint or "").strip()
    pers = (person_name or "").strip()
    web = (website or "").strip()
    addr = (address or "").strip()

    # Step 1: Direct Excel Match
    workbook_match = _excel_category(comp, hint, raw_text)
    if workbook_match and workbook_match[1] >= 94:
        row, confidence = workbook_match
        main_class = row["main_class"] or "Other Services"
        sub_class = row["sub_class"] or "General"
        label = f"Main Class: {main_class} | Sub Class: {sub_class} | Field: {row['field1']}"
        return Classification(label, confidence)

    # Step 2: Global Search context
    search_context_parts = []
    
    # Check website metadata if printed
    if web:
        site_desc = _website_description(web)
        if site_desc:
            search_context_parts.append(f"Website Info ({web}): {site_desc}")

    # Search DuckDuckGo for company / person context
    search_query = comp
    if not search_query and pers:
        search_query = pers
    if search_query:
        country_hint = ""
        if addr:
            lo = addr.lower()
            if any(c in lo for c in ("pakistan", "karachi", "lahore", "islamabad", "rawalpindi", "faisalabad", "multan", "peshawar", "quetta", "sialkot", "gujranwala")):
                country_hint = " Pakistan"
        ddg_text, _ = _ddg_abstract(f"{search_query}{country_hint}")
        if ddg_text:
            search_context_parts.append(f"DuckDuckGo Summary: {ddg_text}")

    combined_web_context = " ; ".join(search_context_parts)

    # Step 3: LLM Domain Classification (Gemini / NVIDIA)
    if comp or web or raw_text:
        llm_res = _llm_classify(
            company_name=comp,
            person_name=pers,
            job_title=hint,
            address=addr,
            website=web,
            web_context=combined_web_context,
            raw_text=raw_text,
        )
        if llm_res and llm_res.confidence >= 65:
            return llm_res

    # Step 4: If LLM is unreachable, use rule-based industry keywords
    rule_category = _classify_from_text(f"{comp} {hint} {combined_web_context} {raw_text[:200]}")
    if rule_category:
        return Classification(f"Main Class: {rule_category}", 85)

    # Step 5: Fallback to partial workbook match if available
    if workbook_match:
        row, confidence = workbook_match
        main_class = row["main_class"] or "Other Services"
        sub_class = row["sub_class"] or "General"
        label = f"Main Class: {main_class} | Sub Class: {sub_class} | Field: {row['field1']}"
        return Classification(label, confidence)

    return Classification("", 0)


def classify(
    company_name: str = "",
    website: str = "",
    address: str = "",
    business_hint: str = "",
) -> str:
    """Convenience wrapper returning the formatted business classification string."""
    result = classify_with_confidence(company_name=company_name, website=website, address=address, business_hint=business_hint)
    return result.business_type

