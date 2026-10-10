"""Rule + layout based field extraction from OCR lines (no LLM).

Pipeline per card:
  1. Merge OCR boxes that sit on the same visual line, split on "|" / "•".
  2. Pull out *contact* tokens with high-precision patterns:
       email   -> regex (+ OCR fixes for "(at)", spaces, commas)
       phone   -> label-aware segmentation (Tel/Mob/Fax/电话/...) + phonenumbers
       website / social handles -> recognised and removed so they can't become
       a name or an address.
  3. Score the remaining lines as job title / company / address / person name
     using keyword lexicons AND layout: relative font height, position, being
     directly next to a job title, and agreement with the email's local part.
  4. Build the address from address-like seed lines plus adjacent continuation
     lines (city / postcode / country), in reading order.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import phonenumbers

from config import settings
from ocr.types import ExtractedFields, OCRLine
from services.validation_service import (
    EMAIL_RE,
    clean_address,
    clean_name,
    clean_text,
    format_phone,
    normalize_email,
    phone_digits,
)

# --------------------------------------------------------------------------- lexicons

JOB_WORDS = {
    "manager", "director", "ceo", "cto", "cfo", "coo", "cmo", "cio", "founder", "co-founder", "cofounder",
    "president", "vice", "vp", "chairman", "chairperson", "chair", "partner", "owner", "proprietor",
    "engineer", "engr", "developer", "designer", "architect", "consultant", "advisor", "adviser",
    "specialist", "executive", "officer", "head", "lead", "leader", "assistant", "associate", "analyst",
    "coordinator", "supervisor", "representative", "rep", "agent", "sales", "marketing", "accountant",
    "administrator", "admin", "secretary", "treasurer", "principal", "attorney", "lawyer", "advocate",
    "doctor", "physician", "surgeon", "dentist", "professor", "lecturer", "teacher", "trainer",
    "merchandiser", "buyer", "purchaser", "purchasing", "procurement", "logistics", "operations",
    "technician", "intern", "trainee", "staff", "managing", "general", "senior", "junior", "chief",
    "deputy", "regional", "country", "area", "business", "development", "account", "key", "export",
    "import", "hr", "human", "resources", "finance", "financial", "product", "project", "programme",
    "program", "quality", "qa", "qc", "it", "support", "service", "customer", "relations", "public",
    "creative", "art", "editor", "photographer", "broker", "realtor", "pharmacist", "nurse", "md", "gm",
    "dgm", "agm", "avp", "svp", "evp", "bdm", "ceo/founder", "inspector", "auditor", "controller",
}
# Words that on their own are strongly a title (not just a modifier like "senior").
JOB_CORE = {
    "manager", "director", "ceo", "cto", "cfo", "coo", "cmo", "cio", "founder", "co-founder", "cofounder",
    "president", "chairman", "chairperson", "partner", "owner", "proprietor", "engineer", "engr", "developer",
    "designer", "architect", "consultant", "advisor", "adviser", "specialist", "executive", "officer",
    "assistant", "analyst", "coordinator", "supervisor", "representative", "accountant", "administrator",
    "secretary", "attorney", "lawyer", "advocate", "professor", "lecturer", "merchandiser", "buyer",
    "technician", "intern", "trainee", "auditor", "controller", "inspector", "editor", "photographer",
    "broker", "realtor", "pharmacist", "gm", "dgm", "agm", "avp", "svp", "evp", "bdm", "head", "lead",
}
JOB_CJK = ("经理", "总监", "总经理", "董事长", "董事", "主任", "主管", "销售", "工程师", "总裁", "副总",
           "经理人", "业务员", "业务", "厂长", "会计", "助理", "代表", "顾问", "部长", "科长", "采购", "外贸",
           "设计师", "秘书", "总经办", "执行董事", "首席")

COMPANY_WORDS = {
    "ltd", "limited", "llc", "l.l.c", "inc", "incorporated", "corp", "corporation", "co", "company", "gmbh",
    "ag", "sa", "s.a", "srl", "spa", "plc", "pvt", "pte", "bv", "nv", "oy", "ab", "kg", "llp", "lp", "pty",
    "group", "industries", "industry", "industrial", "technologies", "technology", "tech", "solutions",
    "trading", "traders", "enterprises", "enterprise", "holdings", "holding", "international", "intl",
    "global", "systems", "services", "consulting", "consultants", "associates", "partners", "agency",
    "studio", "labs", "lab", "media", "logistics", "manufacturing", "manufacturer", "factory", "mills",
    "textile", "textiles", "garments", "apparel", "exports", "imports", "exporters", "importers", "bank",
    "hospital", "clinic", "university", "college", "institute", "school", "foundation", "association",
    "network", "networks", "software", "digital", "marketing", "pharma", "pharmaceuticals", "motors",
    "electronics", "electric", "engineering", "construction", "builders", "developers", "realty",
    "properties", "estate", "foods", "food", "restaurant", "hotel", "resort", "travel", "tours", "airlines",
    "insurance", "investments", "capital", "ventures", "fund", "law", "firm", "chambers", "store", "stores",
    "mart", "shop", "boutique", "co.,ltd", "co.ltd", "s.a.r.l", "sarl", "kk", "k.k", "bhd", "sdn", "tbk",
}
COMPANY_CJK = ("有限公司", "公司", "集团", "厂", "工厂", "企业", "实业", "贸易", "科技", "股份", "商行", "研究所", "银行")

ADDRESS_STRONG = {
    "street", "st", "str", "strasse", "straße", "road", "rd", "avenue", "ave", "av", "boulevard", "blvd",
    "lane", "ln", "drive", "dr", "way", "highway", "hwy", "suite", "ste", "floor", "fl", "flr", "room", "rm",
    "unit", "building", "bldg", "bld", "tower", "block", "blk", "plaza", "square", "sq", "district", "dist",
    "province", "prov", "zip", "postcode", "p.o", "po", "box", "sector", "phase", "town", "township",
    "village", "county", "state", "city", "estate", "park", "industrial", "zone", "complex", "center",
    "centre", "mall", "market", "bazaar", "colony", "society", "nagar", "chowk", "plot", "house", "flat",
    "apartment", "apt", "near", "opp", "opposite", "behind", "main", "cross", "circle", "place", "pl",
    "court", "ct", "terrace", "parkway", "pkwy", "route", "rte", "via", "calle", "rue", "platz", "weg",
    "no", "no.", "lot", "km", "mile", "gate", "wing", "level", "lvl", "office", "shop", "stall", "hall",
}
ADDRESS_CJK = ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "楼", "室", "层", "大厦", "大楼",
               "广场", "中心", "园", "工业区", "开发区", "邮编", "栋", "座", "单元")
COUNTRIES = {
    "pakistan", "china", "p.r.china", "prc", "usa", "u.s.a", "united states", "america", "uk", "u.k",
    "united kingdom", "england", "scotland", "wales", "ireland", "uae", "u.a.e", "united arab emirates",
    "dubai", "saudi arabia", "ksa", "qatar", "oman", "kuwait", "bahrain", "germany", "deutschland", "france",
    "italy", "spain", "portugal", "netherlands", "belgium", "switzerland", "austria", "sweden", "norway",
    "denmark", "finland", "poland", "russia", "turkey", "türkiye", "india", "bangladesh", "sri lanka",
    "nepal", "afghanistan", "iran", "iraq", "egypt", "nigeria", "kenya", "south africa", "morocco",
    "canada", "mexico", "brazil", "argentina", "chile", "colombia", "peru", "australia", "new zealand",
    "japan", "korea", "south korea", "singapore", "malaysia", "indonesia", "thailand", "vietnam",
    "philippines", "hong kong", "hongkong", "taiwan", "macau", "jordan", "lebanon", "israel", "greece",
}
NOT_A_NAME = {
    "welcome", "thank", "thanks", "contact", "contacts", "head office", "branch", "since", "quality",
    "business card", "visiting card", "www", "http", "email", "e-mail", "phone", "mobile", "fax", "address",
    "website", "web", "tel", "office", "follow us", "scan", "qr", "logo", "est", "established", "member",
    "certified", "iso", "the", "and", "of", "for", "in", "at", "by", "with", "your", "our", "best",
}
HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "eng", "engr", "sir", "madam", "mx", "haji", "hajji",
              "sheikh", "shaikh", "syed", "sayed", "mian", "rana", "ch", "chaudhry"}
POSTNOMINALS = {"mba", "phd", "ph.d", "cpa", "acca", "aca", "fca", "pmp", "msc", "bsc", "ma", "ba", "md", "mbbs",
                "llb", "llm", "jr", "sr", "ii", "iii", "iv", "esq", "cfa", "frcs", "pe", "p.eng"}

# --------------------------------------------------------------------------- patterns

_PHONE_LABEL = re.compile(
    r"(?i)(?<![A-Za-z])("
    r"tel(?:ephone)?\s*/\s*fax|tel(?:ephone)?|phone|ph|m[o0]b(?:ile)?|cell(?:ular)?|hand\s*phone|hp|direct|dir|"
    r"office|off|landline|whats\s*app|wa|uan|hotline|toll\s*free|fax|telefax|"
    r"[tmpfodwc]|电话|手机|传真|座机|移动电话|联系电话"
    r")\s*[:.：#]?\s*(?=[+(\d\-–])"
)
_FAX_LABELS = {"fax", "telefax", "f", "传真"}
_PHONE_CAND = re.compile(r"(?:\+|00)?\s*\(?\+?\d[\d\s().\-]{5,}\d(?:\s*(?:ext|x|ext\.)\s*\d{1,5})?", re.I)
_DATE_LIKE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")
_TLDS = (r"com|net|org|biz|info|io|co|cn|pk|uk|us|de|ae|in|hk|tw|jp|kr|sg|au|ca|fr|it|es|nl|ru|tr|sa|qa|my|"
         r"th|vn|id|ph|eu|me|tv|app|dev|tech|store|shop|online|site|xyz|edu|gov|ai|ch|se|no|dk|fi|pl|be|at|nz|za|br|mx")
_WEBSITE = re.compile(
    rf"(?i)(?<![@\w.])(?:https?://|www\.)[^\s,;|]+|(?<![@\w.])[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:{_TLDS})(?:\.[a-z]{{2}})?(?:/[^\s,;|]*)?(?![\w@])"
)
_SOCIAL = re.compile(
    r"(?i)\b(linkedin|facebook|fb\.com|twitter|instagram|youtube|tiktok|wechat|weixin|skype|telegram|"
    r"snapchat|pinterest|line\s*id|qq)\b[^|]*|微信[^|]*|(?<!\S)@[A-Za-z0-9_.]{3,}\b"
)
_LABEL_ONLY = re.compile(
    r"(?i)^\s*(?:e-?mail|email\s*id|mail|e|web(?:site)?|w|url|tel|phone|mob(?:ile)?|cell|fax|m|t|f|p|"
    r"邮箱|电子邮件|网址|电话|手机|传真)\s*[:.：\-]?\s*$"
)
_CONTACT_LABEL_PREFIX = re.compile(r"(?i)(?:^|\s)(?:e-?mail|email\s*id|mail|web(?:site)?|url|邮箱|电子邮件|网址)\s*[:.：\-]?\s*$")
_POSTCODE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"        # UK
    r"|\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"              # US state + zip
    r"|\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"                 # Canada
    r"|\b\d{5,6}\b"                                  # generic 5–6 digit codes
)
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ.'\-]*")
_HAN = re.compile(r"[一-鿿]")


@dataclass
class _Line:
    text: str
    box: tuple[float, float, float, float]
    score: float
    residual: str = ""
    kind: str = ""                     # job | company | address | name | contact | other
    scores: dict = field(default_factory=dict)

    @property
    def h(self) -> float:
        return max(1.0, self.box[3] - self.box[1])

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


# --------------------------------------------------------------------------- public API

def extract_fields(lines: list[OCRLine], image_size: tuple[int, int] | None = None) -> ExtractedFields:
    out = ExtractedFields()
    rows = _build_rows(lines)
    if not rows:
        out.notes.append("no text")
        return out

    emails: list[str] = []
    phones: list[str] = []
    faxes: list[str] = []
    websites: list[str] = []
    # Website hosts seen anywhere on the card, used to repair OCR-damaged emails.
    hosts = {_host(m.group(0)) for r in rows for m in _WEBSITE.finditer(_remove_emails(r.text))} - {""}

    for row in rows:
        text = row.text
        # emails
        found = _find_emails(text)
        if not found:
            repaired = _repair_email(text, hosts)
            if repaired:
                found = [repaired[0]]
                text = text.replace(repaired[1], " ")
                out.notes.append("email reconstructed from website domain - verify")
        for e in found:
            if e not in emails:
                emails.append(e)
        text = _remove_emails(text)
        # websites / social
        for m in _WEBSITE.finditer(text):
            w = m.group(0).strip(".,;")
            if w not in websites and not re.fullmatch(r"(?i)(no|st|rd|co|ltd|inc)\.\w+", w):
                websites.append(w)
        text = _WEBSITE.sub(" ", text)
        # Any token that is (a subdomain of) an email domain is a website too,
        # whatever its TLD (".photo", ".agency", ...).
        for dom in {e.split("@")[1] for e in emails}:
            pat = re.compile(rf"(?i)(?<![\w@.])(?:https?://)?(?:www\.)?{re.escape(dom)}(?:/\S*)?(?![\w.])")
            for m in pat.finditer(text):
                if m.group(0) not in websites:
                    websites.append(m.group(0))
            text = pat.sub(" ", text)
        text = _SOCIAL.sub(" ", text)
        # phones
        text, p, f = _extract_phones(text)
        phones += [x for x in p if x not in phones]
        faxes += [x for x in f if x not in faxes]
        text = _CONTACT_LABEL_PREFIX.sub(" ", text)
        text = clean_text(text)
        row.residual = "" if (_LABEL_ONLY.match(text) or len(re.sub(r"[\W_]", "", text)) < 2) else text

    content = [r for r in rows if r.residual]
    heights = [r.h for r in rows]
    med_h = statistics.median(heights) if heights else 1.0
    width = image_size[0] if image_size else max(r.box[2] for r in rows)
    height = image_size[1] if image_size else max(r.box[3] for r in rows)

    for r in content:
        r.scores = {
            "job": _job_score(r.residual),
            "company": _company_score(r.residual, emails, websites),
            "address": _address_score(r.residual),
        }

    name_row = _pick_name(content, emails, med_h, height)
    job_row = _pick_job(content, name_row)
    company_row = _pick_company(content, name_row, job_row, width)
    if company_row:
        # Brand names are often rendered as a two-line block beside a logo.
        # Keep the lines together instead of selecting a nearby title fragment.
        company_parts = [company_row]
        for candidate in content:
            if candidate in (name_row, job_row, company_row):
                continue
            gap = candidate.box[1] - company_row.box[3]
            aligned = abs(candidate.box[0] - company_row.box[0]) <= max(8.0, width * 0.06)
            if (aligned and 0 <= gap <= max(company_row.h, candidate.h) * 1.8
                    and candidate.scores["company"] >= 1.0
                    and candidate.scores["company"] >= candidate.scores["address"]):
                company_parts.append(candidate)
        if len(company_parts) > 1:
            company_parts.sort(key=lambda r: r.box[1])
            company_row.residual = " ".join(r.residual for r in company_parts)

    address_rows = _pick_address(content, {id(name_row), id(job_row), id(company_row)}, med_h, width)

    out.name = clean_name(name_row.residual) if name_row else ""
    out.job_title = clean_text(job_row.residual) if job_row else ""
    out.company = clean_text(company_row.residual) if company_row else ""
    out.address = clean_address(", ".join(r.residual for r in address_rows))
    out.email = "; ".join(emails)
    out.telephone = "; ".join(phones)
    out.fax = "; ".join(faxes)
    out.website = "; ".join(websites)

    for label, value in (("name", out.name), ("address", out.address), ("telephone", out.telephone),
                         ("email", out.email)):
        if not value:
            out.notes.append(f"missing {label}")
    return out


# --------------------------------------------------------------------------- rows

def _build_rows(lines: list[OCRLine]) -> list[_Line]:
    """Group OCR boxes into visual lines (the detector sometimes returns single
    words), then split a line where a wide gap indicates separate columns."""
    items = sorted((l for l in lines if l.text and l.text.strip()), key=lambda l: l.box[1])
    if not items:
        return []

    rows: list[dict] = []   # {"items": [...], "y0": float, "y1": float}
    for l in items:
        best, best_overlap = None, 0.0
        for row in rows:
            rh = row["y1"] - row["y0"]
            overlap = min(l.box[3], row["y1"]) - max(l.box[1], row["y0"])
            if overlap < 0.55 * max(1.0, min(l.height, rh)):
                continue
            if not 0.6 <= l.height / max(1.0, rh) <= 1.65:
                continue
            if overlap > best_overlap:
                best, best_overlap = row, overlap
        if best is None:
            rows.append({"items": [l], "y0": l.box[1], "y1": l.box[3]})
        else:
            best["items"].append(l)
            n = len(best["items"])  # running mean keeps the row band stable
            best["y0"] += (l.box[1] - best["y0"]) / n
            best["y1"] += (l.box[3] - best["y1"]) / n

    out: list[_Line] = []
    for row in rows:
        its = sorted(row["items"], key=lambda l: l.box[0])
        h = statistics.median(l.height for l in its)
        segments: list[list[OCRLine]] = [[its[0]]]
        for prev, cur in zip(its, its[1:]):
            if cur.box[0] - prev.box[2] > 3.0 * h:   # column gap
                segments.append([])
            segments[-1].append(cur)
        for seg in segments:
            text = seg[0].text.strip()
            for prev, cur in zip(seg, seg[1:]):
                joiner = "" if _HAN.search(prev.text[-1:]) and _HAN.search(cur.text[:1]) else " "
                text += joiner + cur.text.strip()
            box = (min(l.box[0] for l in seg), min(l.box[1] for l in seg),
                   max(l.box[2] for l in seg), max(l.box[3] for l in seg))
            score = sum(l.score for l in seg) / len(seg)
            # Split rows that pack several fields: "Tel: ... | Email: ..."
            for p in re.split(r"\s+[|•·]\s+|\s*\|\s*", text):
                if p.strip():
                    out.append(_Line(text=p.strip(), box=box, score=score))
    out.sort(key=lambda r: (r.box[1], r.box[0]))
    return out


# --------------------------------------------------------------------------- contacts

def _find_emails(text: str) -> list[str]:
    found = []
    t = unicodedata_fix(text)
    if "@" not in t:
        return found
    # Collapse spaces OCR inserts around '@' and '.'
    t2 = re.sub(r"\s*@\s*", "@", t)
    t2 = re.sub(r"(?<=[\w])\s*\.\s*(?=[\w])", ".", t2)
    for m in EMAIL_RE.finditer(t2):
        e = normalize_email(m.group(0))
        if e:
            local, _, domain = e.rpartition("@")
            found.append(f"{local}@{domain.lower()}")
    return found


def unicodedata_fix(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s*[\(\[\{]\s*at\s*[\)\]\}]\s*", "@", t, flags=re.I)
    return t


_EMAIL_LABEL = re.compile(r"(?i)(?:^|\s)(?:e-?mail(?:\s*id)?|mail|e|邮箱|电子邮件)\s*[:.：]\s*(\S+)")


def _host(url: str) -> str:
    h = re.sub(r"(?i)^(https?://)?(www\.)?", "", url.strip()).split("/")[0].lower().strip(".")
    return h if "." in h else ""


def _repair_email(text: str, hosts: set[str]) -> tuple[str, str] | None:
    """Rebuild an email whose '@' or domain was damaged by OCR, using the card's
    website host: "ahmedegulfhorizon.ae" -> ahmed@gulfhorizon.ae,
    "m.smith@acme." -> m.smith@acme.com. Returns (email, damaged token)."""
    if not hosts:
        return None
    m = _EMAIL_LABEL.search(unicodedata_fix(text))
    if not m:
        return None
    token = m.group(1)
    low = token.lower().strip(".,;")
    for host in sorted(hosts, key=len, reverse=True):
        if "@" in low:
            local, dom = low.split("@", 1)
            dom = dom.rstrip(".")
            if local and dom and dom != host and host.startswith(dom):
                return f"{local}@{host}", token
        elif low.endswith(host) and len(low) > len(host) + 1:
            local = low[: -len(host) - 1]  # drop the character OCR produced instead of '@'
            if re.fullmatch(r"[a-z0-9][a-z0-9._%+\-]*", local):
                return f"{local}@{host}", token
    return None


def _remove_emails(text: str) -> str:
    t = unicodedata_fix(text)
    t = re.sub(r"\s*@\s*", "@", t)
    t = re.sub(r"(?<=[\w])\s*\.\s*(?=[\w])(?=[^@\s]*@|[^\s]*\.[A-Za-z]{2,})", ".", t)
    return EMAIL_RE.sub(" ", t)


def _fix_digit_confusions(s: str) -> str:
    s = re.sub(r"(?<=[\d\s\-()+.])[OoQD](?=[\d\s\-().])", "0", s)
    s = re.sub(r"(?<=[\d\s\-()+.])[lI|](?=[\d\s\-().])", "1", s)
    s = re.sub(r"(?<=\d)[Ss](?=\d)", "5", s)
    s = re.sub(r"(?<=\d)B(?=\d)", "8", s)
    return s


def _extract_phones(text: str) -> tuple[str, list[str], list[str]]:
    """Return (text with phone segments removed, phones, faxes)."""
    phones: list[str] = []
    faxes: list[str] = []
    spans: list[tuple[int, int]] = []
    labels = list(_PHONE_LABEL.finditer(text))

    # Segment boundaries: each label owns the text until the next label.
    segments: list[tuple[int, int, str | None]] = []
    if labels:
        if labels[0].start() > 0:
            segments.append((0, labels[0].start(), None))
        for i, m in enumerate(labels):
            end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
            segments.append((m.start(), end, m.group(1).lower()))
    else:
        segments.append((0, len(text), None))

    for start, end, label in segments:
        seg = text[start:end]
        fixed = _fix_digit_confusions(seg) if (label or "+" in seg) else seg
        for m in _PHONE_CAND.finditer(fixed):
            cand = m.group(0).strip()
            # "Tel: -971 4 ..." — OCR read '+' as a dash. Accept only if it then forms a valid number.
            if label and m.start() > 0 and fixed[m.start() - 1] in "-–—~" and not cand.startswith("+"):
                try:
                    if phonenumbers.is_valid_number(phonenumbers.parse("+" + phone_digits(cand), None)):
                        cand = "+" + cand
                except phonenumbers.NumberParseException:
                    pass
            for piece in re.split(r"\s+/\s+|\s*/\s*(?=\+|0)", cand):
                if not _is_phone(piece, labelled=label is not None):
                    continue
                formatted = format_phone(piece)
                target = faxes if label and re.sub(r"\s", "", label) in _FAX_LABELS else phones
                if all(phone_digits(formatted)[-9:] != phone_digits(x)[-9:] for x in phones + faxes):
                    target.append(formatted)
                spans.append((start, end) if label else (start + m.start(), start + m.end()))

    if not spans:
        return text, phones, faxes
    keep = []
    pos = 0
    for s, e in sorted(spans):
        if s >= pos:
            keep.append(text[pos:s])
            pos = e
        else:
            pos = max(pos, e)
    keep.append(text[pos:])
    return " ".join(keep), phones, faxes


def _is_phone(cand: str, labelled: bool) -> bool:
    c = cand.strip()
    digits = phone_digits(c)
    if not 7 <= len(digits) <= 15:
        return False
    if _DATE_LIKE.match(c):
        return False
    intl = c.startswith("+") or c.startswith("00")
    if intl:
        try:
            num = phonenumbers.parse(re.sub(r"^00", "+", c), None)
            return phonenumbers.is_possible_number(num)
        except phonenumbers.NumberParseException:
            return len(digits) >= 8
    if settings.default_phone_region:
        try:
            num = phonenumbers.parse(c, settings.default_phone_region)
            if phonenumbers.is_possible_number(num):
                return True
        except phonenumbers.NumberParseException:
            pass
    # Unlabelled bare numbers need to look like phones, not house/postcodes.
    return labelled or len(digits) >= 9


# --------------------------------------------------------------------------- scoring

def _words(text: str) -> list[str]:
    return [w.lower().strip(".'-") for w in _WORD.findall(text)]


def _job_score(text: str) -> float:
    ws = _words(text)
    score = sum(1.0 for w in ws if w in JOB_CORE) + sum(0.3 for w in ws if w in JOB_WORDS and w not in JOB_CORE)
    score += sum(1.0 for k in JOB_CJK if k in text)
    if ws and len(ws) > 7:
        score *= 0.5
    return score


def _company_score(text: str, emails: list[str], websites: list[str]) -> float:
    ws = _words(text)
    low = text.lower()
    score = sum(1.5 for w in ws if w in COMPANY_WORDS)
    if re.search(r"(?i)\b(co\.,?\s*ltd|pvt\.?\s*ltd|sdn\.?\s*bhd|s\.?a\.?r\.?l|l\.?l\.?c)\b", low):
        score += 2
    score += sum(2.0 for k in COMPANY_CJK if k in text)
    if "&" in text:
        score += 0.5
    # Company names usually echo the email/web domain ("acme" in info@acme.com).
    stems = {_domain_stem(e.split("@")[-1]) for e in emails} | {_domain_stem(w) for w in websites}
    stems.discard("")
    generic = {"gmail", "yahoo", "hotmail", "outlook", "163", "126", "qq", "live", "icloud", "aol", "sina",
               "foxmail", "mail", "protonmail", "yandex"}
    compact = re.sub(r"[^a-z0-9]", "", low)
    for s in stems - generic:
        if len(s) >= 3 and s in compact:
            score += 2
    return score


def _domain_stem(domain: str) -> str:
    d = re.sub(r"(?i)^(https?://)?(www\.)?", "", domain).split("/")[0].lower()
    parts = d.split(".")
    if len(parts) >= 3 and parts[-2] in {"com", "co", "net", "org", "gov", "edu", "ac"}:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else parts[0]


def _address_score(text: str) -> float:
    low = text.lower()
    ws = _words(text)
    score = 0.0
    if re.match(r"(?i)^\s*(address|addr|add|adresse|location|地址)\s*[:.：\-]", text):
        score += 4
    score += sum(1.5 for w in ws if w in ADDRESS_STRONG and w not in {"no", "main", "office", "house", "center",
                                                                      "centre", "park", "place", "way", "dr"})
    score += sum(0.5 for w in ws if w in {"no", "main", "office", "house", "center", "centre", "park", "place"})
    score += sum(1.5 for k in ADDRESS_CJK if k in text)
    if re.search(r"(?i)\bno\.?\s*\d", text) or re.search(r"\b\d{1,5}[A-Za-z]?\s*[,/-]?\s+[A-Za-z]", text):
        score += 1.0
    # European style "Via Roma 45", "Industriestraße 14", "Rua Augusta 120"
    if re.search(r"[A-Za-zÀ-ÿß]{3,}\.?\s+\d{1,5}[a-zA-Z]?\s*(?:,|$)", text):
        score += 1.0
    if re.search(r"(?i)(stra(?:ß|ss)e|str\.|weg|platz|gasse|allee|damm|laan|straat|gatan|vej)\b", text) or \
            re.search(r"(?i)\b(rua|calle|via|viale|avenida|avenue|piazza|plaza|rue|chemin|carrer)\b", text):
        score += 1.5
    if _POSTCODE.search(text):
        score += 1.5
    # A country alone is weak address evidence: company names often contain
    # country names (for example, "Engro Pakistan Limited").
    if any(re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", low) for c in COUNTRIES):
        score += 1.0
    if text.count(",") >= 2:
        score += 0.75
    return score


def _name_score(r: _Line, emails: list[str], med_h: float, img_h: float, rows: list[_Line]) -> float:
    t = r.residual.strip()
    if not t or "@" in t or re.search(r"\d", t) or len(t) > 45:
        return -99
    han = _HAN.findall(t)
    score = 0.0
    if han:
        han_part = "".join(han)
        latin_part = _HAN.sub(" ", t).strip()
        if not 2 <= len(han_part) <= 4 or re.search(r"[^一-鿿\sA-Za-z.'\-]", t):
            return -99
        if any(k in han_part for k in JOB_CJK + COMPANY_CJK + ADDRESS_CJK):
            return -99
        # "王伟" alone, or bilingual "王伟 Wang Wei"
        if latin_part and not re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){0,2}", latin_part):
            return -99
        score += 2.5 + (0.5 if latin_part else 0.0)
    else:
        tokens = [w for w in re.split(r"[\s,]+", t) if w]
        words = [w for w in tokens if w.lower().strip(".") not in POSTNOMINALS]
        if not 1 <= len(words) <= 5:
            return -99
        if any(not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ.'\-]*\.?", w) for w in words):
            return -99
        core = [w for w in words if w.lower().strip(".") not in HONORIFICS]
        if len(core) >= 2:
            score += 2.0
        elif len(core) == 1 and len(core[0].strip(".")) >= 4 and not core[0].isupper():
            score -= 0.5   # single-word names exist but are rare; only win when nothing else fits
        else:
            return -99     # e.g. logo initials "NS", "ABC"
        if len(core) >= 5:
            score -= 1.0
        if all(w[0].isupper() for w in core):
            score += 1.0
        if any(w.lower().strip(".") in HONORIFICS for w in words[:1]):
            score += 1.5
        if any(len(w.strip(".")) == 1 for w in core) and len(core) <= 1:
            score -= 1.0
        low = t.lower()
        if low in NOT_A_NAME or any(re.search(rf"\b{re.escape(n)}\b", low) for n in NOT_A_NAME if " " in n):
            score -= 3.0
        stop_hits = sum(1 for w in core if w.lower() in NOT_A_NAME)
        score -= 1.5 * stop_hits

    score -= 2.5 * r.scores.get("job", 0)
    score -= 2.0 * r.scores.get("company", 0)
    score -= 1.5 * r.scores.get("address", 0)

    # Larger font than the body text is a strong hint.
    rel = r.h / max(1.0, med_h)
    score += max(-1.0, min(2.0, (rel - 1.0) * 2.0))

    # Directly above/below a job title.
    for o in rows:
        if o is r or o.scores.get("job", 0) < 1:
            continue
        vgap = min(abs(o.box[1] - r.box[3]), abs(r.box[1] - o.box[3]))
        h_overlap = min(o.box[2], r.box[2]) - max(o.box[0], r.box[0])
        aligned = h_overlap > 0 or abs(o.box[0] - r.box[0]) < 2 * med_h
        if vgap < 1.8 * max(r.h, o.h) and aligned:
            score += 2.5
            break

    # Agreement with the email local part (john.smith@ -> John Smith).
    tokens = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", t)]
    for e in emails:
        local = re.sub(r"[^a-z]", "", e.split("@")[0].lower())
        if not local or local in {"info", "sales", "contact", "admin", "office", "support", "hello", "enquiry",
                                  "enquiries", "export", "marketing", "service", "mail", "hr", "accounts"}:
            continue
        hits = sum(1 for w in tokens if len(w) >= 3 and w in local)
        initials = "".join(w[0] for w in tokens)
        if hits >= 1 or (len(initials) >= 2 and local.startswith(initials[:-1]) and tokens[-1] in local):
            score += 3.0 if hits >= 2 else 2.0
            break

    # Slight preference for the upper part of the card.
    if img_h and r.cy < img_h * 0.6:
        score += 0.3
    return score


def _pick_name(rows: list[_Line], emails: list[str], med_h: float, img_h: float) -> _Line | None:
    best, best_score = None, 1.5
    for r in rows:
        s = _name_score(r, emails, med_h, img_h, rows)
        r.scores["name"] = s
        if s > best_score:
            best, best_score = r, s
    if best:
        best.kind = "name"
    return best


def _pick_job(rows: list[_Line], name_row: _Line | None) -> _Line | None:
    cands = [r for r in rows if r is not name_row and r.scores["job"] >= 1 and r.scores["address"] < 2]
    if not cands:
        return None
    if name_row:  # the title closest to the name
        cands.sort(key=lambda r: (abs(r.cy - name_row.cy), -r.scores["job"]))
    else:
        cands.sort(key=lambda r: -r.scores["job"])
    cands[0].kind = "job"
    return cands[0]


def _pick_company(rows: list[_Line], name_row, job_row, width: float) -> _Line | None:
    cands = [r for r in rows if r not in (name_row, job_row) and r.scores["company"] >= 1.5
             and r.scores["company"] >= r.scores["address"]]
    if not cands:
        return None
    # A title can contain company-looking words ("Information Technology").
    # Prefer a compact, upper-case brand block away from the person's title.
    def key(r: _Line) -> tuple[float, float, float]:
        letters = [c for c in r.residual if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in letters) / max(1, len(letters))
        right_block = 0.5 if width and r.box[0] > width * 0.60 else 0.0
        return (r.scores["company"] + upper_ratio + right_block, r.h, -r.box[1])

    best = max(cands, key=key)
    best.kind = "company"
    return best


def _pick_address(rows: list[_Line], exclude: set[int], med_h: float, width: float) -> list[_Line]:
    pool = [r for r in rows if id(r) not in exclude]
    seeds = [r for r in pool if r.scores["address"] >= 2.0 and r.scores["address"] > r.scores["company"]]
    if not seeds:
        return []
    chosen = {id(r): r for r in seeds}
    changed = True
    while changed:  # grow with adjacent continuation lines (city / postcode / country)
        changed = False
        for r in pool:
            if id(r) in chosen or r.scores["job"] >= 1 or r.scores["company"] >= 2 or r.scores.get("name", -99) > 3:
                continue
            for s in list(chosen.values()):
                vgap = min(abs(r.box[1] - s.box[3]), abs(s.box[1] - r.box[3]))
                left_aligned = abs(r.box[0] - s.box[0]) < 0.08 * width
                h_overlap = min(r.box[2], s.box[2]) - max(r.box[0], s.box[0])
                if vgap < 1.6 * max(med_h, min(r.h, s.h)) and (left_aligned or h_overlap > 0.3 * min(
                        r.box[2] - r.box[0], s.box[2] - s.box[0])) and r.scores["address"] >= 0.5:
                    chosen[id(r)] = r
                    changed = True
                    break
    ordered = sorted(chosen.values(), key=lambda r: (r.box[1], r.box[0]))
    for r in ordered:
        r.kind = "address"
    # Drop duplicate fragments (same text twice from overlapping boxes).
    seen, out = set(), []
    for r in ordered:
        key = re.sub(r"\W", "", r.residual.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out
