import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 尝试引入 sentence-transformers（用于语义相似度）
try:
    from sentence_transformers import SentenceTransformer, util

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@st.cache_resource
def get_nli_model():
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


# ========================
# 本地模型配置（LM Studio）
# ========================
LOCAL_API_BASE = "http://127.0.0.1:1234"
LOCAL_CHAT_ENDPOINT = f"{LOCAL_API_BASE}/v1/chat/completions"
LOCAL_MODEL_NAME = "phi-3-mini-4k-instruct"
LOCAL_API_KEY = "lm-studio"  # LM Studio 默认不校验


# ========================
# 默认参考知识（可被用户 reference/context 覆盖）
# ========================
DEFAULT_REFERENCE_BY_SCENARIO = {
    "voltage_stability_interpretation": """
Voltage stability is the ability of a power system to maintain acceptable voltage levels under normal
conditions and after disturbances. It is strongly related to reactive power support, transfer limits,
PV/PQ transitions, and risks of progressive voltage decline leading to voltage collapse.
""".strip(),
    "fault_analysis_protection": """
Fault analysis in power systems requires identifying fault type, location, and protection behavior.
Protection recommendations should follow relay coordination, selective tripping principles,
and operating procedures to avoid unsafe misoperations.
""".strip(),
    "dispatch_scheduling_explanation": """
Dispatch and scheduling should balance load demand, generator constraints, reserve requirements,
and security constraints while minimizing cost and preserving reliability.
""".strip(),
    "operational_recommendation": """
Operational recommendations must prioritize grid safety, procedural compliance, and risk-controlled
execution with verification and operator coordination.
""".strip(),
    "literature_review": """
Literature review responses should provide verifiable references, avoid fabricated citations,
and present methods, limitations, and evidence in a traceable way.
""".strip(),
    "technical_decision_support": """
Engineering decision support responses should be factual, internally consistent, transparent about
assumptions, and explicit about uncertainties and operational safeguards.
""".strip(),
}

DEFAULT_SCENARIOS = {
    "voltage_stability_interpretation": {
        "display_name": "Voltage Stability Interpretation",
        "evaluation_focus": ["factual_reliability", "internal_consistency", "operational_safety_risk"],
        "weights": {
            "factual_reliability": 0.30,
            "task_alignment": 0.16,
            "internal_consistency": 0.20,
            "interpretability_reviewability": 0.10,
            "unsupported_content_risk": 0.14,
            "operational_safety_risk": 0.10,
        },
        "citation_strictness": "medium",
        "numeric_profile": "default",
        "safety_profile": "operations",
    },
    "fault_analysis_protection": {
        "display_name": "Fault Analysis / Protection Explanation",
        "evaluation_focus": ["factual_reliability", "operational_safety_risk", "internal_consistency"],
        "weights": {
            "factual_reliability": 0.26,
            "task_alignment": 0.14,
            "internal_consistency": 0.18,
            "interpretability_reviewability": 0.08,
            "unsupported_content_risk": 0.14,
            "operational_safety_risk": 0.20,
        },
        "citation_strictness": "medium",
        "numeric_profile": "default",
        "safety_profile": "protection",
    },
    "dispatch_scheduling_explanation": {
        "display_name": "Dispatch / Scheduling Explanation",
        "evaluation_focus": ["factual_reliability", "task_alignment", "operational_safety_risk"],
        "weights": {
            "factual_reliability": 0.28,
            "task_alignment": 0.18,
            "internal_consistency": 0.17,
            "interpretability_reviewability": 0.10,
            "unsupported_content_risk": 0.12,
            "operational_safety_risk": 0.15,
        },
        "citation_strictness": "low",
        "numeric_profile": "dispatch",
        "safety_profile": "operations",
    },
    "operational_recommendation": {
        "display_name": "Power-System Operational Recommendation",
        "evaluation_focus": ["operational_safety_risk", "factual_reliability", "unsupported_content_risk"],
        "weights": {
            "factual_reliability": 0.24,
            "task_alignment": 0.14,
            "internal_consistency": 0.16,
            "interpretability_reviewability": 0.08,
            "unsupported_content_risk": 0.14,
            "operational_safety_risk": 0.24,
        },
        "citation_strictness": "low",
        "numeric_profile": "operations",
        "safety_profile": "operations",
    },
    "literature_review": {
        "display_name": "Literature Review in Power Systems",
        "evaluation_focus": ["unsupported_content_risk", "factual_reliability", "task_alignment"],
        "weights": {
            "factual_reliability": 0.24,
            "task_alignment": 0.14,
            "internal_consistency": 0.14,
            "interpretability_reviewability": 0.08,
            "unsupported_content_risk": 0.28,
            "operational_safety_risk": 0.12,
        },
        "citation_strictness": "high",
        "numeric_profile": "default",
        "safety_profile": "general",
    },
    "technical_decision_support": {
        "display_name": "Technical Explanation for Engineering Decision Support",
        "evaluation_focus": ["factual_reliability", "internal_consistency", "operational_safety_risk"],
        "weights": {
            "factual_reliability": 0.28,
            "task_alignment": 0.16,
            "internal_consistency": 0.18,
            "interpretability_reviewability": 0.12,
            "unsupported_content_risk": 0.12,
            "operational_safety_risk": 0.14,
        },
        "citation_strictness": "medium",
        "numeric_profile": "default",
        "safety_profile": "operations",
    },
}

DEFAULT_NUMERIC_CONSTRAINTS = {
    "default": {"hz": [45.0, 65.0], "kv": [0.1, 1000.0], "mw": [0.1, 100000.0], "mvar": [0.1, 100000.0], "pu": [0.5, 1.5]},
    "dispatch": {"hz": [49.0, 51.0], "kv": [0.1, 1000.0], "mw": [1.0, 20000.0], "mvar": [0.1, 5000.0], "pu": [0.85, 1.15]},
    "operations": {"hz": [49.0, 51.0], "kv": [0.1, 1000.0], "mw": [0.1, 100000.0], "mvar": [0.1, 100000.0], "pu": [0.85, 1.15]},
}

DEFAULT_SAFETY_RULES = {
    "general": {
        "action_verbs": ["immediately", "must", "directly", "bypass", "force", "override", "disable", "立刻", "必须", "直接", "强制"],
        "procedure_terms": ["verify", "approval", "dispatch", "protection", "procedure", "contingency", "risk", "caution", "确认", "复核", "规程", "风险"],
        "unsafe_patterns": ["disable protection", "ignore alarm", "skip verification", "bypass relay", "关闭保护", "忽略告警", "跳过校验"],
    },
    "operations": {
        "action_verbs": ["immediately", "must", "force", "override", "trip", "shed", "立刻", "必须", "强制", "切负荷"],
        "procedure_terms": ["dispatch center", "operator", "approval", "contingency", "n-1", "protection", "risk", "verify", "调度", "规程", "保护"],
        "unsafe_patterns": ["trip without confirmation", "shed load immediately without check", "override interlock", "disconnect protection", "未确认直接跳闸"],
    },
    "protection": {
        "action_verbs": ["disable", "bypass", "block", "force", "关闭", "旁路", "封锁", "强制"],
        "procedure_terms": ["relay setting", "test plan", "two-person check", "protection engineer", "safety permit", "继电保护", "试验", "双人复核"],
        "unsafe_patterns": ["disable relay protection", "bypass differential protection", "ignore fault indication", "关闭继电保护", "旁路差动保护"],
    },
}

DEFAULT_DOMAIN_TERMS = [
    "voltage stability", "frequency stability", "rotor angle stability", "transient stability", "small-signal stability",
    "voltage collapse", "load forecasting", "power flow", "optimal power flow", "state estimation", "unit commitment",
    "economic dispatch", "contingency analysis", "n-1 security", "reactive power", "power factor", "transformer",
    "generator", "AVR", "PSS", "AGC", "fault", "short circuit", "relay protection", "differential protection",
    "load shedding", "HVDC", "FACTS", "SVC", "STATCOM", "PMU", "SCADA", "power quality", "microgrid",
    "voltage regulation", "OLTC", "substation", "电压稳定", "频率稳定", "暂态稳定", "潮流", "最优潮流",
    "状态估计", "机组组合", "经济调度", "无功功率", "功率因数", "继电保护", "短路", "配电网", "输电网",
]

NUMERIC_UNIT_ALIASES = {
    "hz": {"hz"},
    "kv": {"kv", "kV"},
    "mw": {"mw", "MW"},
    "mvar": {"mvar", "MVar", "MVAr", "mVar"},
    "pu": {"pu", "p.u.", "p.u"},
}

CITATION_STYLE_REGEX = re.compile(r"\b[A-Z][a-z]+ et al\.?\b|\b(19|20)\d{2}\b|\[\d{1,3}\]")
CLAIM_HINTS = ["paper", "study", "research", "according to", "et al", "doi", "arxiv", "论文", "文献", "研究", "根据", "结论表明"]
UNSUPPORTED_ASSERTION_HINTS = ["prove", "definitely", "always", "guarantee", "毫无疑问", "一定", "绝对"]

DIMENSION_DESCRIPTIONS = {
    "factual_reliability": "Grounding quality combining semantic consistency, numeric plausibility and citation credibility.",
    "task_alignment": "Alignment with the selected scenario intent and optional task prompt.",
    "internal_consistency": "Internal contradiction and coherence quality of generated statements.",
    "interpretability_reviewability": "Clarity and reviewability for engineering human verification.",
    "unsupported_content_risk": "Risk of unsupported, unverifiable or fabricated technical claims.",
    "operational_safety_risk": "Risk of unsafe or procedure-bypassing operational recommendations.",
}


def _load_json_config(file_name: str, default):
    path = Path("config") / file_name
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_scenarios() -> Dict[str, dict]:
    return _load_json_config("scenarios.json", DEFAULT_SCENARIOS)


@st.cache_data(show_spinner=False)
def load_numeric_constraints() -> Dict[str, dict]:
    return _load_json_config("numeric_constraints.json", DEFAULT_NUMERIC_CONSTRAINTS)


@st.cache_data(show_spinner=False)
def load_safety_rules() -> Dict[str, dict]:
    return _load_json_config("safety_rules.json", DEFAULT_SAFETY_RULES)


@st.cache_data(show_spinner=False)
def load_domain_terms() -> List[str]:
    return _load_json_config("domain_terms.json", DEFAULT_DOMAIN_TERMS)


@st.cache_resource
def get_st_model():
    if not ST_AVAILABLE:
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def extract_domain_terms(text: str, domain_terms: Optional[List[str]] = None) -> List[str]:
    if not text:
        return []
    terms = domain_terms or load_domain_terms()
    lowered = text.lower()
    matched = set()
    for term in terms:
        has_cjk = re.search(r"[\u4e00-\u9fff]", term) is not None
        if has_cjk:
            if term in text:
                matched.add(term)
        else:
            if re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered):
                matched.add(term)
    return sorted(matched)


def score_domain_coverage(text: str, target_terms: int = 8) -> float:
    matched = extract_domain_terms(text)
    if not matched:
        return 0.0
    return min(1.0, len(matched) / target_terms)


def score_semantic_consistency(text: str, reference_text: Optional[str]) -> float:
    model = get_st_model()
    if model is None or not reference_text:
        return 0.5
    embeddings = model.encode([reference_text, text], convert_to_tensor=True)
    sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def extract_numeric_claims(text: str) -> List[Tuple[float, str]]:
    pattern = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kV|kv|Hz|hz|MW|mw|MVar|MVAr|mvar|mVar|p\.u\.|p\.u|pu)")
    claims = []
    for match in pattern.finditer(text):
        value = float(match.group("value"))
        unit = match.group("unit").lower().replace(".", "")
        claims.append((value, unit))
    return claims


def analyze_numeric_claims(text: str, scenario: dict) -> Dict[str, object]:
    constraints_map = load_numeric_constraints()
    profile = scenario.get("numeric_profile", "default")
    ranges = constraints_map.get(profile, constraints_map.get("default", DEFAULT_NUMERIC_CONSTRAINTS["default"]))
    claims = extract_numeric_claims(text)
    if not claims:
        return {"total": 0, "ok": 0, "out_of_range": 0, "issues": [], "score": 0.7, "profile": profile}

    total = 0
    ok = 0
    issues = []
    for value, unit in claims:
        total += 1
        canonical = None
        for key, aliases in NUMERIC_UNIT_ALIASES.items():
            if unit in {a.lower().replace(".", "") for a in aliases}:
                canonical = key
                break
        if canonical is None:
            issues.append(f"{value} {unit} (unit not recognized)")
            continue
        low, high = ranges.get(canonical, DEFAULT_NUMERIC_CONSTRAINTS["default"][canonical])
        if low <= value <= high:
            ok += 1
        else:
            issues.append(f"{value} {canonical} (expected {low}-{high})")

    score = max(0.0, min(1.0, ok / total)) if total else 0.7
    return {
        "total": total,
        "ok": ok,
        "out_of_range": max(0, total - ok),
        "issues": issues,
        "score": score,
        "profile": profile,
    }


def extract_citations(text: str) -> Dict[str, List[str]]:
    doi_pattern = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
    arxiv_pattern = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
    title_pattern = re.compile(r"[\"“”](.+?)[\"“”]")
    dois = list({m.group(0) for m in doi_pattern.finditer(text)})
    arxivs = list({m.group(0) for m in arxiv_pattern.finditer(text)})
    titles = list({m.group(1).strip() for m in title_pattern.finditer(text) if len(m.group(1).strip()) > 6})
    return {"doi": dois, "arxiv": arxivs, "title": titles}


@st.cache_data(show_spinner=False)
def verify_doi(doi: str) -> bool:
    try:
        return requests.get(f"https://api.crossref.org/works/{doi}", timeout=6).status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False)
def verify_arxiv(arxiv_id: str) -> bool:
    try:
        resp = requests.get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}", timeout=6)
        return resp.status_code == 200 and "<entry>" in resp.text
    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False)
def verify_title(title: str) -> bool:
    try:
        resp = requests.get("https://api.crossref.org/works", params={"query.bibliographic": title, "rows": 1}, timeout=6)
        if resp.status_code != 200:
            return False
        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return False
        return items[0].get("score", 0) >= 20
    except (requests.RequestException, ValueError):
        return False


def analyze_citations(text: str, scenario: dict) -> Dict[str, object]:
    strictness = scenario.get("citation_strictness", "medium")
    claim_present = any(h in text.lower() for h in CLAIM_HINTS) or bool(CITATION_STYLE_REGEX.search(text))
    citations = extract_citations(text)
    total = len(citations["doi"]) + len(citations["arxiv"]) + len(citations["title"])

    if total == 0:
        if strictness == "high" and claim_present:
            score = 0.05
        elif claim_present:
            score = 0.15
        else:
            score = 0.7
        return {
            "total": 0,
            "verified": 0,
            "claim_present": claim_present,
            "strictness": strictness,
            "score": score,
            "unverified_items": [],
            "citations": citations,
        }

    verified = 0
    unverified_items = []
    for doi in citations["doi"]:
        if verify_doi(doi):
            verified += 1
        else:
            unverified_items.append(f"DOI not verified: {doi}")
    for arxiv_id in citations["arxiv"]:
        if verify_arxiv(arxiv_id):
            verified += 1
        else:
            unverified_items.append(f"arXiv not verified: {arxiv_id}")
    for title in citations["title"]:
        if verify_title(title):
            verified += 1
        else:
            unverified_items.append(f"Title not verified: {title}")

    ratio = verified / total
    if strictness == "high":
        ratio *= 0.9
    if claim_present and verified == 0:
        ratio *= 0.25

    return {
        "total": total,
        "verified": verified,
        "claim_present": claim_present,
        "strictness": strictness,
        "score": max(0.0, min(1.0, ratio)),
        "unverified_items": unverified_items,
        "citations": citations,
    }


def score_relevance_semantic(text: str, prompt: str) -> Optional[float]:
    model = get_st_model()
    if model is None:
        return None
    embeddings = model.encode([prompt, text], convert_to_tensor=True)
    sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def score_relevance_keyword(text: str, prompt: str) -> float:
    prompt_words = set(prompt.lower().split())
    resp_words = set(text.lower().split())
    if not prompt_words or not resp_words:
        return 0.0
    overlap = len(prompt_words.intersection(resp_words))
    if overlap == 0:
        return 0.0
    if overlap < 5:
        return 0.3
    if overlap < 15:
        return 0.5
    if overlap < 30:
        return 0.7
    return 0.9


def score_task_alignment(text: str, prompt: Optional[str], scenario: dict) -> float:
    domain = score_domain_coverage(text)
    if not prompt:
        return max(0.0, min(1.0, 0.45 + 0.55 * domain))
    kw = score_relevance_keyword(text, prompt)
    sem = score_relevance_semantic(text, prompt)
    if sem is None:
        return max(0.0, min(1.0, 0.7 * kw + 0.3 * domain))
    return max(0.0, min(1.0, 0.35 * kw + 0.5 * sem + 0.15 * domain))


def score_logical(text: str) -> float:
    tokenizer, model = get_nli_model()
    sentences = [s.strip() for s in re.split(r"[.!?。\n]", text) if len(s.strip()) > 5]
    if len(sentences) < 2:
        return 0.5

    contradiction = 0
    pairs = 0
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            pairs += 1
            inputs = tokenizer(sentences[i], sentences[j], return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs).item()  # 0 contradiction / 1 neutral / 2 entailment
            if label == 0:
                contradiction += 1

    if pairs == 0:
        return 0.5
    return max(0.0, min(1.0, 1 - contradiction / pairs))


def score_clarity(text: str) -> float:
    import textstat

    try:
        fk = textstat.flesch_kincaid_grade(text)
        if fk < 7:
            return 0.4
        if fk <= 10:
            return 0.7
        if fk <= 12.5:
            return 0.9
        if fk <= 14:
            return 0.8
        if fk <= 16:
            return 0.65
        if fk <= 18:
            return 0.5
        return 0.3
    except Exception:
        return 0.5


def assess_operational_safety(text: str, scenario: dict) -> Dict[str, object]:
    rules_map = load_safety_rules()
    profile = scenario.get("safety_profile", "general")
    rules = rules_map.get(profile, rules_map.get("general", DEFAULT_SAFETY_RULES["general"]))
    lowered = text.lower()

    evidence = []
    risk_points = 0.0

    unsafe_hits = [p for p in rules.get("unsafe_patterns", []) if p.lower() in lowered]
    if unsafe_hits:
        risk_points += min(0.6, 0.2 * len(unsafe_hits))
        evidence.extend([f"Unsafe pattern detected: {h}" for h in unsafe_hits])

    action_hits = [w for w in rules.get("action_verbs", []) if w.lower() in lowered]
    procedure_hits = [w for w in rules.get("procedure_terms", []) if w.lower() in lowered]
    if action_hits and not procedure_hits:
        risk_points += 0.25
        evidence.append("Action-oriented recommendation without procedural safeguards")

    high_risk_context = any(k in lowered for k in ["fault", "trip", "load shedding", "blackout", "短路", "跳闸", "切负荷"])
    if high_risk_context and not any(k in lowered for k in ["risk", "caution", "verify", "contingency", "谨慎", "风险", "确认"]):
        risk_points += 0.25
        evidence.append("High-risk context without cautionary language")

    overconfident = any(k in lowered for k in ["definitely", "guarantee", "must always", "绝对", "一定"])
    if overconfident:
        risk_points += 0.15
        evidence.append("Overconfident action wording detected")

    risk_score = max(0.0, min(1.0, risk_points))
    return {
        "risk_score": risk_score,
        "profile": profile,
        "unsafe_hits": unsafe_hits,
        "action_hits": action_hits,
        "procedure_hits": procedure_hits,
        "evidence": evidence,
    }


def estimate_unsupported_claim_signal(text: str, citation_analysis: Dict[str, object]) -> float:
    lowered = text.lower()
    unsupported_assertions = sum(1 for h in UNSUPPORTED_ASSERTION_HINTS if h in lowered)
    claim_present = citation_analysis.get("claim_present", False)
    has_verified = citation_analysis.get("verified", 0) > 0

    signal = 0.0
    if claim_present and not has_verified:
        signal += 0.5
    if unsupported_assertions > 0:
        signal += min(0.5, 0.2 * unsupported_assertions)
    return max(0.0, min(1.0, signal))


def score_factual_reliability(semantic: float, numeric_score: float, citation_score: float) -> float:
    return max(0.0, min(1.0, 0.5 * semantic + 0.25 * numeric_score + 0.25 * citation_score))


def score_unsupported_content_risk(citation_score: float, numeric_score: float, unsupported_signal: float) -> float:
    risk = 0.5 * (1 - citation_score) + 0.3 * (1 - numeric_score) + 0.2 * unsupported_signal
    return max(0.0, min(1.0, risk))


def derive_risk_level(overall_risk_score: float) -> str:
    if overall_risk_score < 0.33:
        return "Low"
    if overall_risk_score < 0.66:
        return "Medium"
    return "High"


def build_flagged_evidence_items(
    citation_analysis: Dict[str, object],
    numeric_analysis: Dict[str, object],
    logical_score: float,
    unsupported_risk: float,
    safety_result: Dict[str, object],
    scenario_id: str,
) -> List[Dict[str, str]]:
    items = []

    for issue in citation_analysis.get("unverified_items", []):
        items.append({"severity": "high", "type": "unverified_citation", "detail": issue})

    for issue in numeric_analysis.get("issues", []):
        items.append({"severity": "medium", "type": "suspicious_numeric_value", "detail": issue})

    if logical_score < 0.6:
        items.append({
            "severity": "medium",
            "type": "internal_contradiction_risk",
            "detail": f"Low internal consistency score ({logical_score:.2f}) indicates potential contradiction.",
        })

    if unsupported_risk > 0.6:
        items.append({
            "severity": "high",
            "type": "unsupported_content_risk",
            "detail": f"Unsupported content risk is high ({unsupported_risk:.2f}); human verification is required.",
        })

    for evidence in safety_result.get("evidence", []):
        items.append({"severity": "high", "type": "operational_safety_alert", "detail": evidence})

    if not items:
        items.append({
            "severity": "low",
            "type": "no_major_flag",
            "detail": f"No critical evidence flags detected for scenario '{scenario_id}', but human review is still recommended.",
        })

    return items


def evaluate_response(
    response_text: str,
    scenario_id: str,
    original_prompt: Optional[str] = None,
    reference_context: Optional[str] = None,
) -> Dict[str, object]:
    scenarios = load_scenarios()
    scenario = scenarios.get(scenario_id, DEFAULT_SCENARIOS["technical_decision_support"])
    reference = reference_context.strip() if reference_context else DEFAULT_REFERENCE_BY_SCENARIO.get(scenario_id)

    semantic = score_semantic_consistency(response_text, reference)
    numeric_analysis = analyze_numeric_claims(response_text, scenario)
    citation_analysis = analyze_citations(response_text, scenario)
    logical_score = score_logical(response_text)
    clarity_score = score_clarity(response_text)
    task_alignment = score_task_alignment(response_text, original_prompt, scenario)
    safety_result = assess_operational_safety(response_text, scenario)

    factual_reliability = score_factual_reliability(
        semantic=semantic,
        numeric_score=float(numeric_analysis["score"]),
        citation_score=float(citation_analysis["score"]),
    )
    unsupported_signal = estimate_unsupported_claim_signal(response_text, citation_analysis)
    unsupported_risk = score_unsupported_content_risk(
        citation_score=float(citation_analysis["score"]),
        numeric_score=float(numeric_analysis["score"]),
        unsupported_signal=unsupported_signal,
    )

    dimension_scores = {
        "factual_reliability": factual_reliability,
        "task_alignment": task_alignment,
        "internal_consistency": logical_score,
        "interpretability_reviewability": clarity_score,
        "unsupported_content_risk": unsupported_risk,
        "operational_safety_risk": float(safety_result["risk_score"]),
    }

    weights = scenario["weights"]
    normalized_for_verification = {
        "factual_reliability": dimension_scores["factual_reliability"],
        "task_alignment": dimension_scores["task_alignment"],
        "internal_consistency": dimension_scores["internal_consistency"],
        "interpretability_reviewability": dimension_scores["interpretability_reviewability"],
        "unsupported_content_risk": 1 - dimension_scores["unsupported_content_risk"],
        "operational_safety_risk": 1 - dimension_scores["operational_safety_risk"],
    }

    verification_score = 0.0
    for metric, value in normalized_for_verification.items():
        verification_score += value * weights.get(metric, 0.0)

    overall_risk_score = max(0.0, min(1.0, 1 - verification_score))
    risk_level = derive_risk_level(overall_risk_score)
    flagged_items = build_flagged_evidence_items(
        citation_analysis=citation_analysis,
        numeric_analysis=numeric_analysis,
        logical_score=logical_score,
        unsupported_risk=unsupported_risk,
        safety_result=safety_result,
        scenario_id=scenario_id,
    )

    return {
        "scenario": scenario,
        "dimension_scores": dimension_scores,
        "weights": weights,
        "verification_score": verification_score,
        "overall_risk_score": overall_risk_score,
        "risk_level": risk_level,
        "flagged_evidence_items": flagged_items,
        "diagnostics": {
            "semantic_consistency": semantic,
            "numeric": numeric_analysis,
            "citations": citation_analysis,
            "safety": safety_result,
            "domain_terms": extract_domain_terms(response_text),
            "domain_coverage": score_domain_coverage(response_text),
            "unsupported_signal": unsupported_signal,
        },
    }


def build_dimension_table(dimension_scores: dict, weights: dict) -> pd.DataFrame:
    rows = []
    for metric, score in dimension_scores.items():
        w = weights.get(metric, 0.0)
        if metric in {"unsupported_content_risk", "operational_safety_risk"}:
            contribution = (1 - score) * w
            direction = "higher = higher risk"
        else:
            contribution = score * w
            direction = "higher = better grounding"

        rows.append({
            "Dimension": metric,
            "Score (0-1)": round(score, 3),
            "Weight": round(w, 3),
            "Weighted Contribution": round(contribution, 3),
            "Interpretation": DIMENSION_DESCRIPTIONS.get(metric, ""),
            "Direction": direction,
        })

    return pd.DataFrame(rows)


def build_verification_summary(result: Dict[str, object]) -> str:
    risk = result["overall_risk_score"]
    level = result["risk_level"]
    items = result["flagged_evidence_items"]
    high_count = sum(1 for i in items if i.get("severity") == "high")
    medium_count = sum(1 for i in items if i.get("severity") == "medium")
    return (
        f"Overall hallucination risk is **{level}** (score={risk:.2f}). "
        f"Flagged evidence items: {len(items)} (high={high_count}, medium={medium_count}). "
        "Prioritize human verification on high-severity items before operational use."
    )


def render_primary_report(title: str, result: Dict[str, object]):
    st.markdown(f"### {title}")
    risk_level = result["risk_level"]
    risk_score = result["overall_risk_score"]

    if risk_level == "Low":
        st.success(f"Overall Hallucination Risk Level: {risk_level} ({risk_score:.2f})")
    elif risk_level == "Medium":
        st.warning(f"Overall Hallucination Risk Level: {risk_level} ({risk_score:.2f})")
    else:
        st.error(f"Overall Hallucination Risk Level: {risk_level} ({risk_score:.2f})")

    st.markdown("#### Verification Summary")
    st.write(build_verification_summary(result))

    st.markdown("#### Flagged Evidence Items")
    for item in result["flagged_evidence_items"]:
        sev = item["severity"].upper()
        st.write(f"- [{sev}] {item['type']}: {item['detail']}")


def render_diagnostics(result: Dict[str, object]):
    diag = result["diagnostics"]
    citation = diag["citations"]
    numeric = diag["numeric"]
    safety = diag["safety"]

    st.markdown("#### Evidence-Based Verification Report")
    st.write(f"- Semantic consistency: `{diag['semantic_consistency']:.3f}`")
    st.write(
        f"- Citation credibility: score `{citation['score']:.3f}` | strictness `{citation['strictness']}` | "
        f"verified `{citation['verified']}` / `{citation['total']}`"
    )
    st.write(
        f"- Numeric plausibility: score `{numeric['score']:.3f}` | out-of-range `{numeric['out_of_range']}` / `{numeric['total']}`"
    )
    st.write(f"- Safety risk score: `{safety['risk_score']:.3f}` (profile: {safety['profile']})")
    st.write(f"- Domain coverage: `{diag['domain_coverage']:.3f}` with `{len(diag['domain_terms'])}` matched terms")

    if citation.get("unverified_items"):
        st.write("- Unverified citations:")
        for item in citation["unverified_items"]:
            st.write(f"  - {item}")

    if numeric.get("issues"):
        st.write("- Suspicious numeric values:")
        for issue in numeric["issues"]:
            st.write(f"  - {issue}")

    if safety.get("evidence"):
        st.write("- Safety alerts:")
        for e in safety["evidence"]:
            st.write(f"  - {e}")


def call_local_model(prompt: str) -> str:
    try:
        payload = {
            "model": LOCAL_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 512,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LOCAL_API_KEY}"}
        resp = requests.post(LOCAL_CHAT_ENDPOINT, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return f"[LM Studio Error {resp.status_code}] {resp.text}"
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Exception] {str(e)}"


def main():
    scenarios = load_scenarios()
    scenario_ids = list(scenarios.keys())

    st.set_page_config(page_title="Post-Generation Hallucination Risk Assessor", page_icon="🛡️", layout="wide")
    st.title("🛡️ Post-Generation Hallucination Risk Assessor (Power-System LLM Applications)")
    st.caption(
        "This system assesses hallucination risk in generated power-system LLM responses through "
        "post-generation verification."
    )

    with st.sidebar:
        st.header("⚙️ Assessment Settings")
        mode = st.radio(
            "Assessment mode",
            ["Single Response Assessment (Primary)", "Prompt Comparison (Optional)"],
            index=0,
        )

        scenario_id = st.selectbox(
            "Select task scenario",
            scenario_ids,
            format_func=lambda x: scenarios[x].get("display_name", x),
        )

        selected_scenario = scenarios[scenario_id]
        st.markdown("---")
        st.subheader("Scenario Focus")
        st.write(", ".join(selected_scenario.get("evaluation_focus", [])))

        st.markdown("---")
        st.subheader("Scenario-aware Weights")
        for metric, w in selected_scenario.get("weights", {}).items():
            st.write(f"- **{metric}** → `{w:.2f}`")

        st.markdown("---")
        st.subheader("📡 Local Model Status")
        try:
            r = requests.get(f"{LOCAL_API_BASE}/v1/models", timeout=3)
            if r.status_code == 200:
                st.success("✅ LM Studio API reachable")
            else:
                st.warning(f"⚠️ Reachable but status {r.status_code}")
        except Exception as e:
            st.error("❌ Cannot reach LM Studio at 127.0.0.1:1234")
            st.caption(str(e))

    if mode == "Single Response Assessment (Primary)":
        st.markdown("### Step 1. Provide Assessed Response")
        assessed_response = st.text_area(
            "Assessed response (required)",
            height=260,
            placeholder="Paste generated LLM response to be verified...",
        )

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            original_prompt = st.text_area(
                "Optional original prompt",
                height=130,
                placeholder="Prompt that produced the response (optional)",
            )
        with col_in2:
            reference_context = st.text_area(
                "Optional reference answer/context",
                height=130,
                placeholder="Authoritative context for semantic grounding (optional)",
            )

        run = st.button("🚀 Run Post-Generation Verification", type="primary")
        if not run:
            return

        if not assessed_response.strip():
            st.error("Please provide the generated response to assess.")
            return

        with st.spinner("Running multi-dimensional post-generation verification..."):
            result = evaluate_response(
                response_text=assessed_response,
                scenario_id=scenario_id,
                original_prompt=original_prompt,
                reference_context=reference_context,
            )

        render_primary_report("Primary Risk Output", result)

        st.markdown("### Dimension-wise Risk Profile")
        df = build_dimension_table(result["dimension_scores"], result["weights"])
        st.dataframe(df, width="stretch")

        with st.expander("Show full verification evidence report", expanded=True):
            render_diagnostics(result)

    else:
        st.markdown("### Optional Prompt Comparison Mode (Secondary)")
        st.info("This mode is secondary. Primary thesis workflow is single-response post-generation verification.")

        c1, c2 = st.columns(2)
        with c1:
            custom_prompt = st.text_area("Prompt A", height=220, key="cmp_a")
        with c2:
            baseline_prompt = st.text_area("Prompt B", height=220, key="cmp_b")

        run_cmp = st.button("🧪 Generate + Assess Both Responses")
        if not run_cmp:
            return

        if not custom_prompt.strip() or not baseline_prompt.strip():
            st.error("Please provide both prompts for optional comparison mode.")
            return

        with st.spinner("Generating responses from local LLM and assessing risk..."):
            resp_a = call_local_model(custom_prompt)
            resp_b = call_local_model(baseline_prompt)
            result_a = evaluate_response(resp_a, scenario_id=scenario_id, original_prompt=custom_prompt)
            result_b = evaluate_response(resp_b, scenario_id=scenario_id, original_prompt=baseline_prompt)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Prompt A Response")
            st.code(resp_a)
            render_primary_report("Prompt A Risk Result", result_a)
        with col_b:
            st.subheader("Prompt B Response")
            st.code(resp_b)
            render_primary_report("Prompt B Risk Result", result_b)


if __name__ == "__main__":
    main()
