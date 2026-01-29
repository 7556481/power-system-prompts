import streamlit as st
import pandas as pd
import requests
import re
from typing import Optional, Tuple, List, Dict

# 尝试引入 sentence-transformers（用于语义相似度）
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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
# 电力系统领域参考知识（用于 Accuracy & Hallucination）
# ========================
REFERENCE_EXPLANATIONS = {
    "voltage stability": """
Voltage stability in power systems refers to the ability of the system to maintain acceptable
voltage levels at all buses under normal operating conditions and after being subjected to a disturbance.
It is strongly related to reactive power balance, generator and transmission limits, PV/PQ bus behaviour,
load characteristics, and the risk of progressive voltage decline that may lead to voltage collapse.
Typical mitigation measures include adequate reactive power support, OLTC, FACTS, and appropriate margins
under heavy loading.
""".strip(),

    "load forecasting": """
Load forecasting in power systems is the process of predicting future electrical demand over different
time horizons (short-, medium-, long-term) based on historical load data, weather, calendar effects and
other influencing factors. Accurate forecasts are essential for unit commitment, economic dispatch,
reserve allocation and secure operation.
""".strip(),

    "power flow": """
Power flow (load flow) analysis computes bus voltages, active/reactive injections and branch flows in the
steady state for a given network topology, generator setpoints and load conditions. It is based on solving
nonlinear AC power flow equations and is used to check operating limits, losses and feasibility of scenarios.
""".strip(),
}


def get_reference_explanation(concept: str) -> Optional[str]:
    if not concept:
        return None
    return REFERENCE_EXPLANATIONS.get(concept.lower().strip())


# ========================
# Sentence-Transformers 模型
# ========================
@st.cache_resource
def get_st_model():
    if not ST_AVAILABLE:
        return None
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformer(model_name)
    except Exception:
        return None


# ========================
# V3 指标权重（0–1 之间，加起来 = 1）
# ========================
WEIGHTS = {
    "accuracy": 0.35,
    "relevance": 0.15,
    "logical": 0.20,
    "clarity": 0.10,
    "hallucination": 0.15,
    "safety": 0.05,
}

# 维度文本描述（用于表格）
METRIC_DESCRIPTIONS = {
    "accuracy": "专业正确性 / 与领域标准解释一致 + 数据与引用可信",
    "relevance": "是否紧扣任务和提示词要求",
    "logical": "推理与结构是否有清晰的因果与层次",
    "clarity": "语言是否清晰、易懂，适合工程师阅读",
    "hallucination": "是否存在严重概念错误或胡编乱造（分数越高越安全，含引用核验）",
    "safety": "是否体现运行安全意识，避免危险或违规建议",
}

# Likert 描述
METRIC_CRITERIA = {
    "accuracy": {
        0: "完全错误或与标准相反",
        1: "大量错误，严重偏离领域知识",
        2: "部分正确，但有明显误解",
        3: "大致正确，有一些模糊或小错误",
        4: "高度一致，数据与引用基本可信",
        5: "与标准解释高度一致、数据与引用严谨",
    },
    "relevance": {
        0: "几乎完全不相关",
        1: "大部分内容偏离任务",
        2: "部分相关但有大量跑题",
        3: "基本相关，偶尔有偏离",
        4: "高度相关，紧扣任务要求",
        5: "完美贴合任务与上下文",
    },
    "logical": {
        0: "完全缺乏逻辑结构",
        1: "逻辑很弱，难以跟随",
        2: "有一些结构，但较混乱",
        3: "基本有清晰结构与步骤",
        4: "逻辑良好，因果链条清楚",
        5: "逻辑结构非常清晰，推理严谨",
    },
    "clarity": {
        0: "几乎无法理解",
        1: "非常拗口或混乱",
        2: "能看懂，但阅读费力",
        3: "基本清晰，偶有模糊",
        4: "清晰易懂，表达自然",
        5: "极其清晰、紧凑、易读",
    },
    "hallucination": {
        0: "严重幻觉：大量胡编、概念严重错误",
        1: "明显幻觉：多个关键点错误或混淆",
        2: "存在可见概念错误或可疑说法",
        3: "大体可靠但有少量可疑内容",
        4: "基本无幻觉，引用可验证或未涉及",
        5: "无明显幻觉，引用与数据高度可信",
    },
    "safety": {
        0: "建议存在严重安全隐患或违规操作",
        1: "安全意识很弱，几乎不考虑风险",
        2: "偶尔提到风险，但不系统",
        3: "基本体现安全考虑",
        4: "对于风险与约束有较好说明",
        5: "系统性地考虑运行安全与约束",
    },
}


# ========================
# 事实核验工具（数据与引用）
# ========================
NUMERIC_RANGES = {
    "hz": (45.0, 65.0),          # 常见电网频率范围（大幅偏离视为可疑）
    "kv": (0.1, 1000.0),         # 高压电网常见电压等级范围
    "mw": (0.1, 100000.0),       # 系统规模差异很大，粗略过滤极端值
    "mvar": (0.1, 100000.0),
    "pu": (0.5, 1.5),            # 标幺电压通常围绕 1.0 p.u.
}

NUMERIC_UNIT_ALIASES = {
    "hz": {"hz"},
    "kv": {"kv", "kV"},
    "mw": {"mw", "MW"},
    "mvar": {"mvar", "MVar", "MVAr", "mVar"},
    "pu": {"pu", "p.u.", "p.u"},
}

CLAIM_HINTS = [
    "paper", "study", "research", "according to", "et al", "doi", "arxiv",
    "论文", "文献", "研究", "根据", "结论表明",
]


def extract_numeric_claims(text: str) -> List[Tuple[float, str]]:
    pattern = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kV|kv|Hz|hz|MW|mw|MVar|MVAr|mvar|mVar|p\.u\.|p\.u|pu)"
    )
    claims = []
    for match in pattern.finditer(text):
        value = float(match.group("value"))
        unit_raw = match.group("unit")
        unit = unit_raw.lower().replace(".", "")
        claims.append((value, unit))
    return claims


def score_numeric_plausibility(text: str) -> float:
    claims = extract_numeric_claims(text)
    if not claims:
        return 0.7  # 无数值可核验，保持中性

    total = 0
    ok = 0
    for value, unit in claims:
        total += 1
        canonical = None
        for key, aliases in NUMERIC_UNIT_ALIASES.items():
            if unit in {alias.lower().replace(".", "") for alias in aliases}:
                canonical = key
                break
        if canonical is None:
            continue
        low, high = NUMERIC_RANGES[canonical]
        if low <= value <= high:
            ok += 1

    if total == 0:
        return 0.7
    return max(0.0, min(1.0, ok / total))


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
        resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=6
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False)
def verify_arxiv(arxiv_id: str) -> bool:
    try:
        resp = requests.get(
            f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
            timeout=6
        )
        return resp.status_code == 200 and "<entry>" in resp.text
    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False)
def verify_title(title: str) -> bool:
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": title, "rows": 1},
            timeout=6
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return False
        score = items[0].get("score", 0)
        return score >= 20
    except (requests.RequestException, ValueError):
        return False


def score_citation_validity(text: str) -> float:
    lowered = text.lower()
    claim_present = any(hint in lowered for hint in CLAIM_HINTS)
    citations = extract_citations(text)
    total = len(citations["doi"]) + len(citations["arxiv"]) + len(citations["title"])

    if total == 0:
        return 0.2 if claim_present else 0.7

    verified = 0
    for doi in citations["doi"]:
        verified += 1 if verify_doi(doi) else 0
    for arxiv_id in citations["arxiv"]:
        verified += 1 if verify_arxiv(arxiv_id) else 0
    for title in citations["title"]:
        verified += 1 if verify_title(title) else 0

    return max(0.0, min(1.0, verified / total))


# ========================
# 评分函数：Accuracy（语义一致性 + 数据可信度）
# ========================
def score_accuracy(text: str, concept: str) -> float:
    """
    综合语义一致性与数据可信度评估，映射到 [0,1]
    """
    model = get_st_model()
    ref = get_reference_explanation(concept)
    semantic = 0.5
    if model is not None and ref:
        embeddings = model.encode([ref, text], convert_to_tensor=True)
        sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
        semantic = (sim + 1.0) / 2.0  # [-1,1] → [0,1]

    numeric = score_numeric_plausibility(text)
    score = 0.6 * semantic + 0.4 * numeric
    return max(0.0, min(1.0, score))


# ========================
# 评分函数：Relevance（与 Prompt 贴合度）
# ========================
def score_relevance_semantic(text: str, prompt: str) -> Optional[float]:
    model = get_st_model()
    if model is None:
        return None
    embeddings = model.encode([prompt, text], convert_to_tensor=True)
    sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
    score = (sim + 1.0) / 2.0
    return max(0.0, min(1.0, score))


def score_relevance_keyword(text: str, prompt: str) -> float:
    prompt_words = set(prompt.lower().split())
    resp_words = set(text.lower().split())
    if not prompt_words or not resp_words:
        return 0.0
    overlap = len(prompt_words.intersection(resp_words))
    if overlap == 0:
        return 0.0
    elif overlap < 5:
        return 0.3
    elif overlap < 15:
        return 0.5
    elif overlap < 30:
        return 0.7
    else:
        return 0.9


def score_relevance(text: str, prompt: str) -> float:
    kw = score_relevance_keyword(text, prompt)
    sem = score_relevance_semantic(text, prompt)
    if sem is None:
        return kw
    # 语义比重更高一点
    return max(0.0, min(1.0, 0.4 * kw + 0.6 * sem))


# ========================
# 评分函数：Logical Structure
# ========================
def score_logical(text: str) -> float:
    """
    Logical score using NLI consistency check.
    Output: 0 to 1
    1 = 完全一致，无逻辑冲突
    0 = 高度矛盾
    """

    tokenizer, model = get_nli_model()

    # 分句
    sentences = [s.strip() for s in re.split(r'[.!?。\n]', text) if len(s.strip()) > 5]
    if len(sentences) < 2:
        return 0.5

    contradiction = 0
    entail = 0
    pairs = 0

    # 对每两句做推理
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            pairs += 1

            inputs = tokenizer(
                sentences[i],
                sentences[j],
                return_tensors="pt",
                truncation=True
            )

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs).item()  # 0:contradiction, 1:neutral, 2:entail

            if label == 0:
                contradiction += 1
            elif label == 2:
                entail += 1

    if pairs == 0:
        return 0.5

    score = 1 - (contradiction / pairs)
    return max(0, min(1, score))



# ========================
# 评分函数：Clarity（清晰度）
# ========================
def score_clarity(text: str) -> float:
        """微调版本 - 更平滑的梯度"""
        import textstat

        try:
            fk = textstat.flesch_kincaid_grade(text)

            if fk < 7:
                return 0.4  # 过于简单但可能清晰
            elif fk <= 10:
                return 0.7  # 偏简单但可用
            elif fk <= 12.5:
                return 0.9  # 工程文档最佳区间
            elif fk <= 14:
                return 0.8
            elif fk <= 16:
                return 0.65
            elif fk <= 18:
                return 0.5
            else:
                return 0.3

        except:
            return 0.5


# ========================
# 评分函数：Hallucination Risk（语义 + 引用核验 + 数据合理性）
# ========================
def score_hallucination(text: str, concept: str) -> float:
    """
    分数越高越“安全”（幻觉越少）。
    组合逻辑：
    - 语义一致性：参考知识匹配度
    - 引用核验：DOI/arXiv/标题可验证性
    - 数值合理性：常见工程范围过滤
    """
    model = get_st_model()
    ref = get_reference_explanation(concept)
    semantic = 0.5
    if model is not None and ref:
        embeddings = model.encode([ref, text], convert_to_tensor=True)
        sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
        semantic = (sim + 1.0) / 2.0

    citation = score_citation_validity(text)
    numeric = score_numeric_plausibility(text)
    score = 0.4 * semantic + 0.4 * citation + 0.2 * numeric
    return max(0.0, min(1.0, score))



# ========================
# 评分函数：Safety
# ========================
def safety_score(text: str) -> float:
    """
    临时 Safety 占位实现：
    - 不做模型推理
    - 保证系统可运行
    """
    if not text.strip():
        return 0.0
    return 1.0  # 默认认为“安全”




# ========================
# 统一评估接口（V3 evaluator）
# ========================
def evaluate_output(response_text: str,
                    prompt_text: str,
                    concept: str):
    """
    返回：
    {
      "scores": {metric: 0-5 likert 分},
      "raw_scores": {metric: 0-1},
      "weighted_score": 0-1,
      "weighted_score_100": 0-100
    }
    """
    # 原始 0–1 分
    raw_scores = {
        "accuracy": score_accuracy(response_text, concept),
        "relevance": score_relevance(response_text, prompt_text),
        "logical": score_logical(response_text),
        "clarity": score_clarity(response_text),
        "hallucination": score_hallucination(response_text, concept),
        "safety": safety_score(response_text),

    }

    # 转成 0–5 Likert
    likert_scores = {
        k: int(round(v * 5)) for k, v in raw_scores.items()
    }

    # 加权总分
    weighted_sum = 0.0
    for metric, v in raw_scores.items():
        w = WEIGHTS.get(metric, 0.0)
        weighted_sum += v * w

    weighted_score = max(0.0, min(1.0, weighted_sum))
    return {
        "scores": likert_scores,
        "raw_scores": raw_scores,
        "weighted_score": weighted_score,
        "weighted_score_100": round(weighted_score * 100, 1),
    }


# ========================
# LM Studio 本地模型调用
# ========================
def call_local_model(prompt: str) -> str:
    try:
        payload = {
            "model": LOCAL_MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 512,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LOCAL_API_KEY}",
        }
        resp = requests.post(
            LOCAL_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        if resp.status_code != 200:
            return f"[LM Studio Error {resp.status_code}] {resp.text}"
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Exception] {str(e)}"


# ========================
# 用于 UI 展示的表格构造
# ========================
def build_metric_table(scores: dict,
                       raw_scores: dict) -> pd.DataFrame:
    rows = []
    for metric, likert in scores.items():
        raw = raw_scores.get(metric, 0.0)
        weight = WEIGHTS.get(metric, 0.0)
        contribution = raw * weight
        descriptor = METRIC_CRITERIA.get(metric, {}).get(likert, "")
        rows.append({
            "Metric": metric,
            "Score (0–5)": likert,
            "Raw (0–1)": round(raw, 3),
            "Weight": weight,
            "Weighted Contribution": round(contribution, 3),
            "Description": METRIC_DESCRIPTIONS.get(metric, ""),
            "Level": descriptor,
        })
    return pd.DataFrame(rows)


# ========================
# Streamlit UI
# ========================
def main():
    st.set_page_config(
        page_title="Prompt Quality Evaluation v3",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Prompt Quality Evaluator v3 (Power System – Scientific Edition)")
    st.caption("Single unified evaluator · Accuracy / Relevance / Hallucination / Logical / Clarity / Safety")

    # -------- 侧边栏 --------
    with st.sidebar:
        st.header("⚙️ Evaluation Settings")

        concept = st.selectbox(
            "Select evaluation concept (for domain reference)",
            ["voltage stability", "load forecasting", "power flow", "other"]
        )

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

        st.markdown("---")
        st.subheader("📊 Metric Weights (V3)")
        for metric, w in WEIGHTS.items():
            st.write(f"- **{metric}** → weight = `{w:.2f}`")

    # -------- 主区域：输入 --------
    st.markdown("### Step 1. Enter Prompts")

    c1, c2 = st.columns(2)
    with c1:
        custom_prompt = st.text_area(
            "Enter your custom prompt (any format)",
            height=240,
            placeholder="Paste your full power-system prompt here (role + task + context + format + constraints)...",
            key="custom_prompt",
        )
    with c2:
        baseline_prompt = st.text_area(
            "Optional baseline prompt (for comparison)",
            height=240,
            placeholder='e.g., "Explain what voltage stability is in a power system."',
            key="baseline_prompt",
        )

    run = st.button("🚀 Run Evaluation", type="primary")

    if not run:
        return

    if not custom_prompt.strip():
        st.error("Please enter at least a custom prompt.")
        return

    # -------- Step 2. 调用本地模型 --------
    st.markdown("### Step 2. Generate AI Responses")

    with st.spinner("Querying local LLM via LM Studio..."):
        custom_response = call_local_model(custom_prompt)
        baseline_response = call_local_model(baseline_prompt) if baseline_prompt.strip() else None

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("Custom Prompt → Model Response")
        st.code(custom_response, language="markdown")
    with col_r2:
        if baseline_response:
            st.subheader("Baseline Prompt → Model Response")
            st.code(baseline_response, language="markdown")
        else:
            st.info("No baseline prompt provided.")

    # -------- Step 3. V3 统一评估 --------
    st.markdown("### Step 3. V3 Unified Evaluation")

    with st.spinner("Evaluating outputs with V3 unified metrics..."):
        eval_custom = evaluate_output(
            response_text=custom_response,
            prompt_text=custom_prompt,
            concept=concept,
        )
        custom_scores = eval_custom["scores"]
        custom_raw = eval_custom["raw_scores"]
        custom_final = eval_custom["weighted_score"]
        custom_final_100 = eval_custom["weighted_score_100"]

        baseline_scores = None
        baseline_raw = None
        baseline_final = None
        baseline_final_100 = None
        if baseline_response:
            eval_base = evaluate_output(
                response_text=baseline_response,
                prompt_text=baseline_prompt,
                concept=concept,
            )
            baseline_scores = eval_base["scores"]
            baseline_raw = eval_base["raw_scores"]
            baseline_final = eval_base["weighted_score"]
            baseline_final_100 = eval_base["weighted_score_100"]

    # -------- Step 4. 总分展示 --------
    st.markdown("### Step 4. Overall Scores")

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Custom prompt V3 score", f"{custom_final_100:.1f} / 100")
    if baseline_final is not None:
        col_s2.metric("Baseline prompt V3 score", f"{baseline_final_100:.1f} / 100")
        diff = custom_final - baseline_final
        pct = (diff / baseline_final * 100) if baseline_final > 0 else 0.0
        col_s3.metric("Relative improvement", f"{pct:+.1f}%")
    else:
        col_s2.metric("Baseline prompt V3 score", "N/A")
        col_s3.metric("Relative improvement", "N/A")

    # -------- Step 5. 维度拆解 --------
    st.markdown("### Step 5. Dimension-wise Breakdown (Custom Prompt)")
    df_custom = build_metric_table(custom_scores, custom_raw)
    st.dataframe(df_custom, width="stretch")

    if baseline_scores is not None and baseline_raw is not None:
        st.markdown("### Dimension-wise Breakdown (Baseline Prompt)")
        df_baseline = build_metric_table(baseline_scores, baseline_raw)
        st.dataframe(df_baseline, width="stretch")

    # -------- Step 6. 可视化与解释 --------
    st.markdown("### Step 6. Visual Overview & Interpretation")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        chart_df = pd.DataFrame({
            "Metric": list(custom_scores.keys()),
            "Score": list(custom_scores.values())
        }).set_index("Metric")
        st.bar_chart(chart_df)

    with col_v2:
        st.markdown("#### Qualitative Interpretation (Custom Prompt)")
        if custom_final >= 0.85:
            st.success("✅ Overall: strong prompt quality, suitable for serious power system experiments.")
        elif custom_final >= 0.7:
            st.warning("⚠️ Overall: acceptable, but there is room for refinement.")
        else:
            st.error("❌ Overall: prompt needs significant improvement before being used in critical systems.")

        low_dims = [m for m, s in custom_scores.items() if s < 3]
        if low_dims:
            st.write("Focus improvement on these dimensions:")
            for m in low_dims:
                st.write(f"- **{m}** – {METRIC_DESCRIPTIONS.get(m, '')}")
        else:
            st.write("All dimensions are reasonably good. You can now safely compare different prompt strategies.")

        st.markdown("---")
        st.markdown("**Note**: V3 evaluator combines embedding similarity with "
                    "citation verification (DOI/arXiv/title lookup) and numeric plausibility checks, "
                    "without LLM-as-a-judge, to keep it stable and reproducible for scientific use in power systems.")


if __name__ == "__main__":
    main()
