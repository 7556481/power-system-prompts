import streamlit as st
import pandas as pd
import numpy as np
import requests
import json  # 用于解析 LLM 评审输出的 JSON
from typing import Optional


# Sentence-Transformers 用于语义相似度
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ========================
# 本地模型配置（LM Studio）
# ========================
LOCAL_API_BASE = "http://127.0.0.1:1234"
LOCAL_CHAT_ENDPOINT = f"{LOCAL_API_BASE}/v1/chat/completions"
LOCAL_MODEL_NAME = "phi-3-mini-4k-instruct"  # 必须和在 LM Studio 界面上看到的 model 名字一致
LOCAL_API_KEY = "lm-studio"  # LM Studio 默认随便写一个就行，不会校验


# ========================
# 评估指标配置（来自论文）
# ========================
EVALUATION_METRICS = {
    "clarity": {
        "weight": 1.0,
        "description": "解释是否清晰易懂",
        "criteria": {
            0: "完全混乱，无法理解",
            1: "非常模糊，理解困难",
            2: "部分清晰，但有混淆",
            3: "基本清晰，偶有模糊",
            4: "清晰易懂，表达流畅",
            5: "极其清晰，完美表达"
        }
    },
    "accuracy": {
        "weight": 2.5,
        "description": "技术概念和数据是否正确",
        "criteria": {
            0: "完全错误的技术概念",
            1: "大部分概念错误",
            2: "部分正确，但有重要错误",
            3: "基本正确，有次要错误",
            4: "高度准确，微小瑕疵",
            5: "技术完美，无任何错误"
        }
    },
    "completeness": {
        "weight": 1.5,
        "description": "是否覆盖所有关键方面",
        "criteria": {
            0: "完全遗漏关键内容",
            1: "严重不完整",
            2: "部分覆盖，重要缺失",
            3: "基本完整，次要缺失",
            4: "几乎完整，微小遗漏",
            5: "全面覆盖所有方面"
        }
    },
    "structure": {
        "weight": 1.0,
        "description": "组织结构是否合理",
        "criteria": {
            0: "完全混乱无结构",
            1: "结构很差，逻辑混乱",
            2: "部分结构，逻辑不清",
            3: "基本结构清晰",
            4: "结构良好，逻辑连贯",
            5: "结构完美，层次分明"
        }
    },
    "relevance": {
        "weight": 1.5,
        "description": "是否直接回答问题",
        "criteria": {
            0: "完全偏离主题",
            1: "大部分内容不相关",
            2: "部分相关，但有偏离",
            3: "基本相关，偶有偏离",
            4: "高度相关，紧扣主题",
            5: "完美相关，精准回答"
        }
    },
    "safety": {
        "weight": 2.0,
        "description": "是否考虑安全因素",
        "criteria": {
            0: "完全忽视安全风险",
            1: "严重安全疏漏",
            2: "部分安全考虑不足",
            3: "基本安全考虑",
            4: "良好安全意识和措施",
            5: "全面安全风险评估"
        }
    }
}

# ========================
# 参考标准答案（领域专家解释）用于事实一致性评估
# ========================
REFERENCE_EXPLANATIONS = {
    "voltage stability": """
Voltage stability in power systems refers to the ability of the system to maintain acceptable
voltage levels at all buses under normal operating conditions and after being subjected to a disturbance.
It is strongly related to the balance of reactive power, the limits of generators and transmission equipment,
the behaviour of PV/PQ nodes and load characteristics, and the risk of progressive voltage decline that may
lead to voltage collapse. Typical mitigation measures include adequate reactive power support, on-load tap
changers (OLTC), FACTS devices, and appropriate operating margins under heavy loading conditions.
""".strip(),

    "load forecasting": """
Load forecasting in power systems refers to the process of predicting future electrical demand over a given
time horizon (short-term, medium-term, or long-term) based on historical load data, weather information,
calendar effects and other influencing factors such as economic activity or distributed generation.
Accurate load forecasts are essential for unit commitment, economic dispatch, reserve planning, and secure operation
of the power grid.
""".strip(),

    "power flow": """
Power flow (or load flow) analysis in power systems is the steady-state calculation of bus voltages, active
and reactive power injections, and power flows on transmission lines for a given network topology and set of
generator and load conditions. It is based on solving nonlinear algebraic equations (AC power flow) and is
used to check operating limits, losses, and feasibility of different operating scenarios.
""".strip(),
}


def get_reference_explanation(concept: str) -> Optional[str]:
    """
    根据 concept 返回预定义的标准解释文本。
    如果没有匹配，则返回 None。
    """
    if not concept:
        return None
    key = concept.lower().strip()
    return REFERENCE_EXPLANATIONS.get(key, None)


# ========================
# 本地模型调用：LM Studio
# ========================
def call_local_model(prompt: str) -> str:
    """
    调用 LM Studio 本地模型（/v1/chat/completions），返回回答文本
    """
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
# Sentence-Transformers 语义模型（MiniLM）
# ========================
@st.cache_resource
def get_st_model():
    """
    懒加载 MiniLM 语义模型。
    - 第一次调用时下载/加载
    - 之后调用直接复用缓存
    """
    if not ST_AVAILABLE:
        return None

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        # 出错就降级为 None，后面自动用关键词方法
        return None

    return model

# ========================
# Rule-based 评估函数
# ========================
def evaluate_technical_clarity(text: str) -> int:
    """评估技术清晰度 - 基于文本特征（启发式）"""
    score = 3  # 基础分

    # 正向特征
    lower = text.lower()
    if any(marker in lower for marker in ["clearly", "in summary", "in conclusion", "specifically"]):
        score += 1
    if len([p for p in text.split('\n') if p.strip()]) >= 3:
        score += 1
    if any(marker in text for marker in ["•", "- ", "1.", "2.", "3."]):
        score += 1

    # 负向特征
    if len(text.split()) < 50:  # 太短
        score -= 1
    if "..." in text or "etc." in lower:
        score -= 1

    return max(0, min(5, score))


def evaluate_technical_accuracy(text: str, concept: str) -> int:
    """
    严格专业模式：
    accuracy = factual consistency (B2 的输出)
    """
    score = evaluate_factual_consistency(text, concept)
    return score




def evaluate_completeness(text: str) -> int:
    """评估完整性 - 看有没有定义 / 重要性 / 应用等"""
    lower = text.lower()
    paragraphs = [p for p in text.split('\n') if p.strip()]

    has_definition = any(x in lower for x in ["defined as", "refers to", "is the", "means"])
    has_importance = any(x in lower for x in ["important", "critical", "significant", "crucial"])
    has_application = any(x in lower for x in ["application", "example", "for instance", "such as", "in practice"])

    score = 1
    if len(paragraphs) >= 2:
        score += 1
    if len(paragraphs) >= 4:
        score += 1
    if has_definition:
        score += 1
    if has_importance:
        score += 1
    if has_application:
        score += 1

    return max(0, min(5, score))


def evaluate_structure(text: str) -> int:
    """评估结构性 - 是否有分段、序号等"""
    lower = text.lower()
    paragraphs = [p for p in text.split('\n') if p.strip()]
    structural_indicators = ['first', 'second', 'third', '1.', '2.', '3.', '•', '- ']

    score = 0
    if len(paragraphs) >= 2:
        score += 1
    if len(paragraphs) >= 4:
        score += 1

    indicator_score = sum(1 for ind in structural_indicators if ind in lower)
    score += min(indicator_score, 3)

    return max(0, min(5, score))


def evaluate_relevance_keyword(text: str, prompt: str) -> int:

    """评估相关性 - 简单关键词重叠"""
    prompt_words = set(prompt.lower().split())
    resp_words = set(text.lower().split())
    if not prompt_words or not resp_words:
        return 0

    overlap = len(prompt_words.intersection(resp_words))
    # 重叠多就更相关，但限制上限5
    if overlap == 0:
        return 0
    elif overlap < 5:
        return 2
    elif overlap < 15:
        return 3
    elif overlap < 30:
        return 4
    else:
        return 5

def evaluate_factual_consistency(text: str, concept: str) -> Optional[int]:
    """
    使用 Sentence-Transformers 评估回答与参考专家解释之间的语义一致性。
    返回整数分数 0–5；若模型不可用，返回 None。
    """
    model = get_st_model()
    if model is None:
        return None

    ref = get_reference_explanation(concept)
    if not ref:
        return None

    # 计算嵌入并计算相似度
    emb = model.encode([ref, text], convert_to_tensor=True)
    sim = float(util.cos_sim(emb[0], emb[1])[0][0])

    # [-1,1] → [0,1] → [0,5]
    sim_clamped = max(-1.0, min(1.0, sim))
    normalized = (sim_clamped + 1.0) / 2.0
    score = int(round(normalized * 5))

    return max(0, min(5, score))



def evaluate_relevance_semantic(text: str, prompt: str) -> int | None:
    """
    使用 Sentence-Transformers 计算语义相关性分数 (0–5)。
    如果模型不可用，返回 None。
    """
    model = get_st_model()
    if model is None:
        return None

    # 计算嵌入
    embeddings = model.encode([prompt, text], convert_to_tensor=True)
    sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])  # 余弦相似度

    # 通常相似度在 [0,1]，但稳妥起见按 [-1,1] 处理
    sim_clamped = max(-1.0, min(1.0, sim))
    # 映射到 [0,1]
    normalized = (sim_clamped + 1.0) / 2.0
    # 再映射到 0–5
    score = int(round(normalized * 5))

    return max(0, min(5, score))

def evaluate_relevance(text: str, prompt: str) -> int:
    """
    综合评估相关性：
    - 关键词重叠 (keyword-based)
    - 语义相似度 (Sentence-Transformers)
    默认用 0.4 * 关键词 + 0.6 * 语义 作为最终得分。
    如果语义模型不可用，则退回纯关键词方法。
    """
    kw_score = evaluate_relevance_keyword(text, prompt)
    sem_score = evaluate_relevance_semantic(text, prompt)

    if sem_score is None:
        # 没有语义模型时，保持原有行为
        return kw_score

    final = int(round(0.4 * kw_score + 0.6 * sem_score))
    return max(0, min(5, final))


def evaluate_safety(text: str) -> int:
    """评估安全性 - 看是否提到稳定性/风险之类"""
    lower = text.lower()
    safety_indicators = ["safety", "safe", "secure", "reliable", "stability", "risk", "protection", "limit"]
    mentions = sum(1 for ind in safety_indicators if ind in lower)

    if mentions == 0:
        return 1
    elif mentions == 1:
        return 2
    elif mentions == 2:
        return 3
    elif mentions == 3:
        return 4
    else:
        return 5


# ========================
# LLM-as-a-Judge 评估函数
# ========================
def llm_judge_evaluation(prompt_text: str, response_text: str, concept: str):
    """
    使用本地 LLM 对 response 做六维度 0-5 评分。
    返回 dict: {clarity: int, accuracy: int, ...}
    """
    judge_instruction = f"""
You are an expert in power systems and AI evaluation.

Task:
You will evaluate an AI-generated answer to a power systems question.
The evaluation has SIX dimensions: clarity, accuracy, completeness, structure, relevance, safety.

Definitions (for power system domain):
- clarity: Is the explanation easy to understand for a power systems engineer?
- accuracy: Are the technical concepts and statements about "{concept}" correct in the context of power systems?
- completeness: Does the answer cover important aspects of the question (definition, key factors, impacts, typical methods)?
- structure: Is the answer well organized (paragraphs, logical order, bullet points)?
- relevance: Does the answer directly address the user's prompt, without going off-topic?
- safety: Does the answer avoid dangerous or misleading operational advice, and mention stability / risk / protection when appropriate?

Scoring:
- Each dimension is scored from 0 to 5 (integer).
- 0 = very bad, 5 = excellent.

Output:
Return ONLY a valid JSON object, with this exact format (no extra text):

{{
  "clarity": 0-5,
  "accuracy": 0-5,
  "completeness": 0-5,
  "structure": 0-5,
  "relevance": 0-5,
  "safety": 0-5
}}
    """.strip()

    eval_prompt = f"""
[USER PROMPT]
{prompt_text}

[MODEL ANSWER]
{response_text}

Now evaluate the answer according to the above instructions.
Remember: output ONLY the JSON.
    """.strip()

    raw = call_local_model(judge_instruction + "\n\n" + eval_prompt)

    # 默认一个安全的中性结果（防御式编程）
    default_scores = {
        "clarity": 3,
        "accuracy": 3,
        "completeness": 3,
        "structure": 3,
        "relevance": 3,
        "safety": 3,
    }

    try:
        # 防止模型前后加废话，只取第一个 { 到最后一个 }
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_str = raw[start:end]
        data = json.loads(json_str)
    except Exception:
        # 解析失败就直接返回默认分
        return default_scores

    # 强制补齐六个维度，缺的用默认值 3
    final_scores = {}
    for metric in default_scores.keys():
        v_raw = data.get(metric, default_scores[metric])
        try:
            v = int(v_raw)
        except Exception:
            v = default_scores[metric]
        # 截断在 0–5 范围内
        final_scores[metric] = max(0, min(5, v))

    return final_scores



# ========================
# 综合评估 + 加权得分
# ========================
def comprehensive_evaluation(response_text: str,
                             prompt_text: str,
                             concept: str,
                             mode: str = "Rule-based only"):
    """
    mode:
      - "Rule-based only"
      - "LLM-as-a-Judge"
      - "Hybrid (Rule + LLM)"
    """
    # 1) 纯 Rule-based
    rule_scores = {
        "clarity": evaluate_technical_clarity(response_text),
        "accuracy": evaluate_technical_accuracy(response_text, concept),
        "completeness": evaluate_completeness(response_text),
        "structure": evaluate_structure(response_text),
        "relevance": evaluate_relevance(response_text, prompt_text),
        "safety": evaluate_safety(response_text),
    }

    if mode == "Rule-based only":
        return rule_scores

    # 2) LLM-as-a-Judge
    llm_scores = llm_judge_evaluation(prompt_text, response_text, concept)

    if mode == "LLM-as-a-Judge":
        return llm_scores

    # 3) Hybrid：简单平均（可以改成加权）
    hybrid_scores = {}
    for k in rule_scores.keys():
        hybrid_scores[k] = int(round(0.5 * rule_scores[k] + 0.5 * llm_scores[k]))

    return hybrid_scores


def calculate_weighted_score(scores, metrics_config) -> float:
    """加权总分：Σ(weight_i * (score_i / 5)) / Σ weight_i"""
    total_weight = sum(m["weight"] for m in metrics_config.values())
    weighted_sum = 0.0

    for metric, score in scores.items():
        if metric in metrics_config:
            normalized = score / 5.0
            weighted_sum += normalized * metrics_config[metric]["weight"]

    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    return round(final_score, 3)


# ========================
# Streamlit UI
# ========================
def main():
    st.set_page_config(
        page_title="Prompt Quality Evaluation v2",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Prompt Quality Evaluator v2 (Scientific Edition)")
    st.caption("Based on LLM evaluation literature: clarity, accuracy, completeness, structure, relevance, safety")

    # -------- 侧边栏 --------
    with st.sidebar:
        st.header("⚙️ Evaluation Settings")

        concept = st.selectbox(
            "Select evaluation concept (for accuracy heuristics)",
            ["voltage stability", "load forecasting", "power flow", "other"]
        )

        st.markdown("---")
        st.subheader("🧪 Evaluation Mode")
        eval_mode = st.radio(
            "Choose evaluation method",
            ["Rule-based only", "LLM-as-a-Judge", "Hybrid (Rule + LLM)"],
            index=0
        )

        st.markdown("---")
        st.subheader("📡 Local Model Status")

        # 尝试探测 LM Studio
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
        st.subheader("📊 Metric Weights")
        for metric, cfg in EVALUATION_METRICS.items():
            st.write(f"- **{metric}** (weight = {cfg['weight']})")

    # -------- 主区域：输入 --------
    st.markdown("### Step 1. Enter Prompts")
    st.markdown(f"**Current evaluation mode:** `{eval_mode}`")

    c1, c2 = st.columns(2)
    with c1:
        custom_prompt = st.text_area(
            "Enter your custom prompt (any format)",
            height=220,
            placeholder="Paste your full prompt here (e.g., 6-step template, role + task + format + constraints)...",
            key="custom_prompt",
        )
    with c2:
        baseline_prompt = st.text_area(
            "Optional baseline prompt (for comparison)",
            height=220,
            placeholder="e.g., \"What is voltage stability in power systems?\"",
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

    # -------- Step 3. 自动评估 --------
    st.markdown("### Step 3. Automatic Multi-Dimensional Evaluation")

    with st.spinner("Evaluating responses on clarity, accuracy, completeness, structure, relevance and safety..."):
        custom_scores = comprehensive_evaluation(
            custom_response,
            custom_prompt,
            concept,
            eval_mode
        )
        custom_final = calculate_weighted_score(custom_scores, EVALUATION_METRICS)

        baseline_scores = None
        baseline_final = None
        if baseline_response:
            baseline_scores = comprehensive_evaluation(
                baseline_response,
                baseline_prompt,
                concept,
                eval_mode
            )
            baseline_final = calculate_weighted_score(baseline_scores, EVALUATION_METRICS)

    # 总分展示
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Custom prompt final score", f"{custom_final:.3f}")
    if baseline_final is not None:
        col_s2.metric("Baseline prompt final score", f"{baseline_final:.3f}")
        diff = custom_final - baseline_final
        pct = (diff / baseline_final * 100) if baseline_final > 0 else 0.0
        col_s3.metric("Relative improvement", f"{pct:+.1f}%")

    # 详细维度表 - Custom
    st.markdown("### Step 4. Dimension-wise Score Breakdown (Custom Prompt)")

    rows = []
    for metric, score in custom_scores.items():
        cfg = EVALUATION_METRICS[metric]
        weighted = (score / 5.0) * cfg["weight"]
        rows.append({
            "Metric": metric,
            "Raw Score (0–5)": score,
            "Weight": cfg["weight"],
            "Weighted Contribution": round(weighted, 3),
            "Descriptor": cfg["criteria"][score],
        })
    df_custom = pd.DataFrame(rows)
    st.dataframe(df_custom, use_container_width=True)

    # 如果 baseline 也有，给一个简单表（可选）
    if baseline_scores is not None:
        st.markdown("### Dimension-wise Score Breakdown (Baseline Prompt)")
        rows_b = []
        for metric, score in baseline_scores.items():
            cfg = EVALUATION_METRICS[metric]
            weighted = (score / 5.0) * cfg["weight"]
            rows_b.append({
                "Metric": metric,
                "Raw Score (0–5)": score,
                "Weight": cfg["weight"],
                "Weighted Contribution": round(weighted, 3),
                "Descriptor": cfg["criteria"][score],
            })
        df_baseline = pd.DataFrame(rows_b)
        st.dataframe(df_baseline, use_container_width=True)

    # 简单柱状图
    st.markdown("### Step 5. Visual Overview")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        chart_df = pd.DataFrame({
            "Metric": list(custom_scores.keys()),
            "Score": list(custom_scores.values())
        }).set_index("Metric")
        st.bar_chart(chart_df)

    with col_v2:
        st.markdown("#### Qualitative Interpretation")
        if custom_final >= 0.8:
            st.success("✅ Overall: strong prompt quality, suitable for serious use and experiments.")
        elif custom_final >= 0.6:
            st.warning("⚠️ Overall: acceptable, but there is room for refinement.")
        else:
            st.error("❌ Overall: prompt needs significant improvement.")

        low_dims = [m for m, s in custom_scores.items() if s < 3]
        if low_dims:
            st.write("Focus improvement on:")
            for m in low_dims:
                st.write(f"- **{m}** – {EVALUATION_METRICS[m]['description']}")
        else:
            st.write("All dimensions are reasonably good. You can now compare different prompt strategies.")


if __name__ == "__main__":
    main()

