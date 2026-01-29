import streamlit as st
import requests
import json

st.set_page_config(page_title="电力AI助手", page_icon="⚡")

st.title("⚡ 电力AI助手原型 (本地LLM版)")

st.write("""
欢迎使用电力AI助手原型！  
现在我已连接本地 AI 模型（LM Studio），可以为您提供更智能的电力系统分析。
""")

# LM Studio 本地 API 地址
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "phi-3-mini-4k-instruct"  # 你的模型名称（可在 LM Studio 界面中看到）

# 用户输入
user_question = st.text_input("请输入您关于电力系统的问题：", placeholder="例如：什么是潮流计算？")

if user_question:
    with st.spinner('AI正在深度分析中...'):
        try:
            # 构造提示词
            messages = [
                {"role": "system", "content": (
                    "You are an experienced power system engineer specializing in grid stability, "
                    "load forecasting, and energy optimization. You always respond with clear, "
                    "concise, and technically accurate explanations suitable for graduate-level understanding."
                )},
                {"role": "user", "content": (
                    f"Explain the following concept in power systems: {user_question}\n\n"
                    "Your answer must include:\n"
                    "1. A precise and professional definition.\n"
                    "2. Its importance and role in modern power systems.\n"
                    "3. At least one real-world application example.\n"
                    "Keep the explanation around 200–250 words."
                )}
            ]

            # 构造请求体
            payload = {
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
            }

            # 发送到本地LM Studio服务器
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()
                ai_response = data['choices'][0]['message']['content']
                st.success("✅ 专业分析完成！")
                st.markdown(f"**AI分析结果：**\n\n{ai_response}")
            else:
                st.error(f"❌ 请求失败：{response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"程序出错：{str(e)}")

# 侧边栏状态
with st.sidebar:
    st.header("系统状态")
    st.success("✅ 已连接到本地LM Studio模型")
    st.info(f"模型名称：{MODEL}")
    st.caption("如需修改，请在 LM Studio 中切换模型。")
