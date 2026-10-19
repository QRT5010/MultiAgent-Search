"""工程规范知识库检索工具（M8 扩展新增）

在本地规范知识库（651 条条文块）中做关键词匹配检索。
检索策略：2-gram + 3-gram 词频 × IDF（罕见词权重更高），降低长块与高频词稀释。
数据来源：traffic-agent 的 docs/chunks.json（结构感知切分产出）
"""
import json
import math
import re
from pathlib import Path

from langchain_core.tools import tool

from api.monitor import monitor

# 知识库数据文件（assets/ 目录，避开 .gitignore 的 data/ 规则）
_DATA_PATH = Path(__file__).resolve().parents[1] / "assets" / "standards_chunks.json"

_chunks = None   # 懒加载缓存
_df = None       # 文档频率缓存（每个 gram 出现在多少个块中）


def _load_chunks():
    """加载知识库（首次调用时读取，之后复用）"""
    global _chunks
    if _chunks is None:
        _chunks = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _chunks


def _grams(text, n):
    """提取 n-gram 片段（去标点/空白），用于中文关键词匹配"""
    clean = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return [clean[i:i + n] for i in range(len(clean) - n + 1)]


def _build_df(chunks):
    """预计算文档频率：每个 2/3-gram 出现在多少个块里"""
    global _df
    if _df is None:
        df = {}
        for c in chunks:
            seen = set(_grams(c["text"], 2)) | set(_grams(c["text"], 3))
            for g in seen:
                df[g] = df.get(g, 0) + 1
        _df = df
    return _df


@tool
def search_standards(query: str, top_k: int = 3) -> str:
    """在工程规范知识库中检索条文。

    当用户询问公路、城市道路等工程规范的具体要求（技术等级、设计速度、净高、
    车道宽度、服务区、隧道分类等）时使用此工具。
    注意：仅用于工程规范条文检索，不涉及互联网搜索或业务数据库。

    :param query: 检索问题或关键词，例如"公路技术等级划分"
    :param top_k: 返回的条文数量，默认 3，最多 8
    :return: 命中的规范条文（含来源标注）
    """
    # 向前端推送工具调用进度（与 tavily_tool 一致的埋点方式）
    monitor.report_tool("规范知识库检索", {"query": query, "top_k": top_k})

    chunks = _load_chunks()
    df = _build_df(chunks)
    n_docs = len(chunks)

    # 过滤疑问词（"哪几个/是多少"等噪声片段，避免干扰匹配）
    q_clean = query
    for w in ("哪几个", "是多少", "是什么", "怎么样", "什么", "怎么", "如何", "哪些",
              "多少", "应该", "请问", "一下"):
        q_clean = q_clean.replace(w, "")

    # 查询的 2-gram（权重 1）+ 3-gram（权重 2）——更长片段更能锁定精确条文
    q_grams = [(g, 1.0) for g in _grams(q_clean, 2)] + [(g, 2.0) for g in _grams(q_clean, 3)]
    if not q_grams:
        return "查询内容为空，请提供具体的规范问题。"

    scored = []
    for c in chunks:
        text = c["text"]
        raw = 0.0
        for g, w in q_grams:
            cnt = text.count(g)
            if cnt:
                # IDF：越罕见的片段权重越高（高频词如"公路"接近不加分）
                idf = math.log((n_docs + 1) / (df.get(g, 1) + 1)) + 1.0
                raw += cnt * idf * w
        if raw > 0:
            # 长度归一化：分数 / √块长 —— 防止长块靠词频总量碾压精确短块；
            # 下限按 120 字计，防止极短碎片块占便宜
            score = raw / math.sqrt(max(len(text), 120))
            scored.append((score, c))
    scored.sort(key=lambda item: item[0], reverse=True)

    try:
        k = max(1, min(int(top_k), 8))
    except (TypeError, ValueError):
        k = 3
    top = scored[:k]

    if not top:
        return "知识库中未检索到相关规范条文。"

    parts = []
    for _score, c in top:
        parts.append(f"【来源：{c['source']}】\n{c['text']}")
    return "\n\n---\n\n".join(parts)
