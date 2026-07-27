import os
import io
import time
import base64
import requests
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain_groq import ChatGroq

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)
from langchain_community.tools import (
    WikipediaQueryRun,
    ArxivQueryRun,
    DuckDuckGoSearchRun
)

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

import networkx as nx
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

import yt_dlp


# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional — only used if you add these to .env; enables real Google Images results.
# Free to create: https://console.cloud.google.com (Custom Search API) +
# https://programmablesearchengine.google.com (Search Engine ID, turn on "Image search")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    .result-card {
        background-color: #17171A;
        border: 1px solid #2A2A2E;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .result-card h4 {
        margin: 0 0 10px 0;
        font-size: 15px;
        font-weight: 600;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 24px !important;
    }

    /* Mic button styling to match rounded pill look next to input */
    div.stButton > button, div[data-testid="stButton"] button {
        border-radius: 24px !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #2A2A2E;
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 10px 0;
    }

    img {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 AI search agent")
st.caption("Search, voice, mind map, short notes, YouTube, images")
st.caption("Developed by Shanteshwar Ojha |\n Student of Government Polytechnic College")

# =====================================================
# API Key Check (env only, no user input)
# =====================================================

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file. Please add it and restart the app.")
    st.stop()


# =====================================================
# Sidebar
# =====================================================

st.sidebar.markdown("#### ⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Model",
    ["llama-3.3-70b", "llama-3.1-8b"]
)

enable_voice_output = st.sidebar.checkbox("Voice output", value=True)
enable_youtube = st.sidebar.checkbox("YouTube", value=True)
enable_images = st.sidebar.checkbox("Images", value=True)
enable_mindmap = st.sidebar.checkbox("Mind map", value=True)
enable_shortnotes = st.sidebar.checkbox("Short notes", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("LangChain + Groq")
st.sidebar.caption("Contact Us : \n shanteshwarojha2006@gmail.com")

MODEL_MAP = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "llama-3.1-8b": "llama-3.1-8b-instant"
}
model_name = MODEL_MAP.get(model_name, model_name)


# =====================================================
# Session State
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello 👋 Ask me anything — by typing or using the mic below."}
    ]

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# =====================================================
# Helper Functions
# =====================================================

def get_related_images(query, max_results=4):
    """Fetch related images with multiple fallbacks, so something shows up
    for almost any topic: Google (if configured) -> DuckDuckGo (2 tries)
    -> Openverse -> Wikipedia thumbnail."""

    WIKI_HEADERS = {"User-Agent": "AI-Search-Agent/1.0 (educational project; contact: none)"}

    # ---- Attempt 0: Google Custom Search (only if GOOGLE_API_KEY + GOOGLE_CSE_ID set) ----
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CSE_ID,
                    "q": query,
                    "searchType": "image",
                    "num": min(max_results, 10)
                },
                timeout=8
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if items:
                return [
                    {"image": item.get("link"), "title": item.get("title", "")}
                    for item in items[:max_results]
                ]
        except Exception as e:
            st.session_state["_img_error_google"] = str(e)

    # ---- Attempt 1: DuckDuckGo (retry once — token extraction is flaky) ----
    for attempt in range(2):
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.images(
                        query,
                        region="wt-wt",
                        safesearch="moderate",
                        max_results=max_results
                    )
                )
            if results:
                return [
                    {"image": r.get("image") or r.get("thumbnail"), "title": r.get("title", "")}
                    for r in results
                ]
            break
        except Exception as e:
            st.session_state["_img_error_ddg"] = str(e)
            if attempt == 0:
                time.sleep(1.5)

    # ---- Attempt 2: Openverse (public API, no key needed) ----
    try:
        resp = requests.get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": max_results},
            headers={"User-Agent": "AI-Search-Agent/1.0"},
            timeout=8
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            return [
                {"image": r.get("thumbnail") or r.get("url"), "title": r.get("title", "")}
                for r in results[:max_results]
            ]
    except Exception as e:
        st.session_state["_img_error_openverse"] = str(e)

    # ---- Attempt 3: Wikipedia page thumbnail (very reliable, needs a User-Agent header) ----
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": max_results,
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 400
            },
            headers=WIKI_HEADERS,
            timeout=8
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        images = []
        for page in pages.values():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                images.append({"image": thumb, "title": page.get("title", "")})
        if images:
            return images[:max_results]
    except Exception as e:
        ddg_err = st.session_state.get("_img_error_ddg", "unknown")
        ov_err = st.session_state.get("_img_error_openverse", "unknown")
        st.session_state["_img_error"] = (
            f"DuckDuckGo failed ({ddg_err}); Openverse failed ({ov_err}); Wikipedia failed ({e})"
        )
        return []

    ddg_err = st.session_state.get("_img_error_ddg", "")
    ov_err = st.session_state.get("_img_error_openverse", "")
    st.session_state["_img_error"] = (
        f"No images found from any source. DuckDuckGo: {ddg_err} | Openverse: {ov_err}"
    )
    return []


def get_youtube_thumbnails(query, max_results=3):
    """Fetch YouTube video thumbnails using yt-dlp (no API key, no async issues)."""
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }
        search_query = f"ytsearch{max_results}:{query}"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)

        entries = info.get("entries", []) if info else []

        videos = []
        for entry in entries:
            if not entry:
                continue
            video_id = entry.get("id")
            thumbnail = entry.get("thumbnail") or (
                f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None
            )
            videos.append({
                "title": entry.get("title", "Untitled"),
                "thumbnail": thumbnail,
                "link": entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            })
        return videos
    except Exception as e:
        st.session_state["_yt_error"] = str(e)
        return []


def text_to_speech(text, lang="en"):
    """Convert text to speech audio bytes."""
    try:
        tts = gTTS(text=text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"Voice output failed: {e}")
        return None


def build_mind_map_structure(text, topic, n_branches=4, per_branch=2):
    """
    Build a 2-level hierarchy: topic -> branch keyword -> sub-keywords.
    Branches are formed from top sentences/chunks so related words end up
    grouped together instead of one flat list of keywords.
    """
    import re
    from collections import Counter

    stopwords = set([
        "the", "is", "in", "at", "of", "on", "and", "a", "an", "to", "for",
        "with", "this", "that", "it", "as", "are", "was", "were", "be",
        "by", "or", "from", "which", "these", "those", "can", "will",
        "has", "have", "had", "not", "but", "also", "its", "their", "into",
        "such", "than", "then", "there", "over", "more", "when", "some"
    ])

    sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
    sentences = sentences[: n_branches * 2] or [text]

    branches = {}
    used_words = set()

    for sent in sentences:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", sent.lower())
        words = [w for w in words if w not in stopwords and w not in used_words]
        if not words:
            continue

        counted = [w for w, _ in Counter(words).most_common(per_branch + 1)]
        if not counted:
            continue

        branch_label = counted[0].capitalize()
        children = [w for w in counted[1: per_branch + 1]]

        if branch_label.lower() in used_words:
            continue

        branches[branch_label] = children
        used_words.update([branch_label.lower()] + children)

        if len(branches) >= n_branches:
            break

    return branches


CARD_FIGSIZE = (7, 8.6)
CARD_DPI = 140


def generate_mind_map(topic, branches):
    """
    Render a detailed mind map on a FIXED canvas (same size as the short notes
    card): branch cluster-arrows pointing to a central topic box, each branch
    also fans out into small connected sub-keyword pill nodes (tree effect),
    plus a bottom mini flow-diagram summarizing the branches as ordered steps.
    """
    try:
        import matplotlib.patches as mpatches

        branch_list = list(branches.items())
        n = max(len(branch_list), 1)

        fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

        palette = ["#4ECDC4", "#7F77DD", "#F0997B", "#5DCAA5", "#ED93B1"]

        # ---- Title ----
        ax.text(5, 11.5, topic.upper(), fontsize=13, fontweight="bold",
                 color="#26215C", ha="center", va="center")
        ax.plot([0.4, 9.6], [11.05, 11.05], color="#26215C", linewidth=1)

        top_area_y0, top_area_y1 = 3.2, 10.6
        usable_h = top_area_y1 - top_area_y0
        row_h = usable_h / n

        branch_tip_points = []

        for i, (branch, children) in enumerate(branch_list):
            y_center = top_area_y1 - i * row_h - row_h / 2
            color = palette[i % len(palette)]

            # Chevron/arrow box pointing right (branch shape #1)
            arrow_w = min(0.5, row_h * 0.45)
            arrow = mpatches.FancyArrow(
                0.3, y_center, 3.6, 0,
                width=arrow_w, head_width=arrow_w * 1.6, head_length=0.55,
                length_includes_head=True,
                facecolor=color, edgecolor="none", alpha=0.9
            )
            ax.add_patch(arrow)
            ax.text(0.55, y_center + arrow_w * 0.55, branch, fontsize=10.5,
                     fontweight="bold", color="#1A1A1A", va="center", ha="left")

            tip_x = 0.3 + 3.6
            branch_tip_points.append((tip_x, y_center, color))

            # Sub-keyword pill nodes (shape #2) connected via thin lines
            n_children = max(len(children), 1)
            for j, child in enumerate(children):
                sub_y = y_center + (j - (n_children - 1) / 2) * 0.55
                sub_x = tip_x + 1.3

                ax.plot([tip_x + 0.1, sub_x - 0.55], [y_center, sub_y],
                         color=color, linewidth=1.2, alpha=0.8)

                pill = mpatches.FancyBboxPatch(
                    (sub_x - 0.55, sub_y - 0.18), 1.5, 0.36,
                    boxstyle="round,pad=0.02,rounding_size=0.18",
                    facecolor="white", edgecolor=color, linewidth=1.4
                )
                ax.add_patch(pill)
                ax.text(sub_x + 0.2, sub_y, child, fontsize=8, color="#1A1A1A",
                         ha="center", va="center")

        # ---- Central topic box (shape #3) ----
        box_x, box_y = 6.7, top_area_y0 + usable_h * 0.15
        box_w, box_h = 2.9, usable_h * 0.7
        topic_box = mpatches.FancyBboxPatch(
            (box_x, box_y), box_w, box_h,
            boxstyle="round,pad=0.15,rounding_size=0.3",
            facecolor="#3C3489", edgecolor="none"
        )
        ax.add_patch(topic_box)
        ax.text(box_x + box_w / 2, box_y + box_h / 2, topic, fontsize=11,
                 fontweight="bold", color="white", va="center", ha="center", wrap=True)

        # connect each branch tip loosely toward the topic box
        for tip_x, y_center, color in branch_tip_points:
            ax.plot([tip_x - 0.1, box_x], [y_center, box_y + box_h / 2],
                     color=color, linewidth=0.8, alpha=0.35, linestyle="--")

        # ---- Bottom mini flow-diagram (compulsory second diagram) ----
        ax.text(5, 2.5, "PROCESS FLOW", fontsize=9.5, fontweight="bold",
                 color="#26215C", ha="center", va="center")

        flow_labels = [b for b, _ in branch_list][:4] or ["Step 1", "Step 2", "Step 3"]
        n_flow = len(flow_labels)
        flow_w = 8.6 / n_flow
        start_x = 0.7

        for i, label in enumerate(flow_labels):
            fx = start_x + i * flow_w
            color = palette[i % len(palette)]

            box = mpatches.FancyBboxPatch(
                (fx, 1.1), flow_w - 0.5, 0.9,
                boxstyle="round,pad=0.05,rounding_size=0.15",
                facecolor=color, edgecolor="none", alpha=0.9
            )
            ax.add_patch(box)
            ax.text(fx + (flow_w - 0.5) / 2, 1.55, label, fontsize=8,
                     fontweight="bold", color="#1A1A1A", ha="center", va="center", wrap=True)

            if i < n_flow - 1:
                ax.annotate(
                    "", xy=(fx + flow_w - 0.15, 1.55), xytext=(fx + flow_w - 0.5, 1.55),
                    arrowprops=dict(arrowstyle="-|>", color="#26215C", linewidth=1.4)
                )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=CARD_DPI, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        st.session_state["_mindmap_error"] = str(e)
        return None


def invoke_llm_safe(llm, prompt, retries=2, delay=3):
    """Call llm.invoke with retries — Groq's API occasionally rate-limits
    when several LLM calls happen in one turn (agent + formatting + notes)."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = llm.invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay)
    raise last_error


def generate_short_notes_text(llm, query, response_text):
    """Ask the LLM to turn the answer into 5 headings, each with 5 short points."""
    try:
        prompt = f"""
Create structured short notes from the following answer for a visual notes card.

Format EXACTLY like this, with no extra text before or after:

HEADING: <short heading, max 4 words>
- <point 1, max 8 words>
- <point 2, max 8 words>
- <point 3, max 8 words>
- <point 4, max 8 words>
- <point 5, max 8 words>

Repeat this HEADING block 5 times total (5 headings, 5 points each = 25 points).
Keep every point crisp, no full sentences, no extra commentary.

Topic: {query}

Answer:
{response_text}
"""
        return invoke_llm_safe(llm, prompt)
    except Exception as e:
        st.session_state["_notes_error"] = str(e)
        return ""


def parse_short_notes(raw_text):
    """Parse 'HEADING: x' + '- point' blocks into an ordered dict {heading: [points]}."""
    headings = {}
    current = None

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("HEADING"):
            _, _, heading_text = line.partition(":")
            current = heading_text.strip() or f"Section {len(headings) + 1}"
            headings[current] = []
        elif line.lstrip().startswith(("-", "•")) and current:
            point = line.lstrip("-•").strip()
            if point:
                headings[current].append(point)

    # keep at most 5 headings, 5 points each
    trimmed = {}
    for h, pts in list(headings.items())[:5]:
        trimmed[h] = pts[:5]
    return trimmed


def render_topic_diagram(ax, cx, cy, radius, color):
    """Draw a small decorative topic diagram (radial burst icon) at (cx, cy)."""
    import matplotlib.patches as mpatches
    import math

    core = mpatches.Circle((cx, cy), radius * 0.45, facecolor=color, edgecolor="none")
    ax.add_patch(core)

    for i in range(8):
        angle = math.radians(i * 45)
        x1 = cx + radius * 0.55 * math.cos(angle)
        y1 = cy + radius * 0.55 * math.sin(angle)
        x2 = cx + radius * math.cos(angle)
        y2 = cy + radius * math.sin(angle)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, solid_capstyle="round")
        dot = mpatches.Circle((x2, y2), radius * 0.12, facecolor=color, edgecolor="none")
        ax.add_patch(dot)


def render_short_notes_card(title, headings):
    """Render a colorful multi-section notes card on the SAME fixed canvas size
    as the mind map, so both cards display at an equal size side by side."""
    try:
        import matplotlib.patches as mpatches

        palette = ["#4ECDC4", "#F0997B", "#7F77DD", "#5DCAA5", "#ED93B1", "#FAC775"]

        heading_list = list(headings.items()) or [("Notes", ["No content available"])]
        total_points = sum(len(pts) for _, pts in heading_list) or 1
        n_sections = len(heading_list)

        fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

        card = mpatches.FancyBboxPatch(
            (0.2, 0.2), 9.6, 11.6,
            boxstyle="round,pad=0.05,rounding_size=0.35",
            facecolor="#FFF9E8", edgecolor="#2C2C2A", linewidth=1.2
        )
        ax.add_patch(card)

        # ---- Title (same layout convention as the mind map) ----
        ax.text(5, 11.5, title.upper(), fontsize=13, fontweight="bold",
                 color="#26215C", ha="center", va="center")
        render_topic_diagram(ax, 9.0, 11.5, 0.4, "#7F77DD")
        ax.plot([0.4, 9.6], [11.05, 11.05], color="#26215C", linewidth=1)

        # ---- Fit all headings + points inside a FIXED usable area ----
        top_area_y0, top_area_y1 = 0.4, 10.7
        usable_h = top_area_y1 - top_area_y0

        # weighted units: a heading bar counts as 1.3 rows, each point as 1 row
        total_units = n_sections * 1.3 + total_points
        unit_h = usable_h / total_units

        head_font = min(11, max(6.5, unit_h * 11))
        point_font = min(9.5, max(6, unit_h * 9))

        y = top_area_y1
        for h_idx, (heading, points) in enumerate(heading_list):
            color = palette[h_idx % len(palette)]
            bar_h = unit_h * 1.1

            head_bar = mpatches.FancyBboxPatch(
                (0.6, y - bar_h), 8.8, bar_h * 0.9,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=color, edgecolor="none"
            )
            ax.add_patch(head_bar)
            ax.text(1.0, y - bar_h * 0.55, heading, fontsize=head_font,
                     fontweight="bold", color="#1A1A1A", ha="left", va="center")
            y -= bar_h + unit_h * 0.15

            for point in points:
                bullet = mpatches.Circle((1.0, y - unit_h * 0.4), unit_h * 0.09,
                                           facecolor=color, edgecolor="none")
                ax.add_patch(bullet)
                ax.text(1.25, y - unit_h * 0.4, point, fontsize=point_font,
                         color="#2C2C2A", ha="left", va="center")
                y -= unit_h

            y -= unit_h * 0.2

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=CARD_DPI, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        st.session_state["_notes_error"] = str(e)
        return None


def create_txt_file(query, response):
    content = f"Question:\n{query}\n\nAnswer:\n{response}\n"
    return content.encode("utf-8")


def create_pdf_file(query, response):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, height - 2 * cm, "AI Search Agent - Response")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, height - 3 * cm, "Question:")
    c.setFont("Helvetica", 10)
    text_obj = c.beginText(2 * cm, height - 3.7 * cm)
    for line in wrap_text(query, 95):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, height - 5 * cm, "Answer:")
    c.setFont("Helvetica", 10)
    text_obj = c.beginText(2 * cm, height - 5.7 * cm)
    for line in wrap_text(response, 95):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.save()
    buf.seek(0)
    return buf


def wrap_text(text, width):
    import textwrap
    wrapped = []
    for paragraph in text.split("\n"):
        wrapped.extend(textwrap.wrap(paragraph, width) or [""])
    return wrapped


@st.cache_resource
def get_agent(_llm, model_key):
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)

    def safe_wikipedia(query):
        try:
            return wiki_wrapper.run(query)
        except Exception as e:
            return f"Wikipedia lookup failed right now ({e}). Answer using other tools or general knowledge instead."

    def safe_arxiv(query):
        try:
            return arxiv_wrapper.run(query)
        except Exception as e:
            return f"Arxiv lookup failed right now ({e}). Answer using other tools or general knowledge instead."

    def safe_duckduckgo(query):
        try:
            return DuckDuckGoSearchRun().run(query)
        except Exception as e:
            return f"Web search failed right now ({e}). Answer using other tools or general knowledge instead."

    from langchain.tools import Tool

    search = Tool(
        name="duckduckgo_search",
        func=safe_duckduckgo,
        description="Search the web for current information using DuckDuckGo."
    )
    wiki = Tool(
        name="wikipedia",
        func=safe_wikipedia,
        description="Search Wikipedia for factual/background information on a topic."
    )
    arxiv = Tool(
        name="arxiv",
        func=safe_arxiv,
        description="Search Arxiv for research papers on a topic."
    )

    tools = [search, wiki, arxiv]

    agent = initialize_agent(
        tools=tools,
        llm=_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=(
            "Your last response used the wrong format. You MUST reply using exactly this format, "
            "with no extra text:\n"
            "Thought: <your reasoning>\n"
            "Action: <one of the tool names, nothing else>\n"
            "Action Input: <the exact text to search for, no quotes, no parentheses>\n\n"
            "If you already have enough information to answer, instead reply with:\n"
            "Thought: <your reasoning>\n"
            "Final Answer: <your answer>\n\n"
            "Try again now, following this format exactly."
        ),
        handle_tool_error=True,
        max_iterations=15,
        max_execution_time=100,
        early_stopping_method="generate"
    )
    return agent



# =====================================================
# Display Chat History
# =====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# Input Section: Text + Voice
# =====================================================

col1, col2 = st.columns([5, 1])

with col1:
    typed_prompt = st.chat_input("Ask anything...")

with col2:
    voice_text = speech_to_text(
        language="en",
        start_prompt="🎙️",
        stop_prompt="⏹️",
        just_once=True,
        key="mic"
    )

user_prompt = typed_prompt or voice_text


# =====================================================
# Process Query
# =====================================================

if user_prompt:

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        streaming=True,
        temperature=0.3
    )

    # Secondary lightweight model used only for reformatting/short-notes —
    # keeps daily token usage on the main (often bigger) model much lower.
    fast_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        streaming=False,
        temperature=0.3
    )

    agent = get_agent(llm, model_name)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            # Step 1: let the ReAct agent focus purely on answering using tools
            # (no formatting instructions here — that's what breaks its tool-call format)
            raw_answer = agent.run(user_prompt, callbacks=[callback])

            # Step 2: a plain (non-agent) LLM call reformats the raw answer
            format_prompt = f"""
Rewrite the following research answer into this exact structure, in simple clear English:

1. First, 3 to 4 well-developed paragraphs explaining the topic fully.
2. Then a section titled "Key Points" with 5 to 8 short one-line bullet points (use "- " for each).

Do not skip either section. Keep all factual details from the research answer below.

Research answer:
{raw_answer}
"""
            response = invoke_llm_safe(fast_llm, format_prompt)
        except Exception as e:
            response = f"Error:\n\n{e}"

        st.markdown(response)

        # ---------------- Voice Output ----------------
        if enable_voice_output and not response.startswith("Error"):
            audio_fp = text_to_speech(response)
            if audio_fp:
                st.audio(audio_fp, format="audio/mp3")

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_response = response
    st.session_state.last_query = user_prompt

    # ---------------- Extras: Images | YouTube (side by side), Mind Map below ----------------
    if enable_images or enable_youtube:
        st.markdown("<br>", unsafe_allow_html=True)
        card_col1, card_col2 = st.columns(2)

        if enable_images:
            with card_col1:
                st.markdown('<div class="result-card"><h4>🖼️ Related images</h4>', unsafe_allow_html=True)
                st.session_state.pop("_img_error", None)
                images = get_related_images(user_prompt, max_results=4)
                if images:
                    img_cols = st.columns(2)
                    for idx, img in enumerate(images):
                        with img_cols[idx % 2]:
                            try:
                                st.image(
                                    img.get("image") or img.get("thumbnail"),
                                    use_container_width=True
                                )
                            except Exception:
                                st.caption("Image failed to load")
                elif st.session_state.get("_img_error"):
                    st.error(f"Could not fetch images: {st.session_state['_img_error']}")
                else:
                    st.caption("No images found for this query.")
                st.markdown('</div>', unsafe_allow_html=True)

        if enable_youtube:
            with card_col2:
                st.markdown('<div class="result-card"><h4>▶️ YouTube videos</h4>', unsafe_allow_html=True)
                st.session_state.pop("_yt_error", None)
                videos = get_youtube_thumbnails(user_prompt, max_results=3)
                if videos:
                    for vid in videos:
                        v_col1, v_col2 = st.columns([1, 3])
                        with v_col1:
                            if vid.get("thumbnail"):
                                st.image(vid["thumbnail"], use_container_width=True)
                        with v_col2:
                            st.markdown(f"[{vid.get('title', '')[:55]}]({vid.get('link', '#')})")
                elif st.session_state.get("_yt_error"):
                    st.error(f"Could not fetch YouTube results: {st.session_state['_yt_error']}")
                else:
                    st.caption("No videos found for this query.")
                st.markdown('</div>', unsafe_allow_html=True)

    if enable_mindmap or enable_shortnotes:
        st.markdown("<br>", unsafe_allow_html=True)
        mm_col, notes_col = st.columns(2)

        if enable_mindmap:
            with mm_col:
                st.markdown('<div class="result-card"><h4>🧠 Mind map</h4>', unsafe_allow_html=True)
                st.session_state.pop("_mindmap_error", None)
                branches = build_mind_map_structure(response, user_prompt[:25])
                if branches:
                    mindmap_img = generate_mind_map(user_prompt[:25], branches)
                    if mindmap_img:
                        st.image(mindmap_img, use_container_width=True)
                    elif st.session_state.get("_mindmap_error"):
                        st.error(f"Mind map generation failed: {st.session_state['_mindmap_error']}")
                else:
                    st.caption("Not enough content to build a mind map.")
                st.markdown('</div>', unsafe_allow_html=True)

        if enable_shortnotes:
            with notes_col:
                st.markdown('<div class="result-card"><h4>📝 AI short notes</h4>', unsafe_allow_html=True)
                st.session_state.pop("_notes_error", None)
                raw_notes = generate_short_notes_text(fast_llm, user_prompt, response)
                headings = parse_short_notes(raw_notes)
                if headings:
                    notes_img = render_short_notes_card(user_prompt[:30], headings)
                    if notes_img:
                        st.image(notes_img, use_container_width=True)
                        st.download_button(
                            "⬇️ Download short notes",
                            data=notes_img,
                            file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                    elif st.session_state.get("_notes_error"):
                        st.error(f"Short notes generation failed: {st.session_state['_notes_error']}")
                elif st.session_state.get("_notes_error"):
                    st.error(f"Short notes generation failed: {st.session_state['_notes_error']}")
                else:
                    st.caption("Could not generate short notes for this answer.")
                st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Download Options ----------------
    st.markdown('<div class="result-card"><h4>📥 Download this response</h4>', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        txt_data = create_txt_file(user_prompt, response)
        st.download_button(
            "⬇️ Download txt",
            data=txt_data,
            file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with dl_col2:
        pdf_data = create_pdf_file(user_prompt, response)
        st.download_button(
            "⬇️ Download pdf",
            data=pdf_data,
            file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    st.markdown('</div>', unsafe_allow_html=True)