import streamlit as st
import pandas as pd
from openai import OpenAI
import concurrent.futures
import requests
import re
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False


# ============================================================
# APP CONFIG (NO USER-EDITABLE SETTINGS)
# ============================================================

MAX_WORKERS = 3
BATCH_SIZE = 5
SCRAPE_TIMEOUT_SECONDS = 20
TEXT_EXCERPT_CHARS = 3000
MAX_H2 = 20

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# HELPERS
# ============================================================

def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_signals_from_html(html: str) -> dict:
    soup = BeautifulSoup(html or "", "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Title
    title = _clean_text(soup.title.get_text() if soup.title else "")

    # Meta description
    meta_description = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_description = _clean_text(md.get("content", ""))

    # H1
    h1 = ""
    h1_tag = soup.find("h1")
    if h1_tag:
        h1 = _clean_text(h1_tag.get_text())

    # H2s
    h2s = [_clean_text(h.get_text()) for h in soup.find_all("h2")]
    h2s = [h for h in h2s if h][:MAX_H2]

    # Text excerpt
    full_text = _clean_text(soup.get_text(" "))
    text_excerpt = full_text[:TEXT_EXCERPT_CHARS]

    return {
        "title": title,
        "meta_description": meta_description,
        "h1": h1,
        "h2s": h2s,
        "text_excerpt": text_excerpt,
    }


# ============================================================
# HYBRID SCRAPING (Playwright -> Requests+BS4 fallback)
# ============================================================

def scrape_url_hybrid(url: str, timeout: int = SCRAPE_TIMEOUT_SECONDS) -> dict:
    """
    Fetch and parse a URL to extract:
      - final_url (after redirects or navigation)
      - status_code (best-effort; reliable for requests)
      - title, meta_description, h1, h2s, text_excerpt
      - method used: "playwright" or "requests" or "error"
    """

    # 1) Try Playwright first (JS sites)
    if PLAYWRIGHT_AVAILABLE:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_default_navigation_timeout(30_000)
                page.set_default_timeout(10_000)

                page.goto(url, wait_until="networkidle")
                final_url = page.url

                html = page.content()
                signals = _extract_signals_from_html(html)

                browser.close()

                # Note: status_code is not reliably captured in this simple pattern.
                return {
                    "final_url": final_url,
                    "status_code": 200,
                    "method": "playwright",
                    **signals,
                }
        except Exception:
            pass

    # 2) Fallback: Requests + BS4
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()

        signals = _extract_signals_from_html(r.text)
        return {
            "final_url": r.url,
            "status_code": r.status_code,
            "method": "requests",
            **signals,
        }
    except Exception:
        return {
            "final_url": url,
            "status_code": None,
            "method": "error",
            "title": "",
            "meta_description": "",
            "h1": "",
            "h2s": [],
            "text_excerpt": "",
        }


@st.cache_data(ttl=3600, show_spinner=False)
def cached_scrape(url: str) -> dict:
    return scrape_url_hybrid(url)


# ============================================================
# PROMPTING
# ============================================================

def build_messages(keyword: str, url: str, guidelines: str | None, page: dict) -> list:
    system = (
        "You are an expert SEO Content Strategist.\n"
        "You will receive a URL, its target keyword, optional user-provided guidelines, and extracted on-page signals "
        "(title/meta/H1/H2/text excerpt).\n"
        "Use ONLY these inputs to infer what the page is about and what intent it serves.\n"
        "Then generate a comprehensive content brief in the required structure.\n"
        "Do NOT mention scraping. Output ONLY the content brief."
    )

    guidelines_clean = (guidelines or "").strip()
    guidelines_block = f"\nUser Guidelines (optional):\n{guidelines_clean}\n" if guidelines_clean else ""

    user = f"""
URL: {url}
Keyword: {keyword}
{guidelines_block}
Extracted on-page signals:
- Final URL (after redirects): {page.get("final_url","")}
- HTTP Status: {page.get("status_code")}
- Method: {page.get("method")}
- Current Title: {page.get("title","")}
- Current Meta Description: {page.get("meta_description","")}
- Current H1: {page.get("h1","")}
- Current H2s: {page.get("h2s", [])}
- Page Text Excerpt (truncated): {page.get("text_excerpt","")}

Generate a comprehensive content brief respecting the following structure:

URL, Primary Keyword, Secondary Keywords, Title, Meta Description, Headings structure (H1, H2, H3 etc...), Other comments.

Guidelines:
- Primary Keyword should be the provided keyword (unless the page clearly targets a closer variant).
- Secondary Keywords: propose 6–12 closely related terms (no stuffing).
- Title: max ~60 characters (hard limit 65).
- Meta Description: 140–160 characters ideal.
- Headings Structure: propose an improved outline (H1 + 4–10 H2s, with optional H3s).
- Other comments: highlight gaps/opportunities based on the on-page signals and the optional user guidelines.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_one(client: OpenAI, keyword: str, url: str, guidelines: str | None) -> str:
    page = cached_scrape(url)
    messages = build_messages(keyword, url, guidelines, page)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )

    content = (resp.choices[0].message.content or "").strip()
    return content if content else "Error: Empty model response."


def generate_batch(client: OpenAI, batch_rows: list[tuple[int, str, str, str | None]]) -> list[tuple[int, str]]:
    """
    batch_rows: list of (pos, keyword, url, guidelines)
    returns: list of (pos, recommendation)
    """
    out = []
    for pos, keyword, url, guidelines in batch_rows:
        try:
            out.append((pos, generate_one(client, keyword, url, guidelines)))
        except Exception as e:
            out.append((pos, f"Error: {e}"))
    return out


# ============================================================
# STREAMLIT APP UI
# ============================================================

st.title("Bulk SEO Content Brief Generator")

api_key = st.text_input("Enter your OpenAI API key", type="password")

uploaded_file = st.file_uploader(
    "Choose your CSV file. It must contain columns 'Keyword' and 'URL'. "
    "Optional: 'Guidelines'.",
    type=["csv"]
)

st.caption(
    "Scraping uses Playwright when available (better for JS sites), otherwise requests+BeautifulSoup for static HTML. "
    "Results are cached for 1 hour."
)

if uploaded_file and api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Could not initialize OpenAI client: {e}")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Validate required columns
    required_cols = {"Keyword", "URL"}
    if not required_cols.issubset(set(df.columns)):
        st.error("CSV must contain columns: 'Keyword' and 'URL'.")
        st.stop()

    # Normalize
    df = df.dropna(subset=["Keyword", "URL"]).copy()
    df["Keyword"] = df["Keyword"].astype(str).str.strip()
    df["URL"] = df["URL"].astype(str).str.strip()

    has_guidelines_col = "Guidelines" in df.columns
    if has_guidelines_col:
        df["Guidelines"] = df["Guidelines"].astype(str).fillna("").str.strip()
    else:
        df["Guidelines"] = ""

    total = len(df)
    if total == 0:
        st.error("No valid rows found (Keyword/URL).")
        st.stop()

    # Prepare positional rows to preserve order
    rows = []
    for pos, row in enumerate(df.itertuples(index=False)):
        kw = getattr(row, "Keyword")
        url = getattr(row, "URL")
        guidelines = getattr(row, "Guidelines", "") if has_guidelines_col else ""
        guidelines = guidelines if str(guidelines).strip() else None
        rows.append((pos, kw, url, guidelines))

    # Batch
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_batches = len(batches)

    progress_bar = st.progress(0)
    status = st.empty()

    results = [None] * total
    completed_batches = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_batch, client, batch) for batch in batches]

        for fut in concurrent.futures.as_completed(futures):
            batch_results = fut.result()  # list of (pos, rec)
            for pos, rec in batch_results:
                results[pos] = rec

            completed_batches += 1
            progress_bar.progress(min(completed_batches / total_batches, 1.0))
            status.write(
                f"Processed {completed_batches}/{total_batches} batches "
                f"({sum(r is not None for r in results)}/{total} rows)"
            )

    out_df = pd.DataFrame({
        "Keyword": df["Keyword"].tolist(),
        "URL": df["URL"].tolist(),
        # Only include Guidelines in output if it was present in input
        **({"Guidelines": df["Guidelines"].tolist()} if has_guidelines_col else {}),
        "Recommendations": results,
    })

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Content Briefs as CSV", csv_bytes, "content-briefs.csv", "text/csv")
    st.dataframe(out_df, use_container_width=True)
