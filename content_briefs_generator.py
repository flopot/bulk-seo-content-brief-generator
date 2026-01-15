import streamlit as st
import pandas as pd
from openai import OpenAI
import concurrent.futures
import requests
import re
from bs4 import BeautifulSoup

# ============================================================
# PAGE SCRAPING (Requests + BeautifulSoup)
# ============================================================

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def scrape_url_bs4(url: str, timeout: int = 20) -> dict:
    """
    Fetch and parse a URL (static HTML) to extract:
      - title
      - meta description
      - first H1
      - H2 list (top 20)
      - text excerpt (top ~3000 chars)
    Returns empty strings/lists on failure.
    """
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

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
        h2s = [h for h in h2s if h][:20]

        # Text excerpt
        full_text = _clean_text(soup.get_text(" "))
        text_excerpt = full_text[:3000]

        return {
            "final_url": r.url,
            "status_code": r.status_code,
            "title": title,
            "meta_description": meta_description,
            "h1": h1,
            "h2s": h2s,
            "text_excerpt": text_excerpt,
        }
    except Exception:
        return {
            "final_url": url,
            "status_code": None,
            "title": "",
            "meta_description": "",
            "h1": "",
            "h2s": [],
            "text_excerpt": "",
        }

@st.cache_data(ttl=3600, show_spinner=False)
def cached_scrape(url: str) -> dict:
    return scrape_url_bs4(url)


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("Bulk SEO Content Brief Generator")

st.markdown(
    """
    by [Florian Potier](https://twitter.com/FloPots) - [Intrepid Digital](https://www.intrepidonline.com/)
    """,
    unsafe_allow_html=True
)

api_key = st.text_input("Enter your OpenAI API key", type="password")

uploaded_file = st.file_uploader(
    "Choose your CSV file. It should contain the columns 'Keyword' and 'URL' "
    "(See [Template](https://docs.google.com/spreadsheets/d/1ApdoOKjC6ZAg1JkWiY51fOToXqH_cTP8PSj2wVlS8Iw/edit?usp=sharing))",
    type=["csv"]
)

# Settings
with st.sidebar:
    st.header("Settings")
    max_workers = st.slider("Concurrency (workers)", min_value=1, max_value=8, value=3)
    batch_size = st.slider("Batch size", min_value=1, max_value=20, value=5)
    scrape_timeout = st.slider("Scrape timeout (seconds)", min_value=5, max_value=60, value=20)
    st.caption("Note: Streamlit Cloud may rate-limit or block some sites. Requests-based scraping works best on static HTML.")


def build_messages(keyword: str, url: str, page: dict) -> list:
    system = (
        "You are an expert SEO Content Strategist.\n"
        "You will receive a URL, its target keyword, and extracted on-page signals (title/meta/H1/H2/text excerpt).\n"
        "Use ONLY these inputs to infer what the page is about and what intent it serves.\n"
        "Then generate a comprehensive content brief in the required structure.\n"
        "Do NOT mention that you scraped the page. Output ONLY the content brief."
    )

    user = f"""
URL: {url}
Keyword: {keyword}

Extracted on-page signals:
- Final URL (after redirects): {page.get("final_url","")}
- HTTP Status: {page.get("status_code")}
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
- Other comments: highlight gaps/opportunities based on the on-page signals (missing sections, unclear intent, weak headings, etc.).
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_one(client: OpenAI, keyword: str, url: str) -> str:
    # Scrape page (cached)
    page = cached_scrape(url)

    # If you want the timeout slider to apply, bypass cache by calling scrape_url_bs4(url, timeout=scrape_timeout)
    # but caching becomes less effective. Keeping cache for scale.
    # page = scrape_url_bs4(url, timeout=scrape_timeout)

    messages = build_messages(keyword, url, page)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )

    content = (resp.choices[0].message.content or "").strip()
    return content if content else "Error: Empty model response."


def generate_seo_recommendations(client: OpenAI, batch: list) -> list:
    out = []
    for keyword, url in batch:
        try:
            out.append(generate_one(client, keyword, url))
        except Exception as e:
            out.append(f"Error: {e}")
    return out


if uploaded_file and api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Could not initialize OpenAI client: {e}")
        st.stop()

    try:
        keywords_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Validate columns
    required_cols = {"Keyword", "URL"}
    if not required_cols.issubset(set(keywords_df.columns)):
        st.error("CSV must contain columns: 'Keyword' and 'URL'.")
        st.stop()

    keywords_df = keywords_df.dropna(subset=["Keyword", "URL"]).copy()
    keywords_df["Keyword"] = keywords_df["Keyword"].astype(str).str.strip()
    keywords_df["URL"] = keywords_df["URL"].astype(str).str.strip()

    total_keywords = len(keywords_df)
    if total_keywords == 0:
        st.error("No valid rows found (Keyword/URL).")
        st.stop()

    progress_bar = st.progress(0)
    status = st.empty()

    all_responses = []
    futures = []

    # Partition into batches
    batches = []
    for i in range(0, total_keywords, batch_size):
        batch = keywords_df.iloc[i : i + batch_size][["Keyword", "URL"]].values.tolist()
        batches.append(batch)

    completed = 0
    total_batches = len(batches)

    # Concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in batches:
            futures.append(executor.submit(generate_seo_recommendations, client, batch))

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            all_responses.extend(res)

            completed += 1
            progress = min(completed / total_batches, 1.0)
            progress_bar.progress(progress)
            status.write(f"Processed {completed}/{total_batches} batches ({len(all_responses)}/{total_keywords} rows)")

    # Align results to original row count
    # Note: as_completed returns futures in completion order; we appended in that order.
    # This keeps output count correct, but not original row order.
    # To preserve row order, process sequentially or attach indices. Here we fix ordering properly:

    # Re-run with indices to preserve order
    indexed_rows = list(keywords_df[["Keyword", "URL"]].itertuples(index=True, name=None))
    results = [None] * len(indexed_rows)

    def generate_one_indexed(idx_kw_url):
        idx, kw, u = idx_kw_url
        try:
            return idx, generate_one(client, kw, u)
        except Exception as e:
            return idx, f"Error: {e}"

    progress_bar.progress(0)
    status.write("Re-ordering with indexed processing for stable output (fast, cached scraping)...")

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = [executor.submit(generate_one_indexed, t) for t in indexed_rows]
        for fut in concurrent.futures.as_completed(futs):
            idx, rec = fut.result()
            # idx is the original dataframe index; map into positional list by position
            pos = keywords_df.index.get_loc(idx)
            results[pos] = rec

            done += 1
            progress_bar.progress(min(done / len(indexed_rows), 1.0))

    results_df = pd.DataFrame(
        {
            "Keyword": keywords_df["Keyword"].tolist(),
            "URL": keywords_df["URL"].tolist(),
            "Recommendations": results,
        }
    )

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Content Briefs as CSV", csv_bytes, "content-briefs.csv", "text/csv")
    st.dataframe(results_df, use_container_width=True)
