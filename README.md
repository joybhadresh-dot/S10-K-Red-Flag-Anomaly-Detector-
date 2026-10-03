import streamlit as st
import requests
import re
import time
import logging
from typing import List, Optional
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import openai
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure NLTK tokenizers are available silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ==========================================
# 1. SEC EDGAR Data Ingestion Layer
# ==========================================
class SECRateLimiter:
    """Enforces the SEC's strict 10 requests/second limit. We target 8/sec for safety."""
    def __init__(self, max_per_second: int = 8):
        self.min_interval = 1.0 / max_per_second
        self.last_request_time = 0.0

    def wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

class SECFetcher:
    def __init__(self, user_agent: str):
        """SEC requires: 'CompanyName AdminContact@domain.com'"""
        if "@" not in user_agent:
            raise ValueError("SEC EDGAR requires a valid email in the User-Agent header.")
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.limiter = SECRateLimiter()

    def _get(self, url: str) -> requests.Response:
        """Internal wrapper to enforce rate limits and handle 403/429s."""
        self.limiter.wait()
        response = requests.get(url, headers=self.headers, timeout=15)
        response.raise_for_status()
        return response

    def get_cik_from_ticker(self, ticker: str) -> str:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(url).json()
        for _, value in data.items():
            if value['ticker'].upper() == ticker.upper():
                return str(value['cik_str']).zfill(10)
        raise ValueError(f"Ticker {ticker} not found.")

    def get_recent_10k_urls(self, cik: str, count: int = 2) -> List[str]:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        filings = self._get(url).json()['filings']['recent']
        
        urls = []
        for i, form in enumerate(filings['form']):
            if form == '10-K':
                accession_no_clean = filings['accessionNumber'][i].replace("-", "")
                document = filings['primaryDocument'][i]
                urls.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_clean}/{document}")
                if len(urls) == count:
                    break
        return urls

    def extract_item_1a(self, url: str) -> str:
        response = self._get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = re.sub(r'\s+', ' ', soup.get_text(separator=" "))
        
        # Regex capturing Item 1A to Item 1B
        pattern = re.compile(r'ITEM\s+1A\.\s*RISK\s+FACTORS(.*?)(?:ITEM\s+1B\.\s*UNRESOLVED|ITEM\s+2\.\s*PROPERTIES)', re.IGNORECASE)
        matches = pattern.findall(text)
        
        if not matches:
            return ""
        return max(matches, key=len).strip()

# ==========================================
# 2. Caching Wrappers for UI Performance
# ==========================================
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_sec_data(ticker: str, user_agent: str):
    """Caches the SEC network calls so UI interactions don't trigger re-downloads."""
    fetcher = SECFetcher(user_agent)
    cik = fetcher.get_cik_from_ticker(ticker)
    urls = fetcher.get_recent_10k_urls(cik, count=2)
    if len(urls) < 2:
        raise ValueError("Could not find two consecutive 10-Ks.")
    return fetcher.extract_item_1a(urls[0]), fetcher.extract_item_1a(urls[1])

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Loads the ML model once and keeps it in memory."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=False)
def compute_semantic_diff(prior_text: str, current_text: str, threshold: float) -> pd.DataFrame:
    model = load_embedding_model()
    prior_sents = nltk.sent_tokenize(prior_text)
    current_sents = nltk.sent_tokenize(current_text)

    if not prior_sents or not current_sents:
        return pd.DataFrame()

    prior_emb = model.encode(prior_sents, convert_to_tensor=True)
    current_emb = model.encode(current_sents, convert_to_tensor=True)
    cosine_scores = util.cos_sim(current_emb, prior_emb)

    net_new = []
    for i, sent in enumerate(current_sents):
        max_score = cosine_scores[i].max().item()
        if max_score < threshold and len(sent.split()) > 6:
            net_new.append({"Sentence": sent, "Similarity": round(max_score, 3)})

    return pd.DataFrame(net_new).sort_values(by="Similarity")

@st.cache_data(show_spinner=False)
def generate_llm_analysis(delta_df: pd.DataFrame, api_key: str) -> str:
    if delta_df.empty:
        return "No material net-new risks detected."
        
    client = openai.OpenAI(api_key=api_key)
    text_block = " ".join(delta_df['Sentence'].tolist()[:30])
    
    prompt = f"""
    Analyze these net-new 10-K risk factor disclosures. 
    1. Categorize the main new risks.
    2. Assign a Severity Score (1-10) based on cash flow impact.
    3. Provide a 3-bullet executive summary.
    Disclosures: {text_block}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise hedge fund analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ==========================================
# 3. Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="10-K Anomaly Detector", layout="wide")
    st.title("SEC 10-K 'Red Flag' Anomaly Detector 🚩")

    with st.sidebar:
        st.header("Configuration")
        openai_key = st.text_input("OpenAI API Key", type="password")
        sec_user_agent = st.text_input("SEC User Agent (Requires Email)", value="YourName contact@domain.com")
        ticker = st.text_input("Target Ticker (e.g., TSLA)").upper()
        threshold = st.slider("Boilerplate Similarity Threshold", 0.70, 0.99, 0.85, 0.01)
        run_btn = st.button("Run Analysis", type="primary")

    if run_btn:
        if not ticker or not openai_key or "@" not in sec_user_agent:
            st.error("Please provide Ticker, OpenAI Key, and a valid SEC User Agent (must include email).")
            return

        try:
            with st.spinner('1/3 Fetching and parsing SEC EDGAR filings...'):
                current_text, prior_text = fetch_sec_data(ticker, sec_user_agent)

            with st.spinner('2/3 Running NLP Cosine Similarity...'):
                df_delta = compute_semantic_diff(prior_text, current_text, threshold)

            with st.spinner('3/3 Generating AI Risk Report...'):
                analysis = generate_llm_analysis(df_delta, openai_key)

            col1, col2 = st.columns([1, 1])
            col1.metric("Current 10-K Sentences", len(nltk.sent_tokenize(current_text)))
            col2.metric("Net-New Anomalies", len(df_delta), delta_color="inverse")

            st.subheader("🤖 Analyst Report")
            st.info(analysis)
            
            st.subheader("Raw Structural Delta")
            st.dataframe(df_delta, use_container_width=True)

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")

if __name__ == "__main__":
    main()
