"""
SEC 10-K "Red Flag" Anomaly Detector
Save as: app.py
"""

import streamlit as st
import requests
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import openai
import nltk

# Ensure NLTK tokenizers are downloaded silently to prevent runtime crashes
for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package, quiet=True)

# ---------------------------------------------------------
# 1. SEC Data Fetching & Rate Limiting
# ---------------------------------------------------------
class SECFetcher:
    def __init__(self, user_agent: str):
        # The SEC strict requirement: "User-Agent" must contain your Name and Email
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.last_request_time = 0.0

    def _rate_limit(self):
        """Enforces a max of 8 requests / second (SEC limit is 10/sec)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.125:
            time.sleep(0.125 - elapsed)
        self.last_request_time = time.time()

    def get_cik(self, ticker: str) -> str:
        self._rate_limit()
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        for _, v in resp.json().items():
            if v['ticker'].upper() == ticker.upper():
                return str(v['cik_str']).zfill(10)
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR database.")

    def get_last_two_10k_urls(self, cik: str):
        self._rate_limit()
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        filings = resp.json()['filings']['recent']
        
        urls = []
        for i, form in enumerate(filings['form']):
            if form == '10-K':
                # Reformat the accession number by removing dashes for the URL path
                acc_no_clean = filings['accessionNumber'][i].replace("-", "")
                doc = filings['primaryDocument'][i]
                urls.append(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{doc}")
                if len(urls) == 2:
                    break
        return urls

    def extract_item_1a(self, url: str) -> str:
        self._rate_limit()
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        text = re.sub(r'\s+', ' ', soup.get_text(separator=" "))
        
        # Regex capturing Item 1A (Risk Factors) and stopping before Item 1B or 2
        pattern = re.compile(r'ITEM\s+1A\.\s*RISK\s+FACTORS(.*?)(?:ITEM\s+1B\.\s*UNRESOLVED|ITEM\s+2\.\s*PROPERTIES)', re.IGNORECASE)
        matches = pattern.findall(text)
        
        if not matches:
            return ""
        # We take the longest match to avoid accidentally grabbing the Table of Contents mention
        return max(matches, key=len).strip()

# ---------------------------------------------------------
# 2. Caching & Core ML Logic Wrappers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_sec_data(ticker: str, user_agent: str):
    """Caches SEC fetching so the app doesn't redownload files if you tweak UI sliders."""
    fetcher = SECFetcher(user_agent)
    cik = fetcher.get_cik(ticker)
    urls = fetcher.get_last_two_10k_urls(cik)
    if len(urls) < 2:
        raise ValueError("Could not find two consecutive 10-Ks.")
    return fetcher.extract_item_1a(urls[0]), fetcher.extract_item_1a(urls[1])

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Loads the sentence embedding model into memory once."""
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
        # Only keep sentences below our similarity threshold (i.e., NOT boilerplate)
        if max_score < threshold and len(sent.split()) > 6:
            net_new.append({"Sentence": sent, "Similarity": round(max_score, 3)})

    df = pd.DataFrame(net_new)
    if not df.empty:
        df = df.sort_values(by="Similarity").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def generate_llm_analysis(delta_df: pd.DataFrame, api_key: str) -> str:
    if delta_df.empty:
        return "No material net-new risks detected."
        
    client = openai.OpenAI(api_key=api_key)
    # Take the top 30 most unique sentences to avoid LLM context overflow
    text_block = " ".join(delta_df['Sentence'].tolist()[:30])
    
    prompt = f"""
    Analyze these net-new 10-K risk factor disclosures isolated via semantic diffing.
    1. Categorize the main new risks.
    2. Assign a Severity Score (1-10) based on potential cash flow impact.
    3. Provide a 3-bullet executive summary of what management is newly worried about.
    Disclosures: {text_block}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise hedge fund quantitative analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ---------------------------------------------------------
# 3. Streamlit Application UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="10-K Anomaly Detector", layout="wide", page_icon="📈")
    st.title("SEC 10-K 'Red Flag' Anomaly Detector 🚩")

    with st.sidebar:
        st.header("Configuration")
        
        # Check Streamlit secrets first, otherwise prompt the user
        if "OPENAI_API_KEY" in st.secrets:
            openai_key = st.secrets["OPENAI_API_KEY"]
            st.success("OpenAI Key loaded securely.")
        else:
            openai_key = st.text_input("OpenAI API Key", type="password", help="Required for GPT-4o analysis.")
            
        sec_user_agent = st.text_input("SEC User Agent (Email Required)", value="QuantTeam your-email@domain.com")
        ticker = st.text_input("Target Ticker (e.g., TSLA, META, NVDA)").upper()
        threshold = st.slider("Similarity Threshold", 0.70, 0.99, 0.85, 0.01, help="Sentences with similarity below this score are flagged as new anomalies.")
        
        run_btn = st.button("Run Analysis", type="primary")

    if run_btn:
        if not ticker or not openai_key or "@" not in sec_user_agent:
            st.error("Please provide Ticker, OpenAI Key, and a valid SEC User Agent (must include an email address).")
            return

        try:
            with st.spinner('1/3 Fetching and parsing SEC EDGAR filings...'):
                current_text, prior_text = fetch_sec_data(ticker, sec_user_agent)
                if len(current_text) < 100 or len(prior_text) < 100:
                    st.warning("⚠️ Warning: Could not cleanly extract Item 1A. The SEC format for this company may be non-standard.")

            with st.spinner('2/3 Running NLP Cosine Similarity...'):
                df_delta = compute_semantic_diff(prior_text, current_text, threshold)

            with st.spinner('3/3 Generating AI Risk Report...'):
                analysis = generate_llm_analysis(df_delta, openai_key)

            # Display Metrics
            col1, col2 = st.columns([1, 1])
            curr_sents = len(nltk.sent_tokenize(current_text)) if current_text else 0
            col1.metric("Current 10-K Sentences (Total)", curr_sents)
            col2.metric("Net-New Anomalies (Isolated)", len(df_delta), delta_color="inverse")

            # Display Analyst Report
            st.subheader("🤖 AI Analyst Report")
            st.info(analysis)
            
            # Display Raw Data Frame
            st.subheader("Raw Structural Delta (The 'Anomalies')")
            if not df_delta.empty:
                st.markdown("These sentences appear in the current 10-K but do not match anything in last year's filing above the similarity threshold.")
                st.dataframe(df_delta, use_container_width=True)
            else:
                st.write("No significant structural anomalies found between the two filings.")

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")

if __name__ == "__main__":
    main()
