# 🚩 SEC 10-K "Red Flag" Anomaly Detector

> **An institutional-grade NLP pipeline that strips boilerplate legal jargon from SEC 10-K filings to isolate, quantify, and score structural narrative shifts in corporate risk.**

Traditional quantitative sentiment analysis often relies on keyword frequency (e.g., counting the word "lawsuit"). This approach is easily gamed by corporate lawyers and fails to capture structural shifts in a company's fundamental outlook. 

This tool solves that problem. By applying **Sentence Transformers (Cosine Similarity)**, the pipeline compares a company's current 10-K (Item 1A) against its prior year's filing. It automatically discards up to 90% of the text as "legal boilerplate" and isolates the true **Delta**—the net-new disclosures management was forced to add. It then routes these hidden anomalies to an LLM to classify and score the severity of the new risks.

## 🧠 System Architecture

1. **Ingestion Engine:** Dynamically queries the SEC EDGAR REST API to map stock tickers to CIKs, fetching the two most recent 10-K filings.
2. **Regex Parsing:** Extracts Item 1A ("Risk Factors") while avoiding Table of Contents artifacts.
3. **Semantic Diffing (Local NLP):** Tokenizes text into sentences and vectorizes them using Hugging Face's `all-MiniLM-L6-v2`. Calculates a cosine similarity matrix to filter out year-over-year text matches >85%.
4. **LLM Inference:** Passes the highly concentrated "Delta Text" to OpenAI's `gpt-4o` to generate a severity score (1-10) and a concise analyst summary.
5. **UI & Caching:** Streamlit frontend with native data caching to prevent redundant API calls and optimize token spend.

## 🚀 Quickstart

### Prerequisites
* Python 3.9+
* An OpenAI API Key

### Installation

1. Clone the repository:
```bash
git clone [https://github.com/YOUR_USERNAME/sec-10k-anomaly-detector.git](https://github.com/YOUR_USERNAME/sec-10k-anomaly-detector.git)
cd sec-10k-anomaly-detector
