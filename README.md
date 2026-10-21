# NABH Intelligence Assistant

A premium Streamlit workspace for hospital quality and accreditation teams. The app combines semantic retrieval and BM25 keyword retrieval to locate evidence passages from an authorized NABH source PDF and produces answers with source references.

## Features

- Enterprise-style, responsive Streamlit UI
- Hybrid semantic + keyword retrieval
- Evidence passages with source labels/page references
- Suggested accreditation-readiness questions
- Optional Neo4j graph retrieval
- Supports local `.env` and Streamlit Cloud Secrets for the OpenAI API key

## Repository contents

- `app.py` — Streamlit application
- `src/ingestion.py` — PDF extraction and chunking
- `src/retrieval.py` — Hybrid retrieval
- `src/answering.py` — Answer generation
- `src/graph_store.py` — Optional Neo4j support
- `requirements.txt` — Python dependencies

## Run locally

1. Create a virtual environment and activate it.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
4. Place the authorized PDF at `knowledge_base/NABH_Hospital_Accreditation_Standards.pdf`.
5. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Select **Create app** and choose repository `vinonagg/nabh-intelligence-assistant`, branch `main`, main file `app.py`.
3. In app settings, add a secret:

   ```toml
   OPENAI_API_KEY = "your-real-key"
   ```

4. Deploy the app.

### Important: the PDF is not included

This repository intentionally excludes the NABH standards PDF. A cloud deployment cannot read a PDF that exists only on a developer's Mac. The app will show a clear document-not-connected notice until the source file is made available to the cloud runtime through a method you're authorized to use. Do not commit a copyrighted standards document to a public repository unless you have explicit redistribution permission. Do not add API keys or `.env` files to GitHub.

## Secrets and data handling

- Never commit API keys, `.env`, or `.streamlit/secrets.toml`.
- The app sends retrieved evidence passages to the configured OpenAI model to generate answers.
- Do not use identifiable patient data in prompts or documents.
- AI output is an aid only; verify clinical, accreditation, and legal decisions against current official sources.

## Troubleshooting

- **Missing PDF:** place the authorized file at the exact path above or configure an approved private cloud storage integration.
- **OpenAI key not detected:** verify `OPENAI_API_KEY` in Streamlit app Secrets, or enter it in the sidebar for a temporary session.
- **Dependency or build errors:** check the app logs and confirm the selected Python version is supported by the dependencies.
