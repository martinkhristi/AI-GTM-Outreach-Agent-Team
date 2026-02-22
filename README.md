# Infercom GTM Outreach (Streamlit + Agno + Exa)

A multi-agent Streamlit app that automates B2B outreach research and email drafting using:
- **Infercom** (OpenAI-compatible endpoint, EU-hosted inference)
- **Agno** (agent orchestration + memory + SQLite run history)
- **Exa** (company/contact discovery + web & Reddit research)

The pipeline:
1) Find target companies  
2) Find 2–3 decision-makers per company  
3) Collect research insights (website + Reddit)  
4) Generate personalized outreach emails in the selected style  

---

## Features

- ✅ Company discovery via Exa (category: `company`)
- ✅ Contact discovery (roles: GTM, Sales, Partnerships, Product Marketing, Founders Office)
- ✅ Research insights from:
  - official websites (about/blog/product)
  - Reddit discussions (site:reddit.com)
- ✅ Email generation with strict JSON output rules to reduce parsing failures
- ✅ Local persistence using SQLite (`tmp/gtm_outreach.db`)
- ✅ Streamlit UI with style selector: Professional / Casual / Cold / Consultative

---

## Tech Stack

- Python
- Streamlit
- Agno
- ExaTools (Exa)
- Infercom OpenAI-compatible API (`https://api.infercom.ai/v1`)
- SQLite (Agno `SqliteDb`)
- Optional: `json-repair` for resilient JSON parsing

---

## Project Structure

Typical minimal setup:
