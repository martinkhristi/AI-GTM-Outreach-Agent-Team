import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools

# -----------------------------
# Config
# -----------------------------
INFERCOM_BASE_URL = "https://api.infercom.ai/v1"
DEFAULT_MODEL = "gpt-oss-120b"  # model id on Infercom OpenAI-compatible endpoint


def ensure_tmp_dir() -> None:
    os.makedirs("tmp", exist_ok=True)


def extract_json_or_raise(text: str) -> Dict[str, Any]:
    """
    Extract JSON from a model response.
    Tries strict JSON first; if it fails, extracts the outermost JSON object;
    if that still fails, repairs common JSON issues using json-repair.

    Install:
        pip install json-repair
    """
    raw = (text or "").strip()

    # 1) Try strict parse (raw)
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Extract the outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Failed to parse JSON: No JSON object found.\nResponse was:\n{raw}")

    candidate = raw[start : end + 1]

    # 3) Try strict parse (candidate)
    try:
        return json.loads(candidate)
    except Exception as e:
        # 4) Repair JSON (handles missing commas, bad quotes, trailing commas, etc.)
        try:
            from json_repair import repair_json  # pip install json-repair

            repaired = repair_json(candidate, return_objects=False)
            return json.loads(repaired)
        except Exception:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse was:\n{raw}")


def create_infercom_model(model_id: str = DEFAULT_MODEL) -> OpenAIChat:
    """
    Use Agno's OpenAIChat against Infercom OpenAI-compatible endpoint.
    IMPORTANT: uses SAMBANOVA_API_KEY (Infercom key env var in this app).
    """
    api_key = os.getenv("SAMBANOVA_API_KEY", "")
    if not api_key:
        raise RuntimeError("SAMBANOVA_API_KEY is not set")

    return OpenAIChat(
        id=model_id,
        api_key=api_key,
        base_url=INFERCOM_BASE_URL,
        temperature=0.1,
        top_p=0.1,
    )


# -----------------------------
# Agents
# -----------------------------
def create_company_finder_agent() -> Agent:
    ensure_tmp_dir()
    exa_tools = ExaTools(category="company")
    db = SqliteDb(db_file="tmp/gtm_outreach.db")

    return Agent(
        model=create_infercom_model(DEFAULT_MODEL),
        tools=[exa_tools],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_company_finder",
        debug_mode=True,
        instructions=[
            "You are CompanyFinderAgent. Use ExaTools to search the web for companies that match the targeting criteria.",
            "Return ONLY valid JSON with key 'companies' as a list; respect the requested limit provided in the user prompt.",
            "Each item must have: name, website, why_fit (1-2 lines).",
        ],
    )


def create_contact_finder_agent() -> Agent:
    ensure_tmp_dir()
    exa_tools = ExaTools()
    db = SqliteDb(db_file="tmp/gtm_outreach.db")

    return Agent(
        model=create_infercom_model(DEFAULT_MODEL),
        tools=[exa_tools],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_contact_finder",
        debug_mode=True,
        instructions=[
            "You are ContactFinderAgent. Use ExaTools to find 2-3 relevant decision makers per company and their emails if available.",
            "Prioritize roles from Founder's Office, GTM (Marketing/Growth), Sales leadership, Partnerships/Business Development, and Product Marketing.",
            "Search queries can include patterns like '<Company> email format', 'contact', 'team', 'leadership', and role titles.",
            "If direct emails are not found, infer likely email using common formats (e.g., first.last@domain), but mark inferred=true.",
            "Return ONLY valid JSON with key 'companies' as a list; each has: name, contacts: [{full_name, title, email, inferred}]",
        ],
    )


def get_email_style_instruction(style_key: str) -> str:
    styles = {
        "Professional": "Style: Professional. Clear, respectful, and businesslike. Short paragraphs; no slang.",
        "Casual": "Style: Casual. Friendly, approachable, first-name basis. No slang or emojis; keep it human.",
        "Cold": "Style: Cold email. Strong hook in opening 2 lines, tight value proposition, minimal fluff, strong CTA.",
        "Consultative": "Style: Consultative. Insight-led, frames observed problems and tailored solution hypotheses; soft CTA.",
    }
    return styles.get(style_key, styles["Professional"])


def create_email_writer_agent(style_key: str = "Professional") -> Agent:
    """
    Email agent has the highest chance of "JSON-breaking" output.
    We make it very strict + minified + escaped newlines.
    """
    ensure_tmp_dir()
    db = SqliteDb(db_file="tmp/gtm_outreach.db")
    style_instruction = get_email_style_instruction(style_key)

    return Agent(
        model=create_infercom_model(DEFAULT_MODEL),
        tools=[],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_email_writer",
        debug_mode=False,
        instructions=[
            "You are EmailWriterAgent. Write concise, personalized B2B outreach emails.",
            style_instruction,
            "IMPORTANT OUTPUT RULES:",
            "1) Return ONLY valid JSON. No markdown, no code fences, no extra commentary.",
            "2) Output must be MINIFIED JSON (single line).",
            "3) Escape all newlines inside strings as \\n (do not output literal line breaks).",
            "4) Do NOT use unescaped double quotes inside strings. If you need quotes in the email text, use single quotes instead.",
            "Return ONLY valid JSON with key 'emails' as a list of items: {company, contact, subject, body}.",
            "Length: 120-160 words. Include 1-2 lines of strong personalization referencing research insights (company website and Reddit findings).",
            "CTA: suggest a short intro call; include sender company name and calendar link if provided.",
        ],
    )


def create_research_agent() -> Agent:
    """Agent to gather interesting insights from company websites and Reddit."""
    ensure_tmp_dir()
    exa_tools = ExaTools()
    db = SqliteDb(db_file="tmp/gtm_outreach.db")

    return Agent(
        model=create_infercom_model(DEFAULT_MODEL),
        tools=[exa_tools],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_researcher",
        debug_mode=True,
        instructions=[
            "You are ResearchAgent. For each company, collect concise, valuable insights from:",
            "1) Their official website (about, blog, product pages)",
            "2) Reddit discussions (site:reddit.com mentions)",
            "Summarize 2-4 interesting, non-generic points per company that a human would bring up in an email to show genuine effort.",
            "Return ONLY valid JSON with key 'companies' as a list; each has: name, insights: [strings].",
        ],
    )


# -----------------------------
# Pipeline steps
# -----------------------------
def run_company_finder(
    agent: Agent, target_desc: str, offering_desc: str, max_companies: int
) -> List[Dict[str, str]]:
    prompt = (
        f"Find exactly {max_companies} companies that are a strong B2B fit given the user inputs.\n"
        f"Targeting: {target_desc}\n"
        f"Offering: {offering_desc}\n"
        "For each, provide: name, website, why_fit (1-2 lines)."
    )
    resp: RunOutput = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    companies = data.get("companies", [])
    return companies[: max(1, min(max_companies, 10))]


def run_contact_finder(
    agent: Agent,
    companies: List[Dict[str, str]],
    target_desc: str,
    offering_desc: str,
) -> List[Dict[str, Any]]:
    prompt = (
        "For each company below, find 2-3 relevant decision makers and emails (if available). Ensure at least 2 per company when possible, and cap at 3.\n"
        "If not available, infer likely email and mark inferred=true.\n"
        f"Targeting: {target_desc}\nOffering: {offering_desc}\n"
        f"Companies JSON: {json.dumps(companies, ensure_ascii=False)}\n"
        "Return JSON: {companies: [{name, contacts: [{full_name, title, email, inferred}]}]}"
    )
    resp: RunOutput = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_research(agent: Agent, companies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompt = (
        "For each company, gather 2-4 interesting insights from their website and Reddit that would help personalize outreach.\n"
        f"Companies JSON: {json.dumps(companies, ensure_ascii=False)}\n"
        "Return JSON: {companies: [{name, insights: [string, ...]}]}"
    )
    resp: RunOutput = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_email_writer(
    agent: Agent,
    contacts_data: List[Dict[str, Any]],
    research_data: List[Dict[str, Any]],
    offering_desc: str,
    sender_name: str,
    sender_company: str,
    calendar_link: Optional[str],
) -> List[Dict[str, str]]:
    prompt = (
        "Write personalized outreach emails for the following contacts.\n"
        f"Sender: {sender_name} at {sender_company}.\n"
        f"Offering: {offering_desc}.\n"
        f"Calendar link: {calendar_link or 'N/A'}.\n"
        f"Contacts JSON: {json.dumps(contacts_data, ensure_ascii=False)}\n"
        f"Research JSON: {json.dumps(research_data, ensure_ascii=False)}\n"
        "Return JSON with key 'emails' as a list of {company, contact, subject, body}."
    )
    resp: RunOutput = agent.run(prompt)

    # Helpful debug if it still fails:
    # raw = str(resp.content)
    # st.write("DEBUG raw response (email agent):")
    # st.code(raw)

    data = extract_json_or_raise(str(resp.content))
    return data.get("emails", [])


# -----------------------------
# Streamlit UI
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Infercom GTM Outreach", layout="wide")

    # Sidebar: API keys
    st.sidebar.header("API Configuration")
    sambanova_key = st.sidebar.text_input(
        "Infercom API Key",
        type="password",
        value=os.getenv("SAMBANOVA_API_KEY", ""),
        help="Stored only in your Streamlit session as env var SAMBANOVA_API_KEY.",
    )
    exa_key = st.sidebar.text_input(
        "Exa API Key",
        type="password",
        value=os.getenv("EXA_API_KEY", ""),
        help="Used for web/company/contact research via ExaTools.",
    )

    if sambanova_key:
        os.environ["SAMBANOVA_API_KEY"] = sambanova_key
    if exa_key:
        os.environ["EXA_API_KEY"] = exa_key

    if not sambanova_key or not exa_key:
        st.sidebar.warning("Enter both keys to enable the app")

    # Branding
    st.title("Infercom – Powering the Next Wave of Generative AI in Europe 🇪🇺")
    st.caption("Provider of specialized AI Inference Infrastructure for Europe")

    st.info(
        "This app runs a multi-agent workflow to find target companies, identify decision makers, "
        "gather real research insights (website + Reddit), and generate tailored B2B outreach emails — "
        "using Infercom’s OpenAI-compatible EU-hosted inference endpoint.\n\n"
        "⚡ High-performance inference\n"
        "🇪🇺 100% European data residency\n"
        "🔐 Built for compliance (GDPR / EU AI Act) and enterprise workloads"
    )

    col1, col2 = st.columns(2)
    with col1:
        target_desc = st.text_area(
            "Target companies (industry, size, region, tech, etc.)",
            height=110,
            placeholder="Example: Nordic B2B SaaS (50–500 employees) using HubSpot or Salesforce; Denmark/Sweden/Norway",
        )
        offering_desc = st.text_area(
            "Your product/service offering (1–3 sentences)",
            height=110,
            placeholder="Example: We help teams automate inbound lead qualification and personalize outreach using compliant AI workflows.",
        )

    with col2:
        sender_name = st.text_input("Your name", value="Sales Team")
        sender_company = st.text_input("Your company", value="Our Company")
        calendar_link = st.text_input("Calendar link (optional)", value="")
        num_companies = st.number_input("Number of companies", min_value=1, max_value=10, value=5)
        email_style = st.selectbox(
            "Email style",
            options=["Professional", "Casual", "Cold", "Consultative"],
            index=0,
            help="Choose the tone/format for the generated emails",
        )

    if st.button("Start Outreach", type="primary"):
        if not sambanova_key or not exa_key:
            st.error("Please provide Infercom API key + Exa API key in the sidebar")
        elif not target_desc or not offering_desc:
            st.error("Please fill in target companies and offering")
        else:
            progress = st.progress(0)
            stage_msg = st.empty()
            details = st.empty()

            try:
                company_agent = create_company_finder_agent()
                contact_agent = create_contact_finder_agent()
                research_agent = create_research_agent()
                email_agent = create_email_writer_agent(email_style)

                stage_msg.info("1/4 Finding companies...")
                companies = run_company_finder(
                    company_agent,
                    target_desc.strip(),
                    offering_desc.strip(),
                    int(num_companies),
                )
                progress.progress(25)
                details.write(f"Found {len(companies)} companies")

                stage_msg.info("2/4 Finding contacts (2–3 per company)...")
                contacts_data = (
                    run_contact_finder(contact_agent, companies, target_desc.strip(), offering_desc.strip())
                    if companies
                    else []
                )
                progress.progress(50)
                details.write(f"Collected contacts for {len(contacts_data)} companies")

                stage_msg.info("3/4 Researching insights (website + Reddit)...")
                research_data = run_research(research_agent, companies) if companies else []
                progress.progress(75)
                details.write(f"Compiled research for {len(research_data)} companies")

                stage_msg.info("4/4 Writing personalized emails...")
                emails = (
                    run_email_writer(
                        email_agent,
                        contacts_data,
                        research_data,
                        offering_desc.strip(),
                        sender_name.strip() or "Sales Team",
                        sender_company.strip() or "Our Company",
                        calendar_link.strip() or None,
                    )
                    if contacts_data
                    else []
                )
                progress.progress(100)
                details.write(f"Generated {len(emails)} emails")

                st.session_state["gtm_results"] = {
                    "companies": companies,
                    "contacts": contacts_data,
                    "research": research_data,
                    "emails": emails,
                }
                stage_msg.success("Completed ✅")

            except Exception as e:
                stage_msg.error("Pipeline failed ❌")
                st.error(str(e))

    results = st.session_state.get("gtm_results")
    if results:
        companies = results.get("companies", [])
        contacts = results.get("contacts", [])
        research = results.get("research", [])
        emails = results.get("emails", [])

        st.subheader("Top target companies")
        if companies:
            for idx, c in enumerate(companies, 1):
                st.markdown(f"**{idx}. {c.get('name','')}**  ")
                st.write(c.get("website", ""))
                st.write(c.get("why_fit", ""))
        else:
            st.info("No companies found")

        st.divider()
        st.subheader("Contacts found")
        if contacts:
            for c in contacts:
                st.markdown(f"**{c.get('name','')}**")
                for p in c.get("contacts", [])[:3]:
                    inferred = " (inferred)" if p.get("inferred") else ""
                    st.write(f"- {p.get('full_name','')} | {p.get('title','')} | {p.get('email','')}{inferred}")
        else:
            st.info("No contacts found")

        st.divider()
        st.subheader("Research insights")
        if research:
            for r in research:
                st.markdown(f"**{r.get('name','')}**")
                for insight in r.get("insights", [])[:4]:
                    st.write(f"- {insight}")
        else:
            st.info("No research insights")

        st.divider()
        st.subheader("Suggested Outreach Emails")
        if emails:
            for i, e in enumerate(emails, 1):
                with st.expander(f"{i}. {e.get('company','')} → {e.get('contact','')}"):
                    st.write(f"Subject: {e.get('subject','')}")
                    st.text(e.get("body", ""))
        else:
            st.info("No emails generated")

    st.divider()
    st.caption("Infercom – Specialized AI Inference Infrastructure for Europe")


if __name__ == "__main__":
    main()
