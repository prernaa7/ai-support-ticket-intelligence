
import requests
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

st.set_page_config(
    page_title="AI Support Ticket Intelligence",
    page_icon="🎫",
    layout="wide",
)

st.title("AI Support Ticket Intelligence")
st.write("Analyze support tickets using machine learning and AI.")

description = st.text_area(
    "Enter a support ticket",
    placeholder="Describe the customer's issue...",
    height=150,
)

top_k = st.slider("Number of similar results", min_value=1, max_value=5, value=3)


def call_api(endpoint):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json={"description": description, "top_k": top_k},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
        return None


if st.button("Analyze Ticket", type="primary"):
    if not description.strip():
        st.warning("Please enter a ticket description.")
    else:
        with st.spinner("Analyzing ticket..."):
            category = call_api("predict/category")
            priority = call_api("predict/priority")
            summary = call_api("summarize")
            similar = call_api("similar")
            resolution = call_api("resolution")
            knowledge = call_api("knowledge-base")

        if category and priority:
            col1, col2 = st.columns(2)
            col1.metric("Predicted Category", category["category"])
            col2.metric("Predicted Priority", priority["priority"])

        if summary:
            st.subheader("Ticket Summary")
            st.write(summary["summary"])

        if similar:
            st.subheader("Similar Historical Tickets")
            for item in similar.get("results", []):
                with st.expander(
                    f'{item["ticket_id"]} — {item["similarity"]:.1%} similarity'
                ):
                    st.write(f'**Category:** {item["category"]}')
                    st.write(f'**Priority:** {item["priority"]}')
                    st.write(f'**Description:** {item["description"]}')
                    st.write(f'**Resolution:** {item["resolution"]}')

        if resolution:
            st.subheader("Suggested Resolution")
            st.write(resolution["suggested_resolution"])
            st.caption(resolution["note"])

        if knowledge:
            st.subheader("Knowledge Base Articles")
            for article in knowledge.get("articles", []):
                with st.expander(
                    f'{article["article_id"]} — {article["similarity"]:.1%} similarity'
                ):
                    st.write(f'**Category:** {article["category"]}')
                    st.write(article["content"])
