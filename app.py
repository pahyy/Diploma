import streamlit as st

from main import RetailAnalyticsHub

st.set_page_config(page_title="Prodajni analitik", page_icon="🛒", layout="centered")

EXAMPLE_QUESTIONS = [
    "Kakšna je bila prodaja v Ljubljani v letu 2023?",
    "Primerjaj prodajo elektronike v Ljubljani in Mariboru",
    "Kako učinkovite so bile promocije tipa Flash Sale?",
    "Ali so v prodaji leta 2024 kakšne anomalije?",
    "Kateri so najbolje prodajani izdelki?",
]


@st.cache_resource(show_spinner="Nalagam podatke in modele ...")
def get_hub():
    # cache_resource -> the 1M-row CSV is loaded only once, not on every
    # interaction (Streamlit re-runs this whole script on each message).
    return RetailAnalyticsHub()


hub = get_hub()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛒 Prodajni analitik")
    st.caption(
        "Agent za analizo prodajnih podatkov. Vprašaj v slovenščini - "
        "razume trende, primerjave, promocije in anomalije ter si zapomni "
        "kontekst pogovora."
    )

    st.subheader("Primeri vprašanj")
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, use_container_width=True):
            st.session_state.pending_question = question

    st.divider()
    if st.button("🗑️ Nov pogovor", use_container_width=True):
        st.session_state.messages = []
        hub.parser.history.clear()  # forget the conversation context too
        st.rerun()

# ---------------------------------------------------------------- history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("warning"):
            st.warning(message["warning"])
        if message.get("chart_path"):
            st.image(message["chart_path"])
        if message.get("analysis") is not None:
            with st.expander("Podrobnosti analize (JSON)"):
                st.json(message["analysis"])

# ---------------------------------------------------------------- input
prompt = st.chat_input("Vprašaj o prodajnih podatkih ...")

# A click on an example button behaves exactly like typing that question
if not prompt and "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analiziram ..."):
            response = hub.process_query(prompt)

        if response is None:
            content = ("Žal vprašanja nisem uspel obdelati. Poskusi ga "
                       "preoblikovati (npr. dodaj obdobje, mesto ali izdelek).")
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
        else:
            st.markdown(response["summary"])
            if response.get("warning"):
                st.warning(response["warning"])
            if response.get("chart_path"):
                st.image(response["chart_path"])
            with st.expander("Podrobnosti analize (JSON)"):
                st.json(response["analysis"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["summary"],
                "warning": response.get("warning"),
                "chart_path": response.get("chart_path"),
                "analysis": response["analysis"],
            })