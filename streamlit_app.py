import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Gemini RAG Document Assistant",
    layout="wide",
)

# ---------------------------
# Custom Styling (Dark Modern Look)
# ---------------------------
st.markdown(
    """
<style>
.main {
    background-color: #0E1117;
}
section[data-testid="stSidebar"] {
    background-color: #1A1C23;
}
h1, h2, h3 {
    font-weight: 600;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# SIDEBAR - CONTROLS
# ---------------------------
st.sidebar.title("⚙️ Controls")

# Upload Section
st.sidebar.subheader("📤 Upload Document")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

if uploaded_file:
    if st.sidebar.button("Upload & Process"):
        with st.sidebar:
            with st.spinner("Processing..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                }

                try:
                    response = requests.post(f"{API_URL}/upload", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        st.success("✅ Uploaded successfully!")
                        st.write(f"Chunks added: {data['chunks_added']}")
                    else:
                        st.error("Upload failed")

                except Exception as e:
                    st.error(f"Error: {e}")

# Clear Database Button
st.sidebar.markdown("---")
if st.sidebar.button("🗑 Clear Database"):
    try:
        response = requests.delete(f"{API_URL}/clear")
        if response.status_code == 200:
            st.sidebar.success("Database cleared ✅")
        else:
            st.sidebar.error("Failed to clear database")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ---------------------------
# MAIN PAGE - CHAT
# ---------------------------
st.title("📚 Gemini RAG Document Assistant")
st.markdown("Upload documents and ask questions using your RAG system.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query", json={"question": prompt, "top_k": 3}
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]

                    st.markdown(answer)

                    # Show sources
                    if data["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(data["sources"], 1):
                                st.markdown(
                                    f"**{i}. {source['filename']} "
                                    f"(Score: {source['score']:.4f})**"
                                )
                                st.write(source["text"])
                                st.markdown("---")

                    # Save assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                else:
                    st.error("Error getting response")

            except Exception as e:
                st.error(f"Error: {e}")
