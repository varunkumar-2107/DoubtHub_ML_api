from app.py import get_answer, get_domain, recommend_query
from app.py import predict_querry, build_qa_index, get_answer
from app.py import model, vectoriser
import streamlit as st

QA_CLUSTERS = 20                   
MAX_DISTANCE = 1.1  

st.set_page_config(page_title="Smart Doubt Solver", page_icon="💡", layout="centered")
st.title("💡 Smart College Doubt Solver")
st.caption("Enter your doubt: get its domain, an answer (if available) and similar queries.")

try:
    model, vectoriser
except Exception as e:
    st.error(f"Could not load model files. Check model / vectorizer.\n\n{e}")
    st.stop()

User_input = st.text_area("Your query", height=120)

if st.button("Solve my doubt", type="primary"):
    User_input = " ".join(User_input.split())
    if not User_input:
        st.warning("Please enter a query first.")
        st.stop()

    domain = get_domain(User_input)
    st.subheader("Predicted domain")
    st.success(domain)

    qa_index = build_qa_index()

    st.subheader("Answer")
    if domain != domain or domain == 'other':
        st.info(f"No answers are available for **{domain}** yet. The dataset only covers {QA_DOMAIN}.")
    elif qa_index is None:
        st.warning(f"Q&A dataset not found at `{QA_PATH}`.")
    else:
        ans = get_answer(User_input, qa_index)
        if ans is None:
            st.info("No close match found in the Q&A dataset for this query.")
        else:
            st.write(ans["answer"])
            st.caption(f"Matched question: *{ans['question']}*")

    st.subheader("Similar queries")
    try:
        with st.spinner("Finding similar queries..."):
            similar = recommend_querry(User_input)
        if similar:
            for i, q in enumerate(similar, 1):
                st.write(f"{i}. {q.strip()}")
        else:
            st.info("No similar queries found.")
    except Exception as e:
        st.error(f"retrain() failed: {e}")