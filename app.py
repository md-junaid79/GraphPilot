import streamlit as st
from importlib import reload
import agent.graph as agent_graph

st.set_page_config(page_title="GraphPilot Agent UI ", layout="centered")

st.title("GraphPilot — Autonomous coding assistant")

prompt = st.text_area(
    "User prompt",
    value="",
    placeholder="e.g. Build a colourful modern todo app in HTML, CSS and JavaScript",
    height=120,
)

recursion_limit = st.number_input("Recursion limit", min_value=1, max_value=100, value=3)

run_now = st.button("Run agent")

if run_now:
    st.info("Invoking agent graph — this may call your LLM and take some time")
    try:
        # reload the module to pick up local edits without restarting Streamlit
        reload(agent_graph)
        graph = agent_graph.graph
        # invoke the graph
        result = graph.invoke({"user_prompt": prompt}, {"recursion_limit": int(recursion_limit)})
        st.success("Run finished")
        st.subheader("Final State")
        st.json(result)
    except Exception as e:
        st.error(f"Agent run failed: {e}")
        raise

st.markdown("---")
st.write("Notes:")
st.write("- This UI will call your configured LLM when you press Run. Ensure you have credentials set in your environment.")
st.write("- If the graph run is long-running you may want to run it from a terminal instead of Streamlit.")
