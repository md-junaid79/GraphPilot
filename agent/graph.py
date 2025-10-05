from langgraph.graph import StateGraph,START,END
from langchain.globals import set_verbose, set_debug
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
# from langchain_ollama.llms import OllamaLLM

from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from agent.prompts import planner_prompt, architect_prompt, coder_system_prompt
from agent.states import Plan, TaskPlan, CoderState
from agent.tools import run_cmd,write_file, read_file, get_current_directory, list_files

set_debug(True)
set_verbose(True)

load_dotenv()
try:
    llm = ChatOllama(model="codellama")
    print("connection success ✅")
except Exception as e:
    print("Error initializing Ollama. Ensure Ollama is running and 'codellama' is available. Error: {e}")
    llm=None


def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}      #THIS will be added to state dict

def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""
    plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan=plan.model_dump_json()))
    if resp is None:
        raise ValueError("Planner did not return a valid response.")

    resp.plan = plan
    print(resp.model_dump_json())
    return {"task_plan": resp}


def coder_agent(state):
    """Implements each task in the TaskPlan sequentially using tools (coder_tools)."""
    coder_state = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    existing_content = read_file.run(current_task.filepath)

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)

    react_agent.invoke({"messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt}]})

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}
    
# building the grpah
builder = StateGraph(dict)

# add nodes
builder.add_node("planner",planner_agent)
builder.add_node("architect",architect_agent)
builder.add_node("coder",coder_agent)

# add edges
builder.add_edge(START,"planner")
builder.add_edge("planner","architect")
builder.add_edge("architect","coder")
builder.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)
# compile final grpah
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"user_prompt": "Build a colourful modern todo app in html css and javascript"},{"recursion_limit": 10})
    print("Final State:", result)