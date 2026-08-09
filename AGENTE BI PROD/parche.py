import re
from pathlib import Path

files_to_patch = [
    "core/orchestrator.py",
    "agents/planner/graph_planner.py",
    "agents/analyst/graph_analyst.py",
    "agents/supervisor/graph_supervisor.py",
    "agents/sql_agent/graph_sql_agent.py",
    "agents/research/research_node.py",
]

pattern = re.compile(
    r"def\s+("
    r"detect_chitchat_node|chitchat_node|build_harness_context_node|"
    r"sql_agent_wrapper|viz_agent_node|forecaster_node|researcher_node|"
    r"safe_supervisor_node|planner_node|analyst_node|supervisor_node|"
    r"sql_fetch_schema|sql_generate_query|sql_execute_query|sql_validate_and_package|"
    r"research_node"
    r")\(([^)]*?)\)\s*->"
)

for file_path in files_to_patch:
    p = Path(file_path)
    if not p.exists():
        print(f"⚠️  No encontrado: {file_path}")
        continue

    content = p.read_text(encoding="utf-8")

    def add_kwargs(match):
        func_name = match.group(1)
        args = match.group(2).strip()
        if "**kwargs" in args:
            return match.group(0)
        new_args = f"{args}, **kwargs" if args else "**kwargs"
        return f"def {func_name}({new_args}) ->"

    new_content, count = pattern.subn(add_kwargs, content)
    if count:
        p.write_text(new_content, encoding="utf-8")
        print(f"✅ {file_path}: {count} funciones parcheadas")
    else:
        print(f"ℹ️  {file_path}: sin cambios")
