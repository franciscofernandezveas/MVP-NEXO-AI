from typing import Any, Dict
from langgraph.types import interrupt
from langchain_core.messages import AIMessage


def viz_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo HITL para aprobar visualización cuando no fue solicitada explícitamente
    """
    # Contar resultados aptos para visualización
    suitable_results = [
        r for r in state.get("sql_results", [])
        if getattr(r, "can_answer", False) and len(getattr(r, "rows", [])) > 0
    ]
    
    if not suitable_results:
        # No hay datos para visualizar, continuar sin gráfico
        return {
            "viz_approved": False,
            "last_agent": "viz_approval",
            "messages": [AIMessage(content="[Viz Approval] No hay datos aptos para visualización")]
        }

    # Preguntar al usuario
    human_response = interrupt({
        "type": "visualization_approval",
        "message": f"Se encontraron {len(suitable_results)} conjunto(s) de datos que pueden ser visualizados. ¿Deseas generar un gráfico?",
        "options": ["sí", "no"]
    })

    approved = False
    if isinstance(human_response, dict):
        approved = human_response.get("action") == "approve"
    elif isinstance(human_response, str):
        approved = human_response.lower() in ("approve", "yes", "si", "s", "true", "y", "sí")

    return {
        "viz_approved": approved,
        "last_agent": "viz_approval",
        "messages": [AIMessage(content=f"[HITL] Usuario {'APROBÓ' if approved else 'RECHAZÓ'} visualización")]
    }
