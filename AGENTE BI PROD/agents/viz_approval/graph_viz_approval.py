from typing import Any, Dict
from langchain_core.messages import AIMessage


def viz_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo HITL para aprobar visualización cuando no fue solicitada explícitamente.
    
    Nota: langgraph 0.1.19 no soporta interrupt(). 
    Para el MVP, se auto-rechaza la visualización si no fue solicitada explícitamente.
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

    # MVP: auto-rechazar para evitar bloqueo del flujo
    # En versiones futuras, implementar HITL real con interrupt() de langgraph >= 1.x
    approved = False
    logger = __import__("logging").getLogger(__name__)
    logger.info("[Viz Approval] Auto-rechazo en MVP (HITL no disponible en langgraph 0.1.19)")

    return {
        "viz_approved": approved,
        "last_agent": "viz_approval",
        "messages": [AIMessage(content="[Viz Approval] Visualización no aprobada automáticamente en MVP")]
    }
