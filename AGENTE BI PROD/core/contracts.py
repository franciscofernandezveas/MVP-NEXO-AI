from typing import Any, List, Optional, Literal, Dict
from pydantic import BaseModel, Field, field_validator


class SQLPayload(BaseModel):
    """Instrucción técnica individual."""
    task_id: str = Field(default="1", description="Identificador de la subtarea")
    task: str = Field(..., description="Descripción técnica de la tarea SQL")
    metrics: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    filters_description: str = Field(default="", description="Filtros en lenguaje natural")
    time_window: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    execution_strategy: str = Field(
        default="single_view",
        description="daily, compare_periods, historical, monthly, by_branch, by_product..."
    )
    candidate_views: List[str] = Field(
        default_factory=list,
        description="Shortlist de vistas semantic. permitidas para esta tarea"
    )
    preferred_view: Optional[str] = Field(
        default=None,
        description="Vista elegida por el planner como principal (debe estar en candidate_views)"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs de tareas previas cuyos resultados requiere esta subtarea"
    )


class ResearchPlan(BaseModel):
    """Plan de exploración profunda interna (múltiples queries a la BD)."""
    goal: str = Field(..., description="Objetivo del informe de investigación")
    metrics_to_cover: List[str] = Field(default_factory=list)
    dimensions_to_cover: List[str] = Field(default_factory=list)
    time_windows: List[str] = Field(default_factory=list)
    sections: List[str] = Field(
        default_factory=list,
        description="Secciones sugeridas del informe final"
    )
    tasks: List[SQLPayload] = Field(
        ...,
        description="Lista de queries de exploración a ejecutar en la base de datos"
    )


class PlannerContract(BaseModel):
    """Contrato del planner con soporte para demand forecast."""
    intent: str
    goal: str
    question_type: Literal[
        "aggregation", "comparison", "trend", "lookup", "unknown",
        "multi_query", "deep_research", "demand_forecast"
    ]
    metrics: List[str]
    dimensions: List[str]
    filters: str
    time_window: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    tasks: List[SQLPayload] = Field(...)
    confidence: float = Field(ge=0.0, le=1.0)
    visualization_candidate: bool = Field(default=False)
    chart_type_hint: Optional[str] = Field(default="auto")
    needs_followup: bool = Field(default=False)
    followup_reason: Optional[str] = Field(default=None)


class SQLContract(BaseModel):
    """Contrato estricto de salida del SQL Agent."""
    task_id: str = Field(default="1")
    status: Literal["success", "error", "partial", "needs_clarification"]
    generated_sql: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    error_message: Optional[str] = None
    schema_used: List[str] = Field(default_factory=list)
    can_answer: bool = False
    reasoning: str = ""
    needs_followup: bool = False
    warnings: List[str] = Field(default_factory=list)

    allowed_views: List[str] = Field(
        default_factory=list,
        description="Vistas que el harness/planner autorizaron para esta ejecución"
    )
    preferred_view: Optional[str] = Field(
        default=None,
        description="Vista que el planner sugirió como principal"
    )
    semantic_context_used: str = Field(
        default="",
        description="Contexto semántico inyectado al SQL subgraph (recortado para log)"
    )
    query_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confianza de la ejecución (1.0 = éxito verificado)"
    )
    reason_for_view_choice: str = Field(
        default="",
        description="Explicación de por qué se eligió la vista o por qué falló"
    )


class ResearchContract(BaseModel):
    """Resultado del nodo researcher."""
    status: Literal["success", "partial", "error"]
    plan: Optional[ResearchPlan] = None
    task_results: List[SQLContract] = Field(default_factory=list)
    findings: str = ""
    metrics_covered: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    needs_followup: bool = False


class SupervisorDecision(BaseModel):
    """Decisión del supervisor con soporte para forecaster."""
    reasoning: str
    next_agent: Literal[
        "planner", "sql_agent", "analyst", "viz_agent",
        "render_plotly", "viz_approval", "researcher",
        "forecaster",
        "FINISH"
    ]


class ForecastRequest(BaseModel):
    """Parámetros para solicitar una predicción de demanda."""
    producto: str
    sede: str
    n_dias: int = 7
    fecha_inicio: Optional[str] = None


class ForecastResult(BaseModel):
    """Resultado estructurado de una predicción de demanda."""
    forecasts: List[Dict[str, Any]] = Field(default_factory=list)
    modelo_version: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    safety_stock: float = 0.0
    error: Optional[str] = None


class VizSpecContract(BaseModel):
    """Contrato de salida del Viz Agent."""
    status: Literal["success", "error"]
    chart_type: Optional[Literal["bar", "line", "pie", "scatter", "heatmap"]]
    title: Optional[str]
    x_axis: Optional[str]
    y_axis: Optional[str]
    z_axis: Optional[str] = Field(
        default=None,
        description="Para heatmaps: métrica numérica que determina el color de la celda"
    )
    color_column: Optional[str] = Field(
        default=None,
        description="Columna para agrupar series (líneas/barras) o métrica de color (heatmap)"
    )
    orientation: Literal["v", "h"] = "v"
    figure_spec: Optional[Dict[str, Any]]
    reasoning: str
    error_message: Optional[str] = None
    suitable_for_visualization: bool = True

    @field_validator("orientation", mode="before")
    @classmethod
    def default_orientation(cls, v):
        """Si el LLM envía None o un valor vacío, se fuerza a 'v' por defecto."""
        if v is None or v not in ["v", "h"]:
            return "v"
        return v

    @field_validator("figure_spec", mode="before")
    @classmethod
    def ensure_figure_spec_orientation(cls, v, values):
        """Asegura que el diccionario figure_spec también tenga la orientación correcta."""
        if isinstance(v, dict):
            orientation = values.data.get("orientation", "v")
            v["orientation"] = orientation
        return v


class HarnessContext(BaseModel):
    """Paquete de contexto que la capa de harness construye antes del planner."""
    question: str
    intent: str = ""
    task_type: str = "single_view"
    granularity_hint: str = "unknown"
    temporal_scope: str = "none"
    business_rules: List[str] = Field(default_factory=list)
    allowed_views: List[str] = Field(
        default_factory=list,
        description="Todas las vistas candidatas con prefijo semantic."
    )
    preferred_view: Optional[str] = Field(
        default=None,
        description="Vista principal sugerida por RAG + heurísticas"
    )
    semantic_context: str = Field(
        default="",
        description="Texto descriptivo de candidatas para inyectar en prompts"
    )
    ambiguity_notes: List[str] = Field(default_factory=list)
    business_memory: Dict[str, str] = Field(
        default_factory=dict,
        description="Reglas estables parseadas desde AGENTS.md"
    )


class FactItem(BaseModel):
    """Hecho o suposición razonada con trazabilidad."""
    content: str
    source: Literal[
        "plan", "sql_result", "analyst", "viz",
        "forecast", "research", "assumption", "user"
    ]
    verified: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TaskLedger(BaseModel):
    """Libro de registro de la tarea global: hechos, plan y suposiciones."""
    original_question: str = ""
    intent: str = ""
    question_type: Optional[str] = None
    plan: Optional[Any] = None
    facts_verified: List[FactItem] = Field(default_factory=list)
    facts_to_lookup: List[str] = Field(default_factory=list)
    facts_to_derive: List[str] = Field(default_factory=list)
    assumptions: List[FactItem] = Field(default_factory=list)
    educated_guess: Optional[str] = None


class ProgressLedger(BaseModel):
    """Libro de registro del progreso: stall detection, routing e instrucciones."""
    completed_steps: List[str] = Field(default_factory=list)
    stall_count: int = 0
    last_state_hash: Optional[str] = None
    last_progress_reasoning: str = ""
    task_complete: bool = False
    next_agent: Optional[str] = None
    instruction: Optional[str] = None
    unproductive_loop_detected: bool = False
