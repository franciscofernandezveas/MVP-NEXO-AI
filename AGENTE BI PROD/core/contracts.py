from typing import Any, List, Optional, Literal, Dict
from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------
# PRIMITIVAS REUTILIZABLES
# ------------------------------------------------------------------

class FilterClause(BaseModel):
    """Filtro estructurado que el SQL agent puede validar y traducir."""
    column: str = Field(..., description="Nombre exacto de la columna en la vista")
    operator: Literal["=", "ILIKE", ">", "<", ">=", "<=", "BETWEEN", "IN", "NOT IN"] = Field(default="=")
    value: Any = Field(..., description="Valor o lista de valores del filtro")
    reasoning: str = Field(default="", description="Por qué se aplicó este filtro")


class DateRange(BaseModel):
    """Ventana temporal resuelta por el planner."""
    start: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    end: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    grain: Literal["day", "week", "month", "quarter", "year"] = Field(default="day")
    relative_label: Optional[str] = Field(default=None, description="last_30_days, this_month, yesterday, etc.")


# ------------------------------------------------------------------
# SQL PAYLOAD
# ------------------------------------------------------------------

class SQLPayload(BaseModel):
    """Instrucción técnica individual."""
    task_id: str = Field(default="1", description="Identificador de la subtarea")
    task: str = Field(..., description="Descripción técnica de la tarea SQL")
    metrics: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    filters: List[FilterClause] = Field(default_factory=list, description="Filtros estructurados validables")
    filters_description: str = Field(default="", description="Filtros en lenguaje natural (para logs)")
    date_range: Optional[DateRange] = Field(default=None)
    time_window: Optional[str] = Field(default=None, description="Texto original de la ventana temporal")
    assumptions: List[str] = Field(default_factory=list)
    execution_strategy: Literal[
        "single_view", "daily", "compare_periods", "historical", "monthly",
        "by_branch", "by_product", "demand_forecast"
    ] = Field(default="single_view")
    candidate_views: List[str] = Field(
        default_factory=list,
        description="Shortlist de vistas semantic. permitidas para esta tarea"
    )
    preferred_view: Optional[str] = Field(
        default=None,
        description="Vista elegida por el planner como principal (debe estar en candidate_views)"
    )

    @model_validator(mode="after")
    def preferred_in_candidates(self):
        if self.preferred_view and self.candidate_views and self.preferred_view not in self.candidate_views:
            raise ValueError("preferred_view debe estar incluido en candidate_views")
        return self


# ------------------------------------------------------------------
# PLANNER
# ------------------------------------------------------------------

class PlannerContract(BaseModel):
    """Contrato del planner con soporte para demand forecast."""
    intent: str
    goal: str
    question_type: Literal[
        "single_kpi", "multi_kpi", "comparison", "trend", "lookup",
        "deep_research", "demand_forecast", "unknown"
    ]
    metrics: List[str]
    dimensions: List[str]
    filters: List[FilterClause] = Field(default_factory=list)
    filters_description: str = Field(default="")
    date_range: Optional[DateRange] = None
    time_window: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    tasks: List[SQLPayload] = Field(...)
    confidence: float = Field(ge=0.0, le=1.0)
    visualization_candidate: bool = Field(default=False)
    chart_type_hint: Optional[str] = Field(default="auto")
    needs_followup: bool = Field(default=False)
    followup_reason: Optional[str] = Field(default=None)
    followup_question: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def consistency_checks(self):
        if self.question_type == "demand_forecast" and not any(t.execution_strategy == "demand_forecast" for t in self.tasks):
            raise ValueError("question_type=demand_forecast requiere al menos una tarea con execution_strategy=demand_forecast")
        if self.needs_followup and self.followup_question is None:
            raise ValueError("Si needs_followup=true, debe existir followup_question")
        return self


# ------------------------------------------------------------------
# RESEARCH
# ------------------------------------------------------------------

class ResearchPlan(BaseModel):
    """Plan de exploración profunda interna."""
    goal: str = Field(..., description="Objetivo del informe de investigación")
    metrics_to_cover: List[str] = Field(default_factory=list)
    dimensions_to_cover: List[str] = Field(default_factory=list)
    time_windows: List[str] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)
    tasks: List[SQLPayload] = Field(...)


class ResearchContract(BaseModel):
    """Resultado del nodo researcher."""
    status: Literal["success", "partial", "error"]
    plan: Optional[ResearchPlan] = None
    task_results: List[SQLContract] = Field(default_factory=list)
    findings: str = ""
    metrics_covered: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    needs_followup: bool = False


# ------------------------------------------------------------------
# SQL AGENT
# ------------------------------------------------------------------

class SQLContract(BaseModel):
    """Contrato estricto de salida del SQL Agent."""
    task_id: str = Field(default="1")
    status: Literal["success", "error", "partial", "needs_clarification", "unrecoverable"]
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

    allowed_views: List[str] = Field(default_factory=list)
    preferred_view: Optional[str] = None
    semantic_context_used: str = Field(default="")
    query_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason_for_view_choice: str = Field(default="")

    @model_validator(mode="after")
    def unrecoverable_is_not_answerable(self):
        if self.status == "unrecoverable":
            self.can_answer = False
            self.query_confidence = 0.0
        return self


# ------------------------------------------------------------------
# SUPERVISOR
# ------------------------------------------------------------------

class SupervisorDecision(BaseModel):
    """Decisión del supervisor con soporte para forecaster y replanning."""
    reasoning: str
    next_agent: Literal[
        "planner", "sql_agent", "analyst", "viz_agent",
        "render_plotly", "viz_approval", "researcher",
        "forecaster", "FINISH"
    ]
    feedback_to_planner: Optional[str] = None  # Solo se usa si next_agent == "planner"
    feedback_to_sql_agent: Optional[str] = None  # Solo se usa si next_agent == "sql_agent"


# ------------------------------------------------------------------
# FORECAST
# ------------------------------------------------------------------

class ForecastRequest(BaseModel):
    producto: str
    sede: str
    n_dias: int = 7
    fecha_inicio: Optional[str] = None


class ForecastResult(BaseModel):
    forecasts: List[Dict[str, Any]] = Field(default_factory=list)
    modelo_version: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    safety_stock: float = 0.0
    error: Optional[str] = None


# ------------------------------------------------------------------
# VISUALIZACIÓN
# ------------------------------------------------------------------

class VizSpecContract(BaseModel):
    status: Literal["success", "error"]
    chart_type: Optional[Literal["bar", "line", "pie", "scatter"]] = None
    title: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_column: Optional[str] = None
    orientation: str = "v"
    figure_spec: Optional[Dict[str, Any]] = None
    reasoning: str = ""
    error_message: Optional[str] = None
    suitable_for_visualization: bool = True


# ------------------------------------------------------------------
# HARNESS CONTEXT
# ------------------------------------------------------------------

class HarnessContext(BaseModel):
    question: str
    intent: str = ""
    task_type: str = "single_view"
    granularity_hint: str = "unknown"
    temporal_scope: str = "none"
    business_rules: List[str] = Field(default_factory=list)
    allowed_views: List[str] = Field(default_factory=list)
    preferred_view: Optional[str] = None
    semantic_context: str = Field(default="")
    ambiguity_notes: List[str] = Field(default_factory=list)
    business_memory: Dict[str, str] = Field(default_factory=dict)
