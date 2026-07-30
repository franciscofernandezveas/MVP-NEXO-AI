# backend/routes/sales_routes.py

from fastapi import APIRouter
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
# backend/routes/sales_routes.py

from typing import List, Dict, Any, Literal
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.sales_service import generate_executive_report_service

from services.sales_service import (
    get_sales_review_today_service,
    get_sucursal_review_service,
    get_ventas_historico_general_service,
    get_ventas_historico_local_service,
    get_ventas_semana_actual_vs_anterior_service,
    get_sales_producto_daily_service,
    get_dashboard_canjes_resumen_service,
    get_detalle_canjes_completo_service,
    get_dashboard_cortesias_resumen_service,
    get_detalle_cortesias_completo_service,
    get_sucursales_service,
    get_categorias_service,
    get_productos_service,
    get_subcategorias_service,
    get_tipos_producto_service,
    get_rendimiento_categorias_resumen_con_fecha_service,
    get_demanda_horaria_service,
    get_promedio_demanda_horaria_service,
    get_demanda_semanal_service,
    get_detalle_categoria_service,
)


# -----------------------------
# DTOs
# -----------------------------

class SalesReviewTodayDTO(BaseModel):
    totalVentas: float
    ventasAyer: float
    totalTransacciones: int
    ventasMes: float
    transaccionesMes: int
    ticketPromedio: float
    variacionDiaria: Optional[float] = None
    estadoVentas: str


class DemandaHorariaResponseDTO(BaseModel):
    nombre_sede: str
    nombre_dia_semana: str
    fecha: str
    hora: int
    transacciones: int
    ingreso: float
    ticket_promedio: float


class PromedioDemandaHorariaResponseDTO(BaseModel):
    nombre_sede: str
    hora: int
    promedio_transacciones: float
    promedio_ingreso: float
    dias_registrados: int
    porcentaje_demanda: float


# -----------------------------
# ROUTER
# -----------------------------

router = APIRouter(prefix="/sales", tags=["sales"])


# ======================================================================================
# RESUMEN DIARIO
# ======================================================================================
@router.get("/sales-review-today", response_model=List[SalesReviewTodayDTO])
async def sales_review_today():
    """
    Resumen de ventas del día actual.
    """
    return await get_sales_review_today_service()


# ======================================================================================
# VENTAS POR LOCAL
# ======================================================================================
@router.get("/sucursal-review", response_model=List[Dict[str, Any]])
async def sucursal_review():
    return await get_sucursal_review_service()


# ======================================================================================
# HISTÓRICOS
# ======================================================================================
@router.get("/ventas-historico-general", response_model=List[Dict[str, Any]])
async def ventas_historico_general():
    return await get_ventas_historico_general_service()


@router.get("/ventas-historico-local", response_model=List[Dict[str, Any]])
async def ventas_historico_local(
    nombre_sede: Optional[str] = None,
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
):
    return await get_ventas_historico_local_service(
        nombre_sede,
        fecha_inicio,
        fecha_fin
    )


# ======================================================================================
# COMPARACIÓN SEMANAL
# ======================================================================================
@router.get("/ventas-Semana-Actual-vs-anterior", response_model=List[Dict[str, Any]])
async def ventas_semana_actual_vs_anterior():
    return await get_ventas_semana_actual_vs_anterior_service()


# ======================================================================================
# DEMANDA SEMANAL
# ======================================================================================
@router.get("/demanda-semanal", response_model=List[Dict[str, Any]])
async def demanda_semanal():
    return await get_demanda_semanal_service()


# ======================================================================================
# VENTAS POR PRODUCTO (CORE)
# ======================================================================================
@router.get("/ventas-producto", response_model=List[Dict[str, Any]])
async def ventas_producto(
    fecha_inicio: str,
    fecha_fin: str,
    sucursal: Optional[str] = None,
    producto: Optional[str] = None,
    categoria: Optional[str] = None,
    categoria_nueva: Optional[str] = None,
    subcategoria_nueva: Optional[str] = None,
):
    """
    Devuelve ventas diarias por producto desde semantic.sales_producto_daily.
    """
    return await get_sales_producto_daily_service(
        fecha_inicio,
        fecha_fin,
        sucursal,
        producto,
        categoria,
        categoria_nueva,
        subcategoria_nueva
    )


@router.get("/health/sales-producto-daily")
async def health_sales_producto_daily():
    """
    Health-check para verificar que la vista semántica responde.
    """
    try:
        data = await get_sales_producto_daily_service(
            fecha_inicio="2026-01-01",
            fecha_fin="2026-01-01"
        )
        return {
            "status": "ok",
            "registros_muestra": len(data),
            "primer_registro": data[0] if data else None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ======================================================================================
# DIMENSIONES (sin protección para MVP, alineado con main.py)
# ======================================================================================
@router.get("/dimensiones/sedes-promedio")
async def sedes_promedio():
    return await get_sucursales_service()


@router.get("/dim/sucursales")
async def sucursales():
    return await get_sucursales_service()


@router.get("/dim/categorias")
async def categorias():
    return await get_categorias_service()


@router.get("/dim/subcategorias")
async def subcategorias(categoria_nueva: Optional[str] = None):
    return await get_subcategorias_service(categoria_nueva)


@router.get("/dim/tipos-producto")
async def tipos_producto():
    return await get_tipos_producto_service()


@router.get("/dim/productos")
async def productos():
    return await get_productos_service()


# ======================================================================================
# BRANCHES (DATOS AGREGADOS)
# ======================================================================================
@router.get("/branches", response_model=Dict[str, Any])
async def get_branches_data(
    nombre_sede: Optional[str] = None,
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
):
    kpis = await get_sucursal_review_service()
    history = await get_ventas_historico_local_service(
        nombre_sede=nombre_sede,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    )
    return {
        "kpis": kpis,
        "history": history
    }


# ======================================================================================
# DEMANDA HORARIA
# ======================================================================================
@router.get("/demanda-horaria", response_model=List[DemandaHorariaResponseDTO])
async def demanda_horaria(
    nombre_sede: Optional[str] = None,
    fecha: Optional[str] = None,
):
    return await get_demanda_horaria_service(nombre_sede, fecha)


@router.get("/promedio-demanda-horaria", response_model=List[PromedioDemandaHorariaResponseDTO])
async def promedio_demanda_horaria(
    nombre_sede: Optional[str] = None,
    mes: Optional[int] = None,
    anio: Optional[int] = None,
):
    return await get_promedio_demanda_horaria_service(nombre_sede, mes, anio)


# ======================================================================================
# CANJES Y CORTESÍAS
# ======================================================================================
@router.get("/dashboard-canjes-resumen", response_model=List[Dict[str, Any]])
async def dashboard_canjes_resumen():
    return await get_dashboard_canjes_resumen_service()


@router.get("/detalle-canjes-completo", response_model=List[Dict[str, Any]])
async def detalle_canjes_completo(
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
    sucursal: Optional[str] = None,
):
    return await get_detalle_canjes_completo_service(fecha_inicio, fecha_fin, sucursal)


@router.get("/dashboard-cortesias-resumen", response_model=List[Dict[str, Any]])
async def dashboard_cortesias_resumen():
    return await get_dashboard_cortesias_resumen_service()


@router.get("/detalle-cortesias-completo", response_model=List[Dict[str, Any]])
async def detalle_cortesias_completo(
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
    sucursal: Optional[str] = None,
):
    return await get_detalle_cortesias_completo_service(
        fecha_inicio,
        fecha_fin,
        sucursal
    )


# ======================================================================================
# PARTICIPACIÓN Y DETALLE DE CATEGORÍAS
# ======================================================================================
@router.get("/categorias/participacion-filtrada", response_model=List[Dict[str, Any]])
async def categorias_participacion_filtrada(
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
):
    return await get_rendimiento_categorias_resumen_con_fecha_service(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    )


@router.get("/categorias/detalle-filtrado", response_model=List[Dict[str, Any]])
async def categorias_detalle_filtrado(
    categoria: Optional[str] = None,
    sucursal: Optional[str] = None,
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
):
    return await get_detalle_categoria_service(
        categoria=categoria,
        sucursal=sucursal,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    )





class ExecutiveReportRequest(BaseModel):
    views: List[str] = Field(..., min_items=1, description="Lista de vistas semánticas a incluir")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtros comunes del reporte")
    format: Literal["html", "csv"] = "html"


@router.post("/executive-report")
async def executive_report(payload: ExecutiveReportRequest):
    """
    Genera un reporte ejecutivo combinando múltiples vistas semánticas.
    Soporta descarga en HTML o CSV.
    """
    try:
        content = await generate_executive_report_service(
            payload.views,
            payload.filters,
            payload.format,
        )

        if payload.format == "csv":
            return Response(
                content=content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=reporte_ejecutivo.csv"
                },
            )

        return Response(
            content=content,
            media_type="text/html",
            headers={
                "Content-Disposition": "attachment; filename=reporte_ejecutivo.html"
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")
