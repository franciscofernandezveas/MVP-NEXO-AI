# backend/services/sales_service.py
# Refactorizado: sin dependencias de auth para MVP

from decimal import Decimal
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from utils.cache import cacheable

from repositories.sales_repository import (
    get_sales_review_today,
    get_sucursal_review,
    get_ventas_historico_general,
    get_ventas_historico_local,
    get_ventas_semana_actual_vs_anterior,
    get_cantidad_historica_ventas_producto,
    get_productos_vendidos_por_sucursal,
    get_sales_producto_daily,
    get_tipos_producto,
    get_subcategorias_nuevas,
    get_categorias_nuevas,
    get_productos,
    get_sucursales,
    # Nuevas funciones agregadas
    get_dashboard_canjes_resumen,
    get_detalle_canjes_completo,
    get_dashboard_cortesias_resumen,
    get_detalle_cortesias_completo,
    get_rendimiento_categorias_resumen_con_fecha,
    # Funciones para demanda horaria
    get_demanda_horaria,
    get_promedio_demanda_horaria,
    # Detalle por categoría
    get_kpi_categorias_productos_sede,
)


# -----------------------------
# UTILIDADES
# -----------------------------

def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError, TypeError):
        return 0.0


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _normalize_status(text: str) -> str:
    if not text:
        return "stable"

    t = text.lower()

    if "baja" in t or "📉" in t:
        return "down"
    if "aumento" in t or "📈" in t:
        return "up"

    return "stable"


def _empty_kpi():
    return {
        "totalVentas": 0.0,
        "ventasAyer": 0.0,
        "totalTransacciones": 0,
        "ventasMes": 0.0,
        "transaccionesMes": 0,
        "ticketPromedio": 0.0,
        "variacionDiaria": 0.0,
        "estadoVentas": "stable"
    }


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# -----------------------------
# 🔥 SERVICIOS CACHEADOS (ALTO IMPACTO)
# -----------------------------

@cacheable(lambda: "sales_today")
async def get_sales_review_today_service() -> List[Dict[str, Any]]:
    """Servicio principal de revisión de ventas diarias"""
    try:
        data = get_sales_review_today()
        print("🔍 RAW DATA:", data)

        if not data:
            return [_empty_kpi()]

        row = data[0]

        total_ventas = _to_float(row.get("ventas_hoy"))
        ventas_ayer = _to_float(row.get("ventas_ayer"))
        transacciones = _to_int(row.get("transacciones_hoy"))

        ventas_mes = _to_float(row.get("ventas_total_mes"))
        transacciones_mes = _to_int(row.get("transacciones_mes"))

        variacion = _to_float(row.get("variacion_diaria_pct"))
        estado = _normalize_status(row.get("estado_ventas_diarias"))

        ticket = round(total_ventas / transacciones, 2) if transacciones else 0

        result = [{
            "totalVentas": total_ventas,
            "ventasAyer": ventas_ayer,
            "totalTransacciones": transacciones,
            "ventasMes": ventas_mes,
            "transaccionesMes": transacciones_mes,
            "ticketPromedio": ticket,
            "variacionDiaria": variacion,
            "estadoVentas": estado
        }]

        print("✅ RESULT:", result)
        return result

    except Exception as e:
        print("❌ ERROR:", e)
        return [_empty_kpi()]


@cacheable(lambda: "sucursal_review")
async def get_sucursal_review_service():
    """Resumen por sucursal"""
    data = get_sucursal_review() or []

    return [
        {
            "sucursal": row.get("sucursal") or row.get("nombre_sede"),
            "ventas": _to_float(row.get("ventas") or row.get("venta_total")),
            "transacciones": _to_int(row.get("transacciones") or row.get("total_transacciones")),
            "ticket_promedio": _to_float(row.get("ticket_promedio")),
        }
        for row in data
    ]


@cacheable(
    lambda nombre_sede=None, fecha_inicio=None, fecha_fin=None:
    f"ventas_historico_local:"
    f"{(nombre_sede or 'all').lower()}:"
    f"{fecha_inicio or 'min'}:"
    f"{fecha_fin or 'max'}"
)
async def get_ventas_historico_local_service(
    nombre_sede=None,
    fecha_inicio=None,
    fecha_fin=None
):
    data = get_ventas_historico_local(
        nombre_sede,
        fecha_inicio,
        fecha_fin
    ) or []

    return [
        {
            "nombre_sede": row.get("nombre_sede"),
            "fecha_completa": str(row.get("fecha_completa")),
            "total_transacciones": _to_int(row.get("total_transacciones")),
            "venta_total": _to_float(row.get("venta_total")),
            "ticket_promedio": _to_float(row.get("ticket_promedio")),
        }
        for row in data
    ]


@cacheable(lambda: "ventas_historico_general")
async def get_ventas_historico_general_service():
    """Histórico general de ventas"""
    return [
        {
            **row,
            "ventas_dia": _to_float(row.get("ventas_dia")),
            "transacciones_dia": _to_int(row.get("transacciones_dia")),
        }
        for row in (get_ventas_historico_general() or [])
    ]


@cacheable(lambda fecha_inicio, fecha_fin, sucursal=None, producto=None,
          categoria=None, categoria_nueva=None, subcategoria_nueva=None:
          f"sales_producto_daily:{fecha_inicio}:{fecha_fin}:{sucursal or 'all'}:"
          f"{producto or 'all'}:{categoria or 'all'}:{categoria_nueva or 'all'}:"
          f"{subcategoria_nueva or 'all'}")
async def get_sales_producto_daily_service(
    fecha_inicio: str,
    fecha_fin: str,
    sucursal: str = None,
    producto: str = None,
    categoria: str = None,
    categoria_nueva: str = None,
    subcategoria_nueva: str = None
):
    data = get_sales_producto_daily(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        sucursal=sucursal,
        producto=producto,
        categoria=categoria,
        categoria_nueva=categoria_nueva,
        subcategoria_nueva=subcategoria_nueva
    ) or []

    return [
        {
            "fecha": str(row.get("fecha")) if row.get("fecha") else None,
            "sucursal": row.get("sucursal"),
            "producto": row.get("producto"),
            "categoria_original": row.get("categoria_original"),
            "categoria": row.get("categoria"),
            "subcategoria": row.get("subcategoria"),
            "tipo_producto": row.get("tipo_producto"),
            "ventas": _to_float(row.get("ventas")),
            "unidades": _to_float(row.get("unidades"))
        }
        for row in data
    ]


# -----------------------------
# SERVICIOS DIMENSIONALES
# -----------------------------
@cacheable(lambda: "dim_sucursales")
async def get_sucursales_service():
    data = get_sucursales() or []
    return [row.get("sucursal") for row in data if row.get("sucursal")]


@cacheable(lambda: "dim_categorias_nuevas")
async def get_categorias_service():
    """Servicio actualizado para categorías nuevas"""
    data = get_categorias_nuevas() or []
    return [row.get("categoria_nueva") for row in data if row.get("categoria_nueva")]


@cacheable(lambda categoria_nueva=None: f"dim_subcategorias:{categoria_nueva or 'all'}")
async def get_subcategorias_service(categoria_nueva: str = None):
    """Nuevo servicio para subcategorías"""
    data = get_subcategorias_nuevas(categoria_nueva) or []
    return [row.get("subcategoria_nueva") for row in data if row.get("subcategoria_nueva")]


@cacheable(lambda: "dim_tipos_producto")
async def get_tipos_producto_service():
    """Nuevo servicio para tipos de producto"""
    data = get_tipos_producto() or []
    return [row.get("tipo_producto") for row in data if row.get("tipo_producto")]


@cacheable(lambda: "dim_productos")
async def get_productos_service():
    data = get_productos() or []
    return [row.get("producto") for row in data if row.get("producto")]


# -----------------------------
# SERVICIOS DE CANJES Y CORTESÍAS
# -----------------------------
@cacheable(lambda: "dashboard_canjes_resumen")
async def get_dashboard_canjes_resumen_service():
    """Resumen de canjes de fidelización para dashboard principal"""
    data = get_dashboard_canjes_resumen() or []

    return [
        {
            "mes_formato": row.get("mes_formato"),
            "sucursal": row.get("sucursal"),
            "unidades_totales": _to_int(row.get("unidades_totales")),
            "valor_total_canjes": _to_float(row.get("valor_total_canjes"))
        }
        for row in data
    ]


@cacheable(lambda fecha_inicio=None, fecha_fin=None, sucursal=None: f"detalle_canjes:{fecha_inicio}:{fecha_fin}:{sucursal}")
async def get_detalle_canjes_completo_service(fecha_inicio=None, fecha_fin=None, sucursal=None):
    """Detalle completo de canjes para drill-down"""
    data = get_detalle_canjes_completo(fecha_inicio, fecha_fin, sucursal) or []

    return [
        {
            "fecha": str(row.get("fecha")) if row.get("fecha") else None,
            "sucursal": row.get("sucursal"),
            "categoria": row.get("categoria"),
            "producto": row.get("producto"),
            "unidades_fidelizacion": _to_int(row.get("unidades_fidelizacion")),
            "valor_fidelizacion": _to_float(row.get("valor_fidelizacion"))
        }
        for row in data
    ]


@cacheable(lambda: "dashboard_cortesias_resumen")
async def get_dashboard_cortesias_resumen_service():
    data = get_dashboard_cortesias_resumen() or []

    return [
        {
            "mes_formato": row.get("mes_formato"),
            "sucursal": row.get("sucursal"),
            "productos_regalados": _to_int(row.get("productos_regalados")),
            "unidades_totales": _to_int(row.get("unidades_totales")),
            "valor_impacto_total": _to_float(row.get("valor_impacto_total"))
        }
        for row in data
    ]


@cacheable(lambda fecha_inicio=None, fecha_fin=None, sucursal=None: f"detalle_cortesias:{fecha_inicio}:{fecha_fin}:{sucursal}")
async def get_detalle_cortesias_completo_service(fecha_inicio=None, fecha_fin=None, sucursal=None):
    data = get_detalle_cortesias_completo(fecha_inicio, fecha_fin, sucursal) or []

    return [
        {
            "fecha": str(row.get("fecha")) if row.get("fecha") else None,
            "sucursal": row.get("sucursal"),
            "categoria": row.get("categoria"),
            "producto": row.get("producto"),
            "unidades_cortesia": _to_int(row.get("unidades_cortesia")),
            "valor_cortesia": _to_float(row.get("valor_cortesia"))
        }
        for row in data
    ]


# -----------------------------
# SERVICIOS DE DEMANDA POR HORA
# -----------------------------
@cacheable(lambda nombre_sede=None, fecha=None: f"demanda_horaria:{nombre_sede or 'all'}:{fecha or 'latest'}")
async def get_demanda_horaria_service(nombre_sede=None, fecha=None):
    """
    Servicio para obtener demanda horaria específica
    """
    data = get_demanda_horaria(nombre_sede, fecha) or []

    return [
        {
            "nombre_sede": row.get("nombre_sede"),
            "nombre_dia_semana": row.get("nombre_dia_semana"),
            "fecha": str(row.get("fecha")) if row.get("fecha") else None,
            "hora": _to_int(row.get("hora")),
            "transacciones": _to_int(row.get("transacciones")),
            "ingreso": _to_float(row.get("ingreso")),
            "ticket_promedio": _to_float(row.get("ticket_promedio")),
        }
        for row in data
    ]


@cacheable(lambda nombre_sede=None, mes=None, anio=None: f"promedio_demanda_horaria:{nombre_sede or 'all'}:{mes or 'all'}:{anio or 'all'}")
async def get_promedio_demanda_horaria_service(nombre_sede=None, mes=None, anio=None):
    """
    Servicio para obtener promedio de demanda horaria
    """
    data = get_promedio_demanda_horaria(nombre_sede, mes, anio) or []

    total_transacciones = sum(_to_float(row.get("promedio_transacciones")) for row in data)

    return [
        {
            "nombre_sede": row.get("nombre_sede"),
            "hora": _to_int(row.get("hora")),
            "promedio_transacciones": _to_float(row.get("promedio_transacciones")),
            "promedio_ingreso": _to_float(row.get("promedio_ingreso")),
            "dias_registrados": _to_int(row.get("dias_registrados")),
            "porcentaje_demanda": (_to_float(row.get("promedio_transacciones")) / total_transacciones * 100) if total_transacciones > 0 else 0
        }
        for row in data
    ]


# -----------------------------
# DEMANDA SEMANAL
# -----------------------------
@cacheable(lambda: "demanda_semanal")
async def get_demanda_semanal_service():
    """
    Servicio para obtener histórico semanal de ventas.
    Agrupa los datos diarios en semanas ISO.
    """
    data = get_ventas_historico_general() or []

    semanas = {}
    for row in data:
        fecha = _parse_date(row.get("fecha"))
        if not fecha:
            continue

        year, week, _ = fecha.isocalendar()
        key = f"{year}-W{week:02d}"

        if key not in semanas:
            semanas[key] = {
                "anio": year,
                "semana": week,
                "fecha_inicio": fecha,
                "fecha_fin": fecha,
                "ventas_total": 0.0,
                "transacciones_total": 0,
                "dias": 0
            }

        semana = semanas[key]
        semana["ventas_total"] += _to_float(row.get("ventas_dia"))
        semana["transacciones_total"] += _to_int(row.get("transacciones_dia"))
        semana["dias"] += 1

        if fecha < semana["fecha_inicio"]:
            semana["fecha_inicio"] = fecha
        if fecha > semana["fecha_fin"]:
            semana["fecha_fin"] = fecha

    resultado = []
    for key in sorted(semanas.keys()):
        semana = semanas[key]
        ticket = round(semana["ventas_total"] / semana["transacciones_total"], 2) if semana["transacciones_total"] > 0 else 0

        resultado.append({
            "anio": semana["anio"],
            "semana": semana["semana"],
            "periodo": key,
            "fecha_inicio": str(semana["fecha_inicio"]),
            "fecha_fin": str(semana["fecha_fin"]),
            "ventas_total": round(semana["ventas_total"], 2),
            "transacciones_total": semana["transacciones_total"],
            "ticket_promedio": ticket,
            "dias_con_ventas": semana["dias"]
        })

    return resultado


# -----------------------------
# DETALLE POR CATEGORÍA
# -----------------------------
@cacheable(lambda categoria=None, sucursal=None, fecha_inicio=None, fecha_fin=None:
           f"detalle_categoria:{categoria or 'all'}:{sucursal or 'all'}:{fecha_inicio or 'min'}:{fecha_fin or 'max'}")
async def get_detalle_categoria_service(
    categoria: str = None,
    sucursal: str = None,
    fecha_inicio: str = None,
    fecha_fin: str = None
):
    """
    Servicio para detalle de productos por categoría, sucursal y rango de fechas.
    """
    data = get_kpi_categorias_productos_sede(
        categoria=categoria,
        sucursal=sucursal,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    ) or []

    return [
        {
            "fecha_venta": str(row.get("fecha_venta")) if row.get("fecha_venta") else None,
            "sucursal": row.get("sucursal"),
            "categoria": row.get("categoria"),
            "producto": row.get("producto"),
            "unidades_totales": _to_int(row.get("unidades_totales")),
            "ventas_totales": _to_float(row.get("ventas_totales")),
            "ticket_promedio": round(_to_float(row.get("ventas_totales")) / _to_int(row.get("unidades_totales")), 2)
                              if _to_int(row.get("unidades_totales")) > 0 else 0
        }
        for row in data
    ]


# -----------------------------
# OTROS SERVICIOS
# -----------------------------
@cacheable(lambda: "ventas_semana_actual_vs_anterior")
async def get_ventas_semana_actual_vs_anterior_service():
    return get_ventas_semana_actual_vs_anterior() or []


@cacheable(lambda: "cantidad_historica_ventas_producto")
async def get_cantidad_historica_ventas_producto_service():
    return get_cantidad_historica_ventas_producto() or []


@cacheable(lambda: "productos_vendidos_por_sucursal")
async def get_productos_vendidos_por_sucursal_service():
    return get_productos_vendidos_por_sucursal() or []


# -----------------------------
# RENDIMIENTO POR CATEGORÍA (BI)
# -----------------------------
@cacheable(lambda fecha_inicio=None, fecha_fin=None: f"rendimiento_categorias_resumen:{fecha_inicio}:{fecha_fin}")
async def get_rendimiento_categorias_resumen_con_fecha_service(
    fecha_inicio=None,
    fecha_fin=None
):
    """
    Service para distribución de ventas por categoría.
    """
    print("FILTROS:", fecha_inicio, fecha_fin)

    data = get_rendimiento_categorias_resumen_con_fecha(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    ) or []

    return [
        {
            "categoria": row.get("categoria"),
            "ventas_totales_categoria": _to_float(row.get("ventas_totales_categoria")),
            "unidades_totales": _to_int(row.get("unidades_totales")),
        }
        for row in data
    ]
