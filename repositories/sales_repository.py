# backend/repositories/sales_repository.py
# Refactorizado: sin dependencias de auth para MVP

from database.connection import get_connection, release_connection
import psycopg2.extras


# -----------------------------
# CORE DB FUNCTIONS
# -----------------------------
def fetch_all(query: str, params=None):
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query, params) if params else cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if conn:
            release_connection(conn)


def fetch_one(query: str, params=None):
    """Devuelve una sola fila (para KPIs agregados)."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query, params) if params else cursor.execute(query)
        result = cursor.fetchone()
        return dict(result) if result else None
    finally:
        if conn:
            release_connection(conn)


# -----------------------------
# DEMANDA HORARIA
# -----------------------------
def get_demanda_horaria(nombre_sede=None, fecha=None):
    query = """
    SELECT 
        nombre_sede,
        nombre_dia_semana,
        fecha,
        hora,
        transacciones,
        ingreso,
        ticket_promedio
    FROM semantic.mart_operacion_hora
    WHERE 1=1
    """
    params = []

    if nombre_sede:
        query += " AND nombre_sede = %s"
        params.append(nombre_sede)

    if fecha:
        query += " AND fecha = %s::date"
        params.append(fecha)

    query += " ORDER BY hora ASC"

    return fetch_all(query, params)


def get_promedio_demanda_horaria(nombre_sede=None, mes=None, anio=None):
    query = """
    SELECT 
        nombre_sede,
        hora,
        AVG(transacciones) as promedio_transacciones,
        AVG(ingreso) as promedio_ingreso,
        COUNT(*) as dias_registrados
    FROM semantic.mart_operacion_hora
    WHERE 1=1
    """
    params = []

    if nombre_sede:
        query += " AND nombre_sede = %s"
        params.append(nombre_sede)

    if mes:
        query += " AND EXTRACT(MONTH FROM fecha) = %s"
        params.append(mes)

    if anio:
        query += " AND EXTRACT(YEAR FROM fecha) = %s"
        params.append(anio)

    query += """
    GROUP BY nombre_sede, hora
    ORDER BY hora ASC
    """

    return fetch_all(query, params)


# -----------------------------
# DEMANDA SEMANAL
# -----------------------------
def get_demanda_semanal():
    query = """
    SELECT *
    FROM semantic.mart_operacion_hora
    ORDER BY fecha DESC, hora ASC
    LIMIT 1000
    """
    return fetch_all(query)


# -----------------------------
# SALES KPIs
# -----------------------------
def get_sales_review_today():
    query = "SELECT * FROM semantic.sales_review_day;"
    data = fetch_all(query)

    print("🔍 RAW VIEW DATA (sales_review_day):", data)

    if not data:
        print("⚠️ VISTA VACÍA - USANDO FALLBACK CONTROLADO")
        fallback_query = """
        SELECT 
            COALESCE(SUM(ft.monto_total), 0) AS ventas_hoy,
            0 AS ventas_ayer,
            COALESCE(COUNT(ft.id_transaccion), 0) AS transacciones_hoy,
            COALESCE(SUM(ft.monto_total), 0) AS ventas_total_mes,
            COALESCE(COUNT(ft.id_transaccion), 0) AS transacciones_mes,
            0 AS variacion_diaria_pct,
            'Sin datos' AS estado_ventas_diarias
        FROM dw.fact_transacciones ft
        """
        data = fetch_all(fallback_query)
        print("🔍 FALLBACK DATA:", data)

    return data


def get_ventas_mes_actual():
    query = """
        SELECT 
            SUM(ventas_dia) as ventas_mes
        FROM semantic.sales_review_day_history
        WHERE DATE_TRUNC('month', fecha) = DATE_TRUNC('month', CURRENT_DATE)
    """
    return fetch_all(query)


# -----------------------------
# OTRAS CONSULTAS
# -----------------------------
def get_sucursal_review():
    return fetch_all("SELECT * FROM semantic.sales_review_locales_latest;")


def get_ventas_historico_general():
    return fetch_all("SELECT * FROM semantic.sales_review_day_history;")


def get_ventas_historico_local(nombre_sede=None, fecha_inicio=None, fecha_fin=None):
    query = """
        SELECT
            nombre_sede,
            fecha_completa,
            total_transacciones,
            venta_total,
            ticket_promedio
        FROM semantic.sales_review_locales
        WHERE 1=1
    """
    params = []

    if nombre_sede:
        query += " AND nombre_sede = %s"
        params.append(nombre_sede)

    if fecha_inicio and fecha_fin:
        query += " AND fecha_completa BETWEEN %s::date AND %s::date"
        params.extend([fecha_inicio, fecha_fin])
    elif fecha_inicio:
        query += " AND fecha_completa >= %s::date"
        params.append(fecha_inicio)
    elif fecha_fin:
        query += " AND fecha_completa <= %s::date"
        params.append(fecha_fin)

    query += " ORDER BY fecha_completa ASC, nombre_sede"

    return fetch_all(query, params)


def get_ventas_semana_actual_vs_anterior():
    return fetch_all("SELECT * FROM semantic.sales_week;")


def get_cantidad_historica_ventas_producto():
    return fetch_all("SELECT * FROM semantic.sales_product_general;")


def get_productos_vendidos_por_sucursal():
    return fetch_all("SELECT * FROM dw.product_performance;")


def get_productos_vendidos_total_mensual():
    return fetch_all("SELECT * FROM dw.product_performance_overview;")


# -----------------------------
# VENTAS POR PRODUCTO (DESDE VISTA SEMÁNTICA)
# -----------------------------
def get_sales_producto_daily(
    fecha_inicio: str,
    fecha_fin: str,
    sucursal: str = None,
    producto: str = None,
    categoria: str = None,
    categoria_nueva: str = None,
    subcategoria_nueva: str = None
):
    query = """
    SELECT
        fecha,
        sucursal,
        producto,
        categoria_original,
        categoria,
        subcategoria,
        tipo_producto,
        ventas,
        unidades
    FROM semantic.sales_producto_daily
    WHERE fecha BETWEEN %s AND %s
    """

    params = [fecha_inicio, fecha_fin]

    if sucursal:
        query += " AND sucursal = %s"
        params.append(sucursal)

    if producto:
        query += " AND producto ILIKE %s"
        params.append(f"%{producto}%")

    if categoria_nueva:
        query += " AND categoria = %s"
        params.append(categoria_nueva)
    elif categoria:
        query += " AND categoria_original = %s"
        params.append(categoria)

    if subcategoria_nueva:
        query += " AND subcategoria = %s"
        params.append(subcategoria_nueva)

    query += " ORDER BY fecha DESC, ventas DESC"

    return fetch_all(query, params)


# -----------------------------
# DIMENSIONES DESDE VISTA SEMÁNTICA
# -----------------------------
def get_sucursales():
    query = """
    SELECT DISTINCT sucursal
    FROM semantic.sales_producto_daily
    WHERE sucursal IS NOT NULL
    ORDER BY sucursal
    """
    return fetch_all(query, [])


def get_categorias_nuevas():
    query = """
    SELECT DISTINCT categoria AS categoria_nueva
    FROM semantic.sales_producto_daily
    WHERE categoria IS NOT NULL
    ORDER BY categoria
    """
    return fetch_all(query, [])


def get_subcategorias_nuevas(categoria_nueva: str = None):
    query = """
    SELECT DISTINCT subcategoria AS subcategoria_nueva
    FROM semantic.sales_producto_daily
    WHERE subcategoria IS NOT NULL
    """
    params = []

    if categoria_nueva:
        query += " AND categoria = %s"
        params.append(categoria_nueva)

    query += " ORDER BY subcategoria"

    return fetch_all(query, params)


def get_tipos_producto():
    query = """
    SELECT DISTINCT tipo_producto
    FROM semantic.sales_producto_daily
    WHERE tipo_producto IS NOT NULL
    ORDER BY tipo_producto
    """
    return fetch_all(query, [])


def get_productos():
    query = """
    SELECT DISTINCT producto
    FROM semantic.sales_producto_daily
    WHERE producto IS NOT NULL
    ORDER BY producto
    """
    return fetch_all(query, [])


# -----------------------------
# DASHBOARD CANJES & CORTESÍAS
# -----------------------------
def get_dashboard_canjes_resumen():
    query = """
    SELECT 
        TO_CHAR(mes, 'Mon YYYY') as mes_formato,
        sucursal,
        unidades_totales,
        valor_total_canjes
    FROM semantic.dashboard_canjes_resumen
    WHERE mes = DATE_TRUNC('month', CURRENT_DATE)
    ORDER BY unidades_totales DESC;
    """
    return fetch_all(query)


def get_detalle_canjes_completo(fecha_inicio=None, fecha_fin=None, sucursal=None):
    base_query = """
    SELECT * FROM semantic.kpi_fidelizacion_detalle
    WHERE 1=1
    """
    params = []

    if fecha_inicio and fecha_fin:
        base_query += " AND fecha BETWEEN %s AND %s"
        params.extend([fecha_inicio, fecha_fin])

    if sucursal:
        base_query += " AND sucursal = %s"
        params.append(sucursal)

    base_query += " ORDER BY fecha DESC, unidades_fidelizacion DESC"

    return fetch_all(base_query, params)


def get_dashboard_cortesias_resumen():
    query = """
    SELECT 
        TO_CHAR(mes, 'Mon YYYY') as mes_formato,
        sucursal,
        productos_regalados,
        unidades_totales,
        valor_impacto_total
    FROM semantic.dashboard_cortesias_resumen
    WHERE mes = DATE_TRUNC('month', CURRENT_DATE)
    ORDER BY unidades_totales DESC;
    """
    return fetch_all(query)


def get_detalle_cortesias_completo(fecha_inicio=None, fecha_fin=None, sucursal=None):
    base_query = """
    SELECT * FROM semantic.kpi_cortesia_detalle
    WHERE 1=1
    """
    params = []

    if fecha_inicio and fecha_fin:
        base_query += " AND fecha BETWEEN %s AND %s"
        params.extend([fecha_inicio, fecha_fin])

    if sucursal:
        base_query += " AND sucursal = %s"
        params.append(sucursal)

    base_query += " ORDER BY fecha DESC, unidades_cortesia DESC"

    return fetch_all(base_query, params)


# -----------------------------
# CATEGORÍAS POR PRODUCTO & KPIS RELACIONADOS
# -----------------------------
def get_fechas_disponibles():
    query = """
    SELECT DISTINCT fecha
    FROM semantic.kpi_categorias_diario
    ORDER BY fecha DESC;
    """
    return fetch_all(query)


def get_categorias_kpi():
    query = """
    SELECT DISTINCT categoria
    FROM semantic.kpi_categorias_diario
    WHERE categoria IS NOT NULL
    ORDER BY categoria;
    """
    return fetch_all(query)


def get_sucursales_kpi():
    query = """
    SELECT DISTINCT sucursal
    FROM semantic.kpi_categorias_productos_sede
    WHERE sucursal IS NOT NULL
    ORDER BY sucursal;
    """
    return fetch_all(query)


def get_productos_por_categoria(categoria=None):
    query = """
    SELECT DISTINCT producto
    FROM semantic.kpi_categorias_productos_sede
    WHERE 1=1
    """
    params = []

    if categoria:
        query += " AND categoria = %s"
        params.append(categoria)

    query += " ORDER BY producto;"

    return fetch_all(query, params)


def get_kpi_categorias_diario(fecha_inicio=None, fecha_fin=None, categoria=None):
    query = """
    SELECT *
    FROM semantic.kpi_categorias_diario
    WHERE 1=1
    """
    params = []

    if fecha_inicio and fecha_fin:
        query += " AND fecha BETWEEN %s AND %s"
        params.extend([fecha_inicio, fecha_fin])

    if categoria:
        query += " AND categoria = %s"
        params.append(categoria)

    query += " ORDER BY fecha DESC, ventas DESC"

    return fetch_all(query, params)


def get_kpi_categorias_productos_sede(categoria=None, sucursal=None, fecha_inicio=None, fecha_fin=None):
    query = """
    SELECT *
    FROM semantic.kpi_categorias_productos_sede
    WHERE 1=1
    """
    params = []

    if categoria:
        query += " AND categoria = %s"
        params.append(categoria)

    if sucursal:
        query += " AND sucursal = %s"
        params.append(sucursal)

    if fecha_inicio and fecha_fin:
        query += " AND fecha_venta BETWEEN %s AND %s"
        params.extend([fecha_inicio, fecha_fin])
    elif fecha_inicio:
        query += " AND fecha_venta >= %s"
        params.append(fecha_inicio)
    elif fecha_fin:
        query += " AND fecha_venta <= %s"
        params.append(fecha_fin)

    query += " ORDER BY ventas_totales DESC"

    return fetch_all(query, params)


def get_rendimiento_categorias_resumen_con_fecha(fecha_inicio=None, fecha_fin=None):
    query = """
    SELECT 
        categoria,
        SUM(subtotal_diario) as ventas_totales_categoria,
        SUM(unidades_vendidas) as unidades_totales
    FROM semantic.dashboard_participacion_categorias
    WHERE 1=1
    """
    params = []

    if not fecha_inicio and not fecha_fin:
        query += """
        AND fecha_completa = (
            SELECT MAX(fecha_completa)
            FROM semantic.dashboard_participacion_categorias
        )
        """
    else:
        if fecha_inicio:
            query += " AND fecha_completa >= %s"
            params.append(fecha_inicio)

        if fecha_fin:
            query += " AND fecha_completa <= %s"
            params.append(fecha_fin)

    query += """
    GROUP BY categoria
    ORDER BY ventas_totales_categoria DESC
    """

    return fetch_all(query, params)
