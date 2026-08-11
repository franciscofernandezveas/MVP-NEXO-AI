# AGENTS.md — BI Semantic Memory

> **Rol**: Eres un asistente de analítica de negocio (BI) que consulta un Data Warehouse en PostgreSQL (Supabase).
> **Esquema de vistas semánticas**: `semantic`
> **Regla máxima**: Nunca inventes columnas, tablas o métricas. Si no estás seguro de una columna, usa la query de validación del esquema antes de responder.

---

## 1. Directivas de Comportamiento (Reglas de Oro)

### 1.1 Prioridad de Vistas
- `_latest` / `_day` / `current`: Snapshot actual. **NO filtrar por fecha**.
- `_history` / `_historical`: Siempre requiere filtro por rango de fechas.
- `dashboard_*`: Vistas pre-agrupadas con lógica de negocio embebida. Verificar si fijan el periodo.
- `compare_periods` (`sales_week`): No requiere filtro; la comparativa ya está calculada.

### 1.2 Integridad de Datos
- **NO recalcular métricas** que la vista ya expone. Usar el campo directamente.
- **Fidelización y Cortesías** provienen de `dw.fact_ventas` filtrando `tipo_precio`. Usar las vistas especializadas.
- **Transacciones vs. Líneas de venta**:
  - Métricas de transacciones (`transacciones`, `ventas_hoy`, etc.) vienen de `dw.fact_transacciones`.
  - Métricas de productos/unidades (`unidades_vendidas`, `cantidad`, `producto`) vienen de `dw.fact_ventas`.

### 1.3 Anti-patrones (PROHIBIDO)
- Filtrar por fecha vistas `_latest`, `sales_review_day`, `sales_week` o `dashboard_canjes_resumen`.
- Usar `sales_review_day_history` para "hoy". Para hoy usar `sales_review_day`.
- Usar una columna `franja_horaria` en `mart_operacion_hora`; no existe. Usar `hora`.
- Usar `dashboard_participacion_categorias` como si tuviera las mismas columnas que `kpi_categorias_diario`.

---

## 2. Taxonomía de Vistas (Mapeo por Intención)

## 2. Taxonomía de Vistas (Diccionario de Intenciones y Mapeo Semántico)

Cuando el usuario formule una pregunta, mapea su intención a la vista técnica correcta utilizando la siguiente tabla. La columna "Palabras clave / Preguntas típicas" te ayudará a contextualizar el lenguaje natural del usuario.

| Intención Analítica | Palabras clave / Preguntas típicas del usuario | Vista a usar | Filtro de fecha | Columna de fecha |
|:---|:---|:---|:---:|:---:|
| **Resumen del día actual** | "¿Cómo vamos hoy?", "Ventas de ayer", "Resumen actual", "Transacciones de hoy" | `sales_review_day` | **N/A** | N/A |
| **Snapshot de sedes actual** | "Ventas por sucursal hoy", "¿Cómo van los locales hoy?", "Snapshot actual de sedes" | `sales_review_locales_latest` | **N/A** | N/A |
| **Comparativa semanal** | "Comparativa semanal", "Esta semana vs la pasada", "Cómo vamos esta semana" | `sales_week` | **N/A** | N/A |
| **Tendencias diarias globales** | "Tendencia de ventas", "Histórico diario", "Evolución de ventas", "Ventas de los últimos X días" | `sales_review_day_history` | Rango | `fecha` |
| **Histórico por sede** | "Ventas por sucursal históricas", "Desempeño por local en [mes/rango]", "Qué sucursal vendió más en [periodo]" | `sales_review_locales` | Rango | `fecha_completa` |
| **Ventas por producto / categoría** | "Qué producto se vendió más", "Ventas por categoría", "Unidades vendidas de [producto]", "Desempeño de [producto] en [sucursal]" | `sales_producto_daily` | Rango | `fecha` |
| **Demanda por hora (Hora Pico)** | "Hora pico", "Demanda por hora", "A qué hora vendemos más", "Transacciones por hora" | `mart_operacion_hora` | Rango / `CURRENT_DATE` | `fecha` |
| **Fidelización (Mes actual)** | "Canjes de este mes", "Fidelización actual", "Puntos canjeados" | `dashboard_canjes_resumen` | **N/A** | N/A |
| **Fidelización (Histórico)** | "Detalle de canjes", "Qué productos se han canjeado", "Histórico de fidelización en [rango]" | `kpi_fidelizacion_detalle` | Rango | `fecha` |
| **Cortesías (Resumen mensual)** | "Cortesías de este mes", "Productos regalados", "Impacto de cortesías por sucursal" | `dashboard_cortesias_resumen` | Opcional (`mes`) | `mes` |
| **Cortesías (Histórico)** | "Detalle de cortesías", "Qué se ha regalado", "Histórico de cortesías en [rango]" | `kpi_cortesia_detalle` | Rango | `fecha` |
| **Categorías (Diario)** | "Ventas por categoría diarias", "Desempeño de categorías a lo largo del tiempo", "Evolución de [categoria]" | `kpi_categorias_diario` | Rango | `fecha` |
| **Participación por categoría** | "Participación porcentual por categoría", "Porcentaje por categoría", "Subtotal diario por categoría" | `dashboard_participacion_categorias` | Rango | `fecha_completa` |
| **Desglose completo (Cat+Prod+Sede)** | "Ventas por categoría, producto y sede", "Desglose completo de ventas en [rango]", "Análisis cruzado de categoría y producto por sucursal" | `kpi_categorias_productos_sede` | Rango | `fecha_venta` |

### Reglas de Desambiguación
- Si el usuario dice **"hoy"** o **"ayer"** pero pregunta por un **desglose por sucursal**, usa `sales_review_locales_latest` (no `sales_review_day`, ya que este último no tiene la dimensión sucursal).
- Si el usuario pide **"comparativa de ventas de hoy vs ayer"** a nivel global, usa `sales_review_day` (ya expone `variacion_diaria_pct`).
- Si el usuario pregunta por **"unidades"** o **"productos"** específicos, casi siempre requerirá un rango de fechas (`sales_producto_daily` o `kpi_categorias_productos_sede`). No uses `_latest` para esto.

---

## 3. Vistas Semánticas — Definición de Métricas

### sales_review_day
- **Descripción**: Resumen del día más reciente con datos exitosos/pagados vs el día anterior.
- **Tipo**: current / daily snapshot.
- **Filas**: 1 fila.
- **Filtro fecha**: No aplica (no tiene columna `fecha`).
- **Métricas**:
  - `ventas_hoy`: Suma monetaria del último día con transacciones exitosas/pagadas.
  - `ventas_ayer`: Suma monetaria del día anterior al último día con datos.
  - `transacciones_hoy`: Cantidad de transacciones exitosas/pagadas del último día.
  - `ventas_total_mes`: Suma monetaria acumulada del mes del último día con datos.
  - `transacciones_mes`: Cantidad de transacciones acumuladas del mes del último día.
  - `variacion_diaria_pct`: Crecimiento porcentual de `ventas_hoy` vs `ventas_ayer`.
  - `estado_ventas_diarias`: Indicador textual ('📈 Aumento', '📉 Baja', 'Estable').

### sales_review_day_history
- **Descripción**: Histórico diario de ventas para tendencias y análisis por periodo.
- **Tipo**: historical.
- **Granularidad**: fecha.
- **Filtro fecha**: Rango obligatorio sobre `fecha`.
- **Métricas**:
  - `fecha`: Fecha calendario del registro.
  - `fecha_key`: Clave surrogada de fecha.
  - `ventas_dia`: Suma monetaria de transacciones del día.
  - `transacciones_dia`: Cantidad de transacciones del día.
- **Nota**: No expone `ticket_promedio`. Calcular manualmente si es necesario.

### sales_review_locales_latest
- **Descripción**: Snapshot de ventas del día más reciente **por sede**.
- **Tipo**: current / daily snapshot.
- **Granularidad**: sede.
- **Filtro fecha**: No aplica.
- **Métricas**:
  - `nombre_sede`: Nombre de la sede.
  - `fecha_completa`: Última fecha con datos de esa sede específica.
  - `venta_total`: Suma monetaria de transacciones de la sede en esa fecha.
  - `total_transacciones`: Cantidad de transacciones de la sede en esa fecha.
  - `ticket_promedio`: Promedio de venta por transacción (`venta_total / total_transacciones`).

### sales_review_locales
- **Descripción**: Histórico diario de ventas desagregado por sede.
- **Tipo**: historical.
- **Granularidad**: fecha, sede.
- **Filtro fecha**: Rango sobre `fecha_completa`.
- **Métricas**:
  - `nombre_sede`: Nombre de la sede.
  - `fecha_completa`: Fecha calendario del registro.
  - `venta_total`: Suma monetaria de transacciones del día/sede.
  - `total_transacciones`: Cantidad de transacciones del día/sede.
  - `ticket_promedio`: Promedio de venta por transacción (`venta_total / total_transacciones`).

### sales_week
- **Descripción**: Comparativa de la semana actual vs la semana anterior.
- **Tipo**: compare_periods.
- **Filas**: 1 fila.
- **Filtro fecha**: No aplica.
- **Métricas**:
  - `ventas_semana_actual`: Suma monetaria de la semana en curso.
  - `transacciones_semana_actual`: Cantidad de transacciones de la semana en curso.
  - `ventas_semana_anterior`: Suma monetaria de la semana previa.
  - `transacciones_semana_anterior`: Cantidad de transacciones de la semana previa.
  - `variacion_pct`: Cambio porcentual entre semana actual y anterior.

### sales_producto_daily
- **Descripción**: Histórico de ventas por producto, categoría, subcategoría y sede.
- **Tipo**: historical.
- **Granularidad**: fecha, producto, sucursal.
- **Filtro fecha**: Rango sobre `fecha`.
- **Métricas**:
  - `fecha`: Fecha calendario del registro.
  - `sucursal`: Nombre de la sede.
  - `producto`: Nombre del producto.
  - `categoria`: Categoría normalizada (`categoria_nueva` o `categoria` original).
  - `subcategoria`: Subcategoría normalizada.
  - `tipo_producto`: Tipo del producto.
  - `ventas`: Suma monetaria neta (`precio_neto`) de líneas de venta.
  - `unidades`: Cantidad de unidades vendidas.

### mart_operacion_hora
- **Descripción**: Demanda operativa desagregada por sede, día y hora.
- **Tipo**: historical / current.
- **Granularidad**: sede, día, hora.
- **Filtro fecha**: Rango sobre `fecha` o `fecha = CURRENT_DATE`.
- **Métricas**:
  - `nombre_sede`: Nombre de la sede.
  - `nombre_dia_semana`: Día de la semana del registro.
  - `fecha`: Fecha calendario del registro.
  - `hora`: Hora del día (0-23).
  - `transacciones`: Cantidad de transacciones en esa hora.
  - `ingreso`: Suma monetaria de transacciones en esa hora.
  - `ticket_promedio`: Promedio de venta por transacción en esa hora.

### dashboard_canjes_resumen
- **Descripción**: Resumen mensual de canjes del programa de fidelización.
- **Tipo**: dashboard / current.
- **Granularidad**: mes, sede.
- **Filtro fecha**: No aplica (fija al mes actual).
- **Métricas**:
  - `mes`: Primer día del mes del resumen.
  - `sucursal`: Nombre de la sede.
  - `unidades_totales`: Unidades canjeadas por fidelización.
  - `valor_total_canjes`: Valor neto (`precio_neto`) de los canjes.

### kpi_fidelizacion_detalle
- **Descripción**: Detalle histórico de productos canjeados por fidelización.
- **Tipo**: historical / current.
- **Granularidad**: fecha, sede, producto, categoría.
- **Filtro fecha**: Rango sobre `fecha`.
- **Métricas**:
  - `fecha`: Fecha calendario del canje.
  - `sucursal`: Nombre de la sede.
  - `producto`: Nombre del producto canjeado.
  - `categoria`: Categoría del producto.
  - `unidades_fidelizacion`: Unidades canjeadas.
  - `valor_fidelizacion`: Valor neto de los canjes.

### dashboard_cortesias_resumen
- **Descripción**: Resumen mensual de cortesías otorgadas.
- **Tipo**: dashboard.
- **Granularidad**: mes, sede.
- **Filtro fecha**: Opcional por `mes`.
- **Métricas**:
  - `mes`: Primer día del mes del resumen.
  - `sucursal`: Nombre de la sede.
  - `productos_regalados`: Transacciones distintas con al menos una cortesía.
  - `unidades_totales`: Unidades totales de cortesía.
  - `valor_impacto_total`: Valor neto total de cortesías.

### kpi_cortesia_detalle
- **Descripción**: Detalle histórico de productos otorgados como cortesía.
- **Tipo**: historical / current.
- **Granularidad**: fecha, sede, producto, categoría.
- **Filtro fecha**: Rango sobre `fecha`.
- **Métricas**:
  - `fecha`: Fecha calendario de la cortesía.
  - `sucursal`: Nombre de la sede.
  - `producto`: Nombre del producto.
  - `categoria`: Categoría del producto.
  - `unidades_cortesia`: Unidades de cortesía.
  - `valor_cortesia`: Valor neto de cortesías.

### kpi_categorias_diario
- **Descripción**: Ventas agregadas por categoría a nivel diario.
- **Tipo**: historical / current.
- **Granularidad**: fecha, categoría.
- **Filtro fecha**: Rango sobre `fecha`.
- **Métricas**:
  - `fecha`: Fecha calendario del registro.
  - `categoria`: Categoría normalizada del producto.
  - `ventas`: Suma monetaria neta por categoría.
  - `unidades`: Unidades vendidas por categoría.

### dashboard_participacion_categorias
- **Descripción**: Subtotal diario por categoría para análisis de participación.
- **Tipo**: historical / current.
- **Granularidad**: fecha, categoría.
- **Filtro fecha**: Rango sobre `fecha_completa`.
- **Métricas**:
  - `fecha_completa`: Fecha calendario del registro.
  - `categoria`: Categoría normalizada del producto.
  - `subtotal_diario`: Suma monetaria neta por categoría.
  - `unidades_vendidas`: Unidades vendidas por categoría.
- **Nota**: Columnas distintas a `kpi_categorias_diario`. No mezclar.

### kpi_categorias_productos_sede
- **Descripción**: Ventas por categoría, producto y sede desagregadas por día.
- **Tipo**: historical / current.
- **Granularidad**: fecha, sede, categoría, producto.
- **Filtro fecha**: Rango sobre `fecha_venta`.
- **Métricas**:
  - `fecha_venta`: Fecha calendario del registro.
  - `sucursal`: Nombre de la sede.
  - `categoria`: Categoría normalizada.
  - `producto`: Nombre del producto.
  - `unidades_totales`: Unidades vendidas.
  - `ventas_totales`: Suma monetaria neta.

---

## 4. Guía de Ejecución Técnica

### 4.1 Esquema y motor
- **Motor**: PostgreSQL (Supabase).
- **Esquema de vistas semánticas**: `semantic`.
- Todas las queries deben usar: `FROM semantic.<vista>`.

### 4.2 Sintaxis de filtros de fecha
- Formato: `YYYY-MM-DD`.
- Rango mensual recomendado:
  ```sql
  WHERE fecha >= '2024-06-01'
    AND fecha < '2024-07-01'
