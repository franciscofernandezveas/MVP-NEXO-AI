from fastapi import APIRouter, Depends, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import psycopg2
import psycopg2.extras
import os
import decimal
import jwt
import logging
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/reports", tags=["reports"])
logger = logging.getLogger("report_routes")
security = HTTPBearer(auto_error=False)

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_JWT_PUBLIC_KEY = os.getenv("SUPABASE_JWT_PUBLIC_KEY", "")


# ═══════════════════════════════════════════════════════════════
# AUTENTICACIÓN SUPABASE AUTH
# ═══════════════════════════════════════════════════════════════

class SimpleUser:
    def __init__(self, email: str, user_id: str = None):
        self.email = email
        self.id = user_id or email


def _decode_with_secret(token: str, secret: str, algorithms: list):
    return jwt.decode(
        token,
        secret,
        algorithms=algorithms,
        options={"verify_aud": False},
    )


def decode_supabase_token(token: str) -> dict:
    """
    Intenta decodificar un JWT de Supabase Auth.
    Soporta HS256 (con JWT Secret) y RS256 (con JWT Public Key).
    """
    # Primero vemos el header sin verificar firma para saber el algoritmo
    try:
        header = jwt.get_unverified_header(token)
        logger.info(f"🔐 Token header: {header}")
    except Exception as e:
        logger.error(f"❌ No se pudo leer header del token: {e}")
        raise jwt.InvalidTokenError("Token malformado")

    alg = header.get("alg", "HS256")
    logger.info(f"🔐 Algoritmo del token: {alg}")

    if alg == "HS256":
        if not SUPABASE_JWT_SECRET:
            raise RuntimeError("SUPABASE_JWT_SECRET no está configurado para token HS256")
        return _decode_with_secret(token, SUPABASE_JWT_SECRET, ["HS256"])

    if alg == "RS256":
        if not SUPABASE_JWT_PUBLIC_KEY:
            raise RuntimeError("SUPABASE_JWT_PUBLIC_KEY no está configurado para token RS256")
        return _decode_with_secret(token, SUPABASE_JWT_PUBLIC_KEY, ["RS256"])

    raise jwt.InvalidTokenError(f"Algoritmo {alg} no soportado")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials if credentials else None

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_supabase_token(token)
        if payload.get("type") not in (None, "access"):
            raise jwt.InvalidTokenError("Tipo de token inválido")

        email = payload.get("email") or payload.get("sub") or "usuario"
        user_id = payload.get("sub")
        return SimpleUser(email=email, user_id=user_id)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado",
        )
    except RuntimeError as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error de configuración del servidor",
        )
    except Exception as e:
        logger.warning(f"Token inválido: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
        )


# ═══════════════════════════════════════════════════════════════
# CONEXIÓN A BASE DE DATOS
# ═══════════════════════════════════════════════════════════════

def get_db_url() -> str:
    candidates = [
        os.getenv("DATABASE_URL"),
        os.getenv("DEMO_DATABASE_URL"),
        os.getenv("NEXO_SUPABASE_URL"),
    ]
    for url in candidates:
        if url and url.startswith("postgresql://"):
            return url
    raise RuntimeError(
        "No se encontró una URL de PostgreSQL válida. "
        "Configura DATABASE_URL, DEMO_DATABASE_URL o NEXO_SUPABASE_URL."
    )


def get_db_connection():
    try:
        return psycopg2.connect(get_db_url())
    except Exception as e:
        logger.error(f"❌ Error conectando a la base de datos: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo conectar a la base de datos.",
        )


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def fmt_num(val, decimals=0):
    if val is None:
        return "0"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if decimals == 0 or v == int(v):
        return f"{int(v):,}".replace(",", ".")
    temp = f"{v:,.{decimals}f}"
    return temp.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")


def compact_hours(hours_str):
    if not hours_str:
        return "–"
    try:
        nums = sorted({int(x.strip()) for x in str(hours_str).split(",") if x.strip().isdigit()})
    except ValueError:
        return str(hours_str)
    if not nums:
        return str(hours_str)

    ranges = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}h" if start == prev else f"{start}h-{prev}h")
            start = prev = n
    ranges.append(f"{start}h" if start == prev else f"{start}h-{prev}h")
    return ", ".join(ranges)


def compact_hours_clock(hours_str):
    if not hours_str:
        return "–"
    try:
        nums = sorted({int(x.strip()) for x in str(hours_str).split(",") if x.strip().isdigit()})
    except ValueError:
        return str(hours_str)
    if not nums:
        return str(hours_str)

    ranges = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start:02d}:00" if start == prev else f"{start:02d}:00 - {prev:02d}:00")
            start = prev = n
    ranges.append(f"{start:02d}:00" if start == prev else f"{start:02d}:00 - {prev:02d}:00")
    return ", ".join(ranges)


def fmt_hora(h):
    if h is None:
        return "–"
    try:
        return f"{int(h):02d}:00"
    except (ValueError, TypeError):
        return str(h)


def action_suggestion(row):
    conc = row.get("concentracion_top_horas") or 0
    hv = row.get("horas_valle") or ""
    valle_count = len([x for x in str(hv).split(",") if x.strip().isdigit()]) if hv else 0

    if conc < 50:
        return "Revisar estrategia completa de sede"
    if valle_count >= 4:
        return "Reforzar demanda en horario extendido"
    if valle_count >= 2:
        return "Activar promociones en horarios intermedios"
    if valle_count == 1:
        return "Evaluar ajuste de turno específico"
    return "Mantener estrategia actual"


# ═══════════════════════════════════════════════════════════════
# TABLA GENÉRICA (items 1-3)
# ═══════════════════════════════════════════════════════════════

def format_table(rows, title):
    if not rows:
        return f'<div class="section"><h3>{title}</h3><p style="color:#888;font-style:italic;">Sin datos.</p></div>'

    raw_keys = rows[0].keys()
    headers = [k.replace("_", " ").title() for k in raw_keys]

    money_kw = [
        "venta","ventas","total","amount","price","precio","ticket",
        "promedio","average","vendido","sold","revenue","income",
        "ingreso","costo","cost","saldo"
    ]
    pct_kw = [
        "variacion","variación","variation","cambio","change","growth",
        "crecimiento","delta","diferencia","diff","porcentaje","percent",
        "%","vs","comparacion","comparison","concentracion","ratio","proporcion"
    ]

    html = f'<div class="section"><h3>{title}</h3><table>'
    html += "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"

    for row in rows:
        html += "<tr>"
        for key in raw_keys:
            val = row[key]
            kl = key.lower()
            cls = []
            cell = ""

            if isinstance(val, (int, float, decimal.Decimal)) and val is not None:
                is_pct = any(p in kl for p in pct_kw)
                is_money = any(m in kl for m in money_kw)

                if is_pct:
                    cell = fmt_num(float(val), 0) + "%"
                    cls += ["numeric", "positive" if float(val) >= 0 else "negative"]
                elif is_money:
                    cell = "$" + fmt_num(val, 0)
                    cls += ["numeric", "money"]
                else:
                    cell = fmt_num(val, 0)
                    cls.append("numeric")
            else:
                cell = str(val) if val is not None else "–"
                if "\n" in cell:
                    cell = cell.replace("\n", "<br>")

            cls_str = " ".join(cls)
            html += f'<td class="{cls_str}">{cell}</td>'
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


# ═══════════════════════════════════════════════════════════════
# TABLA OPERATIVA (ítem 4)
# ═══════════════════════════════════════════════════════════════

def format_operational_table(rows, title):
    if not rows:
        return f'<div class="section"><h3>{title}</h3><p style="color:#888;font-style:italic;">Sin datos.</p></div>'

    display_cols = [
        "nombre_sede", "horas_peak", "trans_peak", "ingreso_peak",
        "horas_valle", "trans_valle", "horas_ticket", "max_ticket",
        "concentracion_top_horas"
    ]
    header_map = {
        "nombre_sede": "SEDE",
        "horas_peak": "HORAS PEAK",
        "trans_peak": "TRANS. PEAK",
        "ingreso_peak": "INGRESO PEAK",
        "horas_valle": "HORAS VALLE",
        "trans_valle": "TRANS. VALLE",
        "horas_ticket": "HORA TICKET",
        "max_ticket": "MAX. TICKET",
        "concentracion_top_horas": "% TOP HORAS"
    }

    html = f'<div class="section"><h3>{title}</h3><table class="operational-table"><thead><tr>'
    for k in display_cols:
        html += f'<th>{header_map.get(k, k)}</th>'
    html += "</tr></thead><tbody>"

    for row in rows:
        html += '<tr class="ops-row">'
        for key in display_cols:
            val = row.get(key)

            if key == "nombre_sede":
                html += f'<td><span class="sede-badge">{val}</span></td>'

            elif key in ("horas_peak", "horas_ticket") and val:
                tag_cls = "tag-peak" if key == "horas_peak" else "tag-ticket"
                dot = "dot-peak" if key == "horas_peak" else "dot-ticket"
                hours = [h.strip() for h in str(val).split(",") if h.strip()]
                tags = "".join([
                    f'<span class="hour-tag {tag_cls}"><span class="{dot}"></span>{h}h</span>'
                    for h in hours
                ])
                html += f"<td>{tags}</td>"

            elif key == "horas_valle":
                html += f'<td><span class="valle-text">{compact_hours(val)}</span></td>'

            elif key == "concentracion_top_horas":
                pct = int(val) if val else 0
                html += f'<td><span class="pct-pill">{pct}%</span></td>'

            elif key in ("ingreso_peak", "max_ticket"):
                money = f"${fmt_num(val, 0)}" if val else "–"
                html += f'<td class="numeric money">{money}</td>'

            elif isinstance(val, (int, float, decimal.Decimal)):
                html += f'<td class="numeric">{fmt_num(val, 0)}</td>'
            else:
                html += f'<td>{val if val is not None else "–"}</td>'
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


# ═══════════════════════════════════════════════════════════════
# PANEL DE CONTROL (ítem 5)
# ═══════════════════════════════════════════════════════════════

def format_control_panel(rows, title):
    if not rows:
        return f'<div class="section"><h3>{title}</h3><p style="color:#888;font-style:italic;">Sin datos.</p></div>'

    html = f'<div class="section"><h3>{title}</h3><div class="cards-grid">'

    for row in rows:
        sede = row.get("nombre_sede", "Sede")
        hp = row.get("horas_peak")
        tp = row.get("trans_peak") or 0
        ip = row.get("ingreso_peak") or 0
        hv = row.get("horas_valle")
        tv = row.get("trans_valle") or 0
        ht = row.get("horas_ticket")
        mt = row.get("max_ticket") or 0
        conc = row.get("concentracion_top_horas") or 0
        iv = row.get("inicio_valle")
        fv = row.get("fin_valle")

        peak_txt = compact_hours_clock(hp) if hp else "–"
        valle_txt = compact_hours_clock(hv) if hv else "–"
        ticket_txt = compact_hours_clock(ht) if ht else "–"

        if iv is None and fv is None:
            crit_txt = "Sin horas críticas detectadas"
        else:
            crit_txt = f"{fmt_hora(iv)} - {fmt_hora(fv)}"

        accion = action_suggestion(row)

        html += f'''
        <div class="control-card">
            <div class="card-header">
                <span class="card-pin">&#128205;</span>
                <h4>{sede}</h4>
            </div>
            <div class="card-body">
                <div class="metric">
                    <span class="m-icon">&#128293;</span>
                    <span class="m-txt">Hora peak: {peak_txt} ({fmt_num(tp, 0)} transacciones, ${fmt_num(ip, 0)})</span>
                </div>
                <div class="metric">
                    <span class="m-icon">&#128308;</span>
                    <span class="m-txt">Hora valle: {valle_txt} ({fmt_num(tv, 0)} transacciones)</span>
                </div>
                <div class="metric">
                    <span class="m-icon">&#128176;</span>
                    <span class="m-txt">Mayor ticket: {ticket_txt} (${fmt_num(mt, 0)})</span>
                </div>
                <div class="metric">
                    <span class="m-icon">&#128202;</span>
                    <span class="m-txt">{conc}% de ventas concentradas en horas peak</span>
                </div>
                <div class="metric">
                    <span class="m-icon">&#9888;</span>
                    <span class="m-txt">Horas críticas: {crit_txt}</span>
                </div>
            </div>
            <div class="card-action">
                <span class="action-label">Acción sugerida</span>
                <p>{accion}</p>
            </div>
        </div>
        '''
    html += "</div></div>"
    return html


# ═══════════════════════════════════════════════════════════════
# QUERY OPERATIVA
# ═══════════════════════════════════════════════════════════════

OPERATIONAL_QUERY = r"""
WITH ultima_fecha AS (
    SELECT MAX(fecha) AS fecha FROM semantic.mart_operacion_hora_kpi
),
base AS (
    SELECT * FROM semantic.mart_operacion_hora_kpi
    WHERE fecha = (SELECT fecha FROM ultima_fecha)
),
peak AS (
    SELECT nombre_sede,
        STRING_AGG(hora::text, ', ' ORDER BY hora) AS horas_peak,
        MAX(total_transacciones) AS trans_peak,
        MAX(ingreso_total) AS ingreso_peak
    FROM base WHERE rank_demanda = 1 GROUP BY nombre_sede
),
valle AS (
    SELECT nombre_sede,
        STRING_AGG(hora::text, ', ' ORDER BY hora) AS horas_valle,
        MIN(total_transacciones) AS trans_valle
    FROM base WHERE rank_baja_demanda = 1 GROUP BY nombre_sede
),
ticket AS (
    SELECT nombre_sede,
        STRING_AGG(hora::text, ', ' ORDER BY hora) AS horas_ticket,
        MAX(ticket_promedio) AS max_ticket
    FROM base WHERE rank_ticket = 1 GROUP BY nombre_sede
),
concentracion AS (
    SELECT nombre_sede,
        ROUND(
            SUM(CASE WHEN rank_demanda <= 3 THEN total_transacciones ELSE 0 END)::numeric
            / NULLIF(SUM(total_transacciones),0),
        2) AS concentracion_top_horas
    FROM base GROUP BY nombre_sede
),
horas_valle_rango AS (
    SELECT nombre_sede, MIN(hora) AS inicio_valle, MAX(hora) AS fin_valle
    FROM base WHERE demanda_relativa < 0.2 GROUP BY nombre_sede
),
sedes AS ( SELECT DISTINCT nombre_sede FROM base )
SELECT
    s.nombre_sede,
    p.horas_peak, p.trans_peak, p.ingreso_peak,
    v.horas_valle, v.trans_valle,
    t.horas_ticket, t.max_ticket,
    c.concentracion_top_horas,
    hv.inicio_valle, hv.fin_valle
FROM sedes s
LEFT JOIN peak p ON s.nombre_sede = p.nombre_sede
LEFT JOIN valle v ON s.nombre_sede = v.nombre_sede
LEFT JOIN ticket t ON s.nombre_sede = t.nombre_sede
LEFT JOIN concentracion c ON s.nombre_sede = c.nombre_sede
LEFT JOIN horas_valle_rango hv ON s.nombre_sede = hv.nombre_sede;
"""


# ═══════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════

def load_report_data():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute("SELECT * FROM semantic.sales_review_day")
        sales_summary = cursor.fetchall()

        cursor.execute("SELECT * FROM semantic.sales_review_locales_latest")
        sales_by_store = cursor.fetchall()

        cursor.execute("SELECT * FROM semantic.sales_week")
        weekly_demand = cursor.fetchall()

        cursor.execute(OPERATIONAL_QUERY)
        operational_summary = cursor.fetchall()

        for row in operational_summary:
            raw = row.get("concentracion_top_horas")
            row["concentracion_top_horas"] = round(float(raw) * 100) if raw is not None else 0

        return {
            "sales_summary": sales_summary,
            "sales_by_store": sales_by_store,
            "weekly_demand": weekly_demand,
            "operational_summary": operational_summary,
        }
    except HTTPException:
        raise
    except psycopg2.Error as e:
        logger.error(f"❌ Error de base de datos generando reporte: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al consultar datos del reporte: {str(e)}",
        )
    except Exception as e:
        logger.error(f"❌ Error generando reporte: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar el reporte: {str(e)}",
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ═══════════════════════════════════════════════════════════════
# CSS PREMIUM
# ═══════════════════════════════════════════════════════════════

CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    body {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: #F5F0EB;
        color: #2C1810;
        padding: 30px;
        margin: 0;
        line-height: 1.5;
    }

    .report-header {
        background: linear-gradient(135deg, #2C1810 0%, #5D4037 100%);
        color: #FFF;
        padding: 35px 40px;
        border-radius: 16px;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px rgba(44,24,16,0.15);
        position: relative;
        overflow: hidden;
    }
    .report-header::before {
        content: "";
        position: absolute;
        top: -50px; right: -50px;
        width: 180px; height: 180px;
        background: rgba(166,124,82,0.15);
        border-radius: 50%;
    }
    .report-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #FFF;
    }
    .report-header .subtitle {
        margin: 8px 0 0;
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.85;
        color: #D7CCC8;
    }
    .date-badge {
        display: inline-block;
        margin-top: 18px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .section {
        background: #FFF;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(44,24,16,0.06);
        margin-bottom: 28px;
        border: 1px solid #EDE0D4;
    }
    .section h3 {
        margin: 0 0 20px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #3E2723;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        font-size: 0.92rem;
    }
    th {
        background-color: #4E342E;
        color: #FFF;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.8px;
        padding: 14px 16px;
        text-align: left;
        text-transform: uppercase;
    }
    td {
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid #F0EAE3;
        color: #444;
        vertical-align: middle;
    }
    tr:nth-child(even) { background-color: #FDFBF7; }
    tr:hover { background-color: #F5F0EB; transition: background-color 0.15s; }
    tr:last-child td { border-bottom: none; }

    td.numeric {
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-weight: 600;
        color: #2C1810;
    }
    td.money { color: #2E7D32; }
    .positive { color: #2E7D32; font-weight: 700; }
    .negative { color: #C62828; font-weight: 700; }

    .operational-table th {
        background: #3E2723;
        font-size: 0.7rem;
        padding: 12px 14px;
        white-space: nowrap;
    }
    .operational-table td {
        padding: 16px 14px;
        vertical-align: middle;
        font-size: 0.9rem;
        border-bottom: 1px solid #EDE0D4;
    }
    .ops-row:hover { background: #F5F0EB; }

    .sede-badge {
        display: inline-block;
        font-weight: 700;
        font-size: 0.95rem;
        color: #3E2723;
        background: #F5F0EB;
        padding: 6px 14px;
        border-radius: 8px;
        border-left: 4px solid #A67C52;
    }

    .hour-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 2px;
        line-height: 1;
        white-space: nowrap;
    }
    .tag-peak { background: #FFF3E0; color: #E65100; border: 1px solid #FFE0B2; }
    .tag-ticket { background: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
    .dot-peak { width: 6px; height: 6px; border-radius: 50%; display: inline-block; background: #E65100; }
    .dot-ticket { width: 6px; height: 6px; border-radius: 50%; display: inline-block; background: #2E7D32; }

    .valle-text {
        font-weight: 600;
        color: #455A64;
        font-size: 0.9rem;
    }

    .pct-pill {
        display: inline-block;
        background: #3E2723;
        color: #FFF;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 12px;
        min-width: 36px;
        text-align: center;
    }

    .cards-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
        gap: 20px;
    }
    .control-card {
        background: #FFF;
        border: 1px solid #EDE0D4;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(44,24,16,0.06);
        display: flex;
        flex-direction: column;
    }
    .card-header {
        background: #F5F0EB;
        padding: 14px 18px;
        border-bottom: 1px solid #EDE0D4;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-pin { font-size: 1.2rem; line-height: 1; }
    .card-header h4 {
        margin: 0;
        font-size: 1.1rem;
        color: #3E2723;
        font-weight: 700;
    }
    .card-body { padding: 16px 18px; flex: 1; }
    .metric {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-bottom: 10px;
        font-size: 0.93rem;
        color: #4E342E;
        line-height: 1.4;
    }
    .m-icon {
        font-size: 1rem;
        width: 22px;
        text-align: center;
        flex-shrink: 0;
        line-height: 1;
    }
    .m-txt { flex: 1; }

    .card-action {
        background: #FFF8E1;
        border-top: 1px solid #FFE0B2;
        padding: 12px 18px;
    }
    .action-label {
        display: block;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700;
        color: #E65100;
        margin-bottom: 4px;
    }
    .card-action p {
        margin: 0;
        font-size: 0.92rem;
        color: #3E2723;
        font-weight: 600;
    }

    footer {
        margin-top: 40px;
        text-align: center;
        color: #A1887F;
        font-size: 0.8rem;
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# ENDPOINT
# ═══════════════════════════════════════════════════════════════

@router.post("/sales-report")
async def generate_sales_report(user: SimpleUser = Depends(get_current_user)):
    try:
        data = load_report_data()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error inesperado en reporte: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar el reporte: {str(e)}",
        )

    today = (datetime.now() - timedelta(days=1)).strftime("%d-%m-%Y")

    html_report = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Ventas</title>
        {CSS_STYLES}
    </head>
    <body>
        <div class="report-header">
            <h1>Reporte Diario de Ventas</h1>
            <span class="date-badge">Fecha de corte: {today}</span>
        </div>

        {format_table(data["sales_summary"], "1. Resumen Diario Total")}
        {format_table(data["sales_by_store"], "2. Ventas por Sucursal (Hoy)")}
        {format_table(data["weekly_demand"], "3. Comparacion Semanal")}
        {format_operational_table(data["operational_summary"], "4. Analisis Operativo por Sede")}
        {format_control_panel(data["operational_summary"], "5. Panel de Control Ejecutivo")}

        <footer>
            Generado automaticamente por el sistema de reportes · Portacafe BI
        </footer>
    </body>
    </html>
    """

    return Response(
        content=html_report,
        media_type="text/html",
        headers={
            "Content-Disposition": "attachment; filename=reporte_ventas.html"
        }
    )
