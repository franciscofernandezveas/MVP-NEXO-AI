from fastapi import APIRouter, Depends, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import psycopg2
import psycopg2.extras
import os
import decimal
import json
import urllib.request
import urllib.error
import logging
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/reports", tags=["reports"])
logger = logging.getLogger("report_routes")
security = HTTPBearer(auto_error=False)

SUPABASE_URL = os.getenv("NEXO_SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("NEXO_SUPABASE_SERVICE_KEY", "")


# ═══════════════════════════════════════════════════════════════
# AUTENTICACIÓN: VALIDACIÓN REMOTA CON SUPABASE AUTH
# ═══════════════════════════════════════════════════════════════

class SimpleUser:
    def __init__(self, email: str, user_id: str = None):
        self.email = email
        self.id = user_id or email


def validate_token_with_supabase(token: str) -> dict:
    """
    Valida el access_token llamando al endpoint /auth/v1/user de Supabase.
    """
    if not SUPABASE_URL:
        raise RuntimeError("NEXO_SUPABASE_URL no está configurado")
    if not SUPABASE_SERVICE_KEY:
        raise RuntimeError("NEXO_SUPABASE_SERVICE_KEY no está configurado")

    url = f"{SUPABASE_URL}/auth/v1/user"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        logger.warning(f"Supabase Auth respondió {e.code}: {error_body}")
        if e.code == 401:
            raise Exception("Token inválido o expirado")
        raise Exception(f"Error de autenticación Supabase: {e.code}")
    except urllib.error.URLError as e:
        logger.error(f"No se pudo conectar a Supabase Auth: {e.reason}")
        raise Exception("No se pudo conectar al servicio de autenticación")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials if credentials else None

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_data = validate_token_with_supabase(token)
        email = user_data.get("email") or user_data.get("id") or "usuario"
        user_id = user_data.get("id")
        return SimpleUser(email=email, user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Error validando token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
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


# ═══════════════════════════════════════════════════════════════
# TABLA GENÉRICA
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

        return {
            "sales_summary": sales_summary,
            "sales_by_store": sales_by_store,
            "weekly_demand": weekly_demand,
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
        {format_table(data["weekly_demand"], "3. Comparación Semanal")}

        <footer>
            Generado automáticamente por el sistema de reportes · Portacafe BI
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
