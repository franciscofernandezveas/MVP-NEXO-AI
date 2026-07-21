import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "")
    DB_NAME = os.getenv("DB_NAME", "")
    DB_PORT = os.getenv("DB_PORT", "6543")

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise Exception("❌ Faltan variables de entorno de base de datos")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Asegurar prefijo postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            dsn=DATABASE_URL,
        )
    return _pool


def get_connection():
    conn = _get_pool().getconn()

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        conn = psycopg2.connect(DATABASE_URL)

    return conn


def release_connection(conn):
    _get_pool().putconn(conn)


def close_all_connections():
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
