# database/config.py
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de PostgreSQL/Supabase
DB_USER = os.getenv("DB_USER", "postgres.qfchrxeaqhvrpyesarqb")
DB_PASSWORD = os.getenv("DB_PASSWORD", "2801")
DB_HOST = os.getenv("DB_HOST", "aws-0-sa-east-1.pooler.supabase.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_PORT = os.getenv("DB_PORT", "6543")

# URI de conexión construida
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# También puedes usar directamente DATABASE_URL si está en el .env
DATABASE_URL = os.getenv("DATABASE_URL", DB_URI)

# Configuración de Supabase (para RAG si lo necesitas)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
