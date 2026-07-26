import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()  # Carga el archivo .env que esté en la misma carpeta

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Faltan SUPABASE_URL o SUPABASE_ANON_KEY en el .env")

supabase = create_client(url, key)


# Probar registro
email = "contacto@qantyxlab.cl"
password = "demo123"

auth = supabase.auth.sign_up({"email": email, "password": password})
print("SIGN UP:", auth)

# Probar login
session = supabase.auth.sign_in_with_password({"email": email, "password": password})
print("ACCESS TOKEN:", session.session.access_token)
