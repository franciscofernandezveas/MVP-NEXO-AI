import getpass
import sys

from .db import create_user_db
from .security import hash_password


def create_admin():
    email = input("Email admin: ").strip()
    password = getpass.getpass("Contraseña: ")
    full_name = input("Nombre completo: ").strip() or None

    password_hash = hash_password(password)
    row = create_user_db(email, password_hash, full_name, role="admin")

    print(f"Usuario admin creado: {row['email']} (id={row['id']}, role={row['role']})")


if __name__ == "__main__":
    # Para ejecutar desde backend/: python -m auth.seed
    create_admin()
