import sys

from .db import get_user_by_email, set_user_role_db


def make_admin():
    email = input("Email del usuario admin: ").strip()
    user = get_user_by_email(email)
    if not user:
        print(f"No existe perfil local para {email}. Inicia sesión primero.")
        sys.exit(1)

    set_user_role_db(user["id"], "admin")
    print(f"{email} ahora es admin (id={user['id']})")


if __name__ == "__main__":
    make_admin()
