from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

from psycopg2.extras import RealDictCursor

from database.connection import get_connection, release_connection


@contextmanager
def db_cursor(commit: bool = False):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_connection(conn)


def create_user_db(
    email: str,
    password_hash: str,
    full_name: Optional[str] = None,
    role: str = "user",
) -> dict:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO app_auth.users (email, password_hash, full_name, role)
            VALUES (%s, %s, %s, %s)
            RETURNING id, email, full_name, role, is_active, email_verified
            """,
            (email.lower(), password_hash, full_name, role),
        )
        return dict(cursor.fetchone())


def get_user_by_email(email: str) -> Optional[dict]:
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, email, password_hash, full_name, role, is_active, email_verified
            FROM app_auth.users
            WHERE email = %s
            LIMIT 1
            """,
            (email.lower(),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict]:
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, email, full_name, role, is_active, email_verified
            FROM app_auth.users
            WHERE id = %s
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def update_user_password(user_id: str, password_hash: str):
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE app_auth.users
            SET password_hash = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (password_hash, user_id),
        )


def store_refresh_token_db(
    user_id: str,
    token_hash: str,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
):
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO app_auth.refresh_tokens (user_id, token_hash, expires_at, ip_address, user_agent)
            VALUES (%s, %s, %s, %s::inet, %s)
            """,
            (user_id, token_hash, expires_at, ip, user_agent),
        )


def verify_refresh_token_db(token_hash: str) -> Optional[dict]:
    with db_cursor(commit=True) as cursor:
        # Limpieza de tokens expirados
        cursor.execute(
            "DELETE FROM app_auth.refresh_tokens WHERE expires_at < NOW()"
        )

        cursor.execute(
            """
            SELECT rt.id, u.id AS user_id, u.email, u.full_name, u.role, u.is_active, u.email_verified
            FROM app_auth.refresh_tokens rt
            JOIN app_auth.users u ON rt.user_id = u.id
            WHERE rt.token_hash = %s
              AND rt.expires_at > NOW()
              AND rt.revoked_at IS NULL
            LIMIT 1
            """,
            (token_hash,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        # Rotación: revocar token usado
        cursor.execute(
            "UPDATE app_auth.refresh_tokens SET revoked_at = NOW() WHERE id = %s",
            (row["id"],),
        )
        return dict(row)


def revoke_refresh_token_db(token_hash: str):
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE app_auth.refresh_tokens
            SET revoked_at = NOW()
            WHERE token_hash = %s AND revoked_at IS NULL
            """,
            (token_hash,),
        )


def revoke_all_user_refresh_tokens_db(user_id: str):
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE app_auth.refresh_tokens
            SET revoked_at = NOW()
            WHERE user_id = %s AND revoked_at IS NULL
            """,
            (user_id,),
        )
