from contextlib import contextmanager
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


def get_user_by_supabase_uid(supabase_uid: str) -> Optional[dict]:
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, email, full_name, role, is_active, email_verified
            FROM app_auth.users
            WHERE id = %s
            LIMIT 1
            """,
            (supabase_uid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, email, full_name, role, is_active, email_verified
            FROM app_auth.users
            WHERE email = %s
            LIMIT 1
            """,
            (email.lower(),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def upsert_user_from_supabase(
    supabase_uid: str,
    email: str,
    full_name: Optional[str] = None,
    role: str = "user",
    email_verified: bool = False,
) -> dict:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO app_auth.users (id, email, full_name, role, email_verified)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                email = EXCLUDED.email,
                full_name = COALESCE(EXCLUDED.full_name, app_auth.users.full_name),
                role = COALESCE(app_auth.users.role, EXCLUDED.role),
                email_verified = EXCLUDED.email_verified,
                updated_at = NOW()
            RETURNING id, email, full_name, role, is_active, email_verified
            """,
            (supabase_uid, email.lower(), full_name, role, email_verified),
        )
        return dict(cursor.fetchone())


def set_user_role_db(supabase_uid: str, role: str) -> Optional[dict]:
    with db_cursor(commit=True) as cursor:
        cursor.execute(
            """
            UPDATE app_auth.users
            SET role = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING id, email, full_name, role, is_active, email_verified
            """,
            (role, supabase_uid),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
