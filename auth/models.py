from typing import Optional
from pydantic import BaseModel


class User(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    email_verified: bool = False


# Estos modelos ya no se usan con Supabase Auth, los dejo por si los usas en otro lugar.
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "user"


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
