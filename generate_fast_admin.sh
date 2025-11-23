#!/bin/bash
# ============================================================================
# ENTERPRISE FAST REACT ADMIN GENERATOR - PRODUCTION READY
# ============================================================================
# Features:
# - Hot Reload Development Mode
# - Production with SSL/TLS (Let's Encrypt)
# - Domain-based routing (Frontend only public)
# - Environment-based configuration (.env)
# - Celery Beat + Flower
# - Security hardening
# - Auto SSL renewal
# ============================================================================

set -e

PROJECT_NAME="${1:-fast_react_admin}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 ENTERPRISE FULL STACK ADMIN GENERATOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📦 Creating: $PROJECT_NAME"
echo "✨ Features: Dev Hot Reload | Production SSL | Domain Mapping"
echo ""

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

#==============================================================================
# ENVIRONMENT CONFIGURATION FILES
#==============================================================================
echo "📝 Creating environment configuration files..."

cat > .env.example << 'ENV_EXAMPLE_EOF'
# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# Copy this file to .env.example as .env for development or .env.prod for production

# ----------------------------------------------------------------------------
# ENVIRONMENT MODE
# ----------------------------------------------------------------------------
ENVIRONMENT=development  # development | production

# ----------------------------------------------------------------------------
# APPLICATION
# ----------------------------------------------------------------------------
APP_NAME=Fast React Admin
APP_VERSION=1.0.0
DEBUG=false

# ----------------------------------------------------------------------------
# DOMAIN & SSL (Production Only)
# ----------------------------------------------------------------------------
DOMAIN=admin.example.com
ADMIN_EMAIL=admin@gmail.com

# Set to 'true' for real SSL certificates, 'false' for staging/testing
SSL_PRODUCTION=false

# ----------------------------------------------------------------------------
# SECURITY
# ----------------------------------------------------------------------------
SECRET_KEY=change-this-to-a-random-64-character-string-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# ----------------------------------------------------------------------------
# DATABASE
# ----------------------------------------------------------------------------
DB_USER=admin
DB_PASSWORD=change-in-production-use-strong-password
DB_NAME=fastadmin
DB_HOST=postgres
DB_PORT=5432

# Full DATABASE_URL (auto-constructed if not provided)
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# ----------------------------------------------------------------------------
# REDIS
# ----------------------------------------------------------------------------
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_URL=redis://${REDIS_HOST}:${REDIS_PORT}/0

# ----------------------------------------------------------------------------
# CELERY
# ----------------------------------------------------------------------------
CELERY_BROKER_URL=redis://${REDIS_HOST}:${REDIS_PORT}/1
CELERY_RESULT_BACKEND=redis://${REDIS_HOST}:${REDIS_PORT}/2

# Celery Task Configuration
TASK_CLEANUP_DAYS=30
TASK_REPORT_HOUR=9

# ----------------------------------------------------------------------------
# CORS (Development Only)
# ----------------------------------------------------------------------------
BACKEND_CORS_ORIGINS=["http://localhost:8001","http://localhost:3000"]

# ----------------------------------------------------------------------------
# RATE LIMITING
# ----------------------------------------------------------------------------
RATE_LIMIT_PER_MINUTE=60

# ----------------------------------------------------------------------------
# FLOWER (Celery Monitoring)
# ----------------------------------------------------------------------------
FLOWER_PORT=5555
FLOWER_USER=admin
FLOWER_PASSWORD=change-in-production

# ----------------------------------------------------------------------------
# PORTS (Development Only)
# ----------------------------------------------------------------------------
BACKEND_PORT=8000
FRONTEND_DEV_PORT=3000
NGINX_HTTP_PORT=8001
NGINX_HTTPS_PORT=8443
POSTGRES_PORT=5432
REDIS_PORT=6379
FLOWER_PORT=5555
ENV_EXAMPLE_EOF

cat > .env << 'ENV_DEV_EOF'
# DEVELOPMENT ENVIRONMENT
ENVIRONMENT=development

# Application
APP_NAME=Fast React Admin
APP_VERSION=1.0.0
DEBUG=False
IS_PRODUCTION=False

# Domain (localhost for dev)
DOMAIN=localhost
ADMIN_EMAIL=admin@gmail.com

# Security (weak for dev, strong for prod)
SECRET_KEY=dev-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
DB_USER=admin
DB_PASSWORD=admin123
DB_NAME=fastadmin
DB_HOST=postgres
DB_PORT=5432
DATABASE_URL=postgresql+asyncpg://admin:admin123@postgres:5432/fastadmin

# Docker Dev PostgreSQL
POSTGRES_DB=fastadmin
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB_HOST=postgres
POSTGRES_DB_PORT=5432

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_URL=redis://redis:6379/0

# Celery
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
TASK_CLEANUP_DAYS=30
TASK_REPORT_HOUR=9

# CORS
BACKEND_CORS_ORIGINS=http://localhost:8001,http://localhost:3000,http://localhost

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# Flower
FLOWER_USER=admin
FLOWER_PASSWORD=admin123

# Ports
BACKEND_PORT=8000
FRONTEND_DEV_PORT=3000
NGINX_HTTP_PORT=8001
NGINX_HTTPS_PORT=8443
POSTGRES_PORT=5432
REDIS_PORT=6379
FLOWER_PORT=5555
ENV_DEV_EOF

cat > .env.prod << 'ENV_PROD_EOF'
# PRODUCTION ENVIRONMENT
ENVIRONMENT=production

# Application
APP_NAME=Fast React Admin
APP_VERSION=1.0.0
DEBUG=False
IS_PRODUCTION=True

# Domain - CHANGE THIS TO YOUR DOMAIN
DOMAIN=admin.yourdomain.com
ADMIN_EMAIL=admin@yourdomain.com

# SSL - Set to true for production certificates
SSL_PRODUCTION=true

# Security - CHANGE ALL THESE IN PRODUCTION
SECRET_KEY=GENERATE-A-SECURE-RANDOM-STRING-HERE-64-CHARS
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database - CHANGE PASSWORD
DB_USER=admin
DB_PASSWORD=CHANGE-TO-STRONG-PASSWORD
DB_NAME=fastadmin
DB_HOST=postgres
DB_PORT=5432
DATABASE_URL=postgresql+asyncpg://admin:CHANGE-TO-STRONG-PASSWORD@postgres:5432/fastadmin

# Docker Prod PostgreSQL
POSTGRES_DB=fastadmin
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB_HOST=postgres
POSTGRES_DB_PORT=5432

# Redis - ADD PASSWORD IN PRODUCTION
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=CHANGE-TO-STRONG-PASSWORD
REDIS_URL=redis://:CHANGE-TO-STRONG-PASSWORD@redis:6379/0

# Celery
CELERY_BROKER_URL=redis://:CHANGE-TO-STRONG-PASSWORD@redis:6379/1
CELERY_RESULT_BACKEND=redis://:CHANGE-TO-STRONG-PASSWORD@redis:6379/2
TASK_CLEANUP_DAYS=30
TASK_REPORT_HOUR=9

# CORS (Restrict to your domain)
BACKEND_CORS_ORIGINS=http://backend.yourdomain.com,https://admin.yourdomain.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Flower - CHANGE PASSWORD
FLOWER_USER=admin
FLOWER_PASSWORD=CHANGE-TO-STRONG-PASSWORD

# Ports (Internal only in production)
BACKEND_PORT=8000
POSTGRES_PORT=5432
REDIS_PORT=6379
FLOWER_PORT=5555
ENV_PROD_EOF

echo "✅ Environment files created (.env.example, .env, .env.prod)"

#==============================================================================
# BACKEND STRUCTURE
#==============================================================================
echo "📦 Creating backend structure..."

mkdir -p backend/app/api/v1/endpoints
mkdir -p backend/app/api/v1/dependencies
mkdir -p backend/app/core
mkdir -p backend/app/models
mkdir -p backend/app/schemas
mkdir -p backend/app/services
mkdir -p backend/app/repositories
mkdir -p backend/app/middleware
mkdir -p backend/app/utils
mkdir -p backend/app/tasks
mkdir -p backend/tests
mkdir -p backend/alembic/versions

# Create __init__.py files
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/api/v1/__init__.py
touch backend/app/api/v1/endpoints/__init__.py
touch backend/app/api/v1/dependencies/__init__.py
touch backend/app/core/__init__.py
touch backend/app/models/__init__.py
touch backend/app/schemas/__init__.py
touch backend/app/services/__init__.py
touch backend/app/repositories/__init__.py
touch backend/app/middleware/__init__.py
touch backend/app/utils/__init__.py
touch backend/app/tasks/__init__.py
touch backend/tests/__init__.py

# Backend pyproject.toml
cat > backend/pyproject.toml << 'PYPROJECT_EOF'
[project]
name = "fast-react-admin-backend"
version = "0.1.0"
description = "Enterprise FastAPI Backend with Celery Beat"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "sqlalchemy>=2.0.25",
    "alembic>=1.13.0",
    "psycopg2-binary>=2.9.9",
    "asyncpg>=0.29.0",
    "redis>=5.0.1",
    "celery>=5.3.4",
    "flower>=2.0.1",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "bcrypt==4.1.2",
    "python-multipart>=0.0.6",
    "email-validator>=2.1.0",
    "python-slugify>=8.0.1",
    "httpx>=0.26.0",
    "inflect>=7.2.0",
    "numpy>=1.26.4",
    "orjson>=3.10.0",
    "pandas>=2.2.1",
    "tenacity>=8.2.3",
    "prometheus-client>=0.19.0",
    "python-dateutil>=2.8.2",
    "python-dotenv>=1.1.0",
]

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["app"]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.4",
    "pytest-asyncio>=0.23.3",
    "pytest-cov>=4.1.0",
    "black>=23.12.1",
    "ruff>=0.1.11",
    "mypy>=1.8.0",
]
PYPROJECT_EOF

# Core Config with Environment Variables
cat > backend/app/core/config.py << 'CONFIG_EOF'
"""Core Configuration Module - Environment Based"""
import os
import json
from functools import lru_cache
from typing import List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """Application Settings - Loaded from Environment"""
    
    # Environment
    ENVIRONMENT: str = "development"
    IS_PRODUCTION: bool = False
    
    # Application
    APP_NAME: str = "Fast React Admin"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # Domain
    DOMAIN: str = "localhost"
    ADMIN_EMAIL: str = "admin@gmail.com"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://admin:admin123@postgres:5432/fastadmin"
    
    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/2"
    CELERY_BEAT_SCHEDULE_FILENAME: str = "/tmp/celerybeat-schedule"
    
    # Tasks
    TASK_CLEANUP_DAYS: int = 30
    TASK_REPORT_HOUR: int = 9
    
    # CORS
    BACKEND_CORS_ORIGINS: Union[str, List[str]] = ["http://localhost:8001"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Flower
    FLOWER_PORT: int = 5555
    FLOWER_USER: str = "admin"
    FLOWER_PASSWORD: str = "admin123"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator('BACKEND_CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                return [v]
            except (json.JSONDecodeError, ValueError):
                # If not valid JSON, treat as comma-separated or single value
                if ',' in v:
                    return [origin.strip() for origin in v.split(',')]
                return [v]
        return v
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
CONFIG_EOF

# Database Module
cat > backend/app/core/database.py << 'DATABASE_EOF'
"""Database Module"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=40,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
DATABASE_EOF

# Security Module
cat > backend/app/core/security.py << 'SECURITY_EOF'
"""Security Module - Production Ready"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


class SecurityService:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash"""
        try:
            if len(plain_password.encode('utf-8')) > 72:
                plain_password = plain_password[:72]
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            print(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        try:
            if len(password.encode('utf-8')) > 72:
                password = password[:72]
            return pwd_context.hash(password)
        except Exception as e:
            print(f"Password hashing error: {e}")
            raise
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        except JWTError:
            return None


security_service = SecurityService()
SECURITY_EOF

echo "✅ Backend core created"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 1 Complete - Backend Core & Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Continue with Part 2 for Models, Services & API..."


# Base Model
cat >> backend/app/models/base.py << 'BASE_MODEL_EOF'
"""Base Model"""

from datetime import datetime
from pytz import UTC
from uuid import uuid4

import inflect
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship, column_property, backref
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import as_declarative, declared_attr

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR
# class TSVector(sa.types.TypeDecorator):
#     impl = TSVECTOR


@as_declarative()
class Base:
    id: Mapped[int]  = mapped_column(sa.Integer, primary_key=True, index=True, autoincrement=True)
    uid: Mapped[UUID] = mapped_column(UUID(as_uuid=True), index=True, default=uuid4, unique=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )
    __name__: str

    # Internal Method to generate table name
    def _generate_table_name(str):
        words = [[str[0]]]
        for c in str[1:]:
            if words[-1][-1].islower() and c.isupper():
                words.append(list(c))
            else:
                words[-1].append(c)
        return inflect.engine().plural(
            "_".join("".join(word) for word in words).lower()
        )

    # Generate __tablename__ automatically in plural form.
    #   
    # i.e 'myTable' model will generate table name 'my_tables'
    @declared_attr
    def __tablename__(cls) -> str:
        return cls._generate_table_name(cls.__name__)
BASE_MODEL_EOF


# User Model
cat >> backend/app/models/user.py << 'USER_MODEL_EOF'
"""User Model"""
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List
from app.models.base import Base


class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str] = mapped_column(String(255), nullable=True)
    last_name: Mapped[str] = mapped_column(String(255), nullable=True)
    phone: Mapped[str] = mapped_column(String(15), unique=True, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    is_email_verified: Mapped[bool] = mapped_column(Boolean(), default=False, nullable=True)
    is_phone_verified: Mapped[bool] = mapped_column(Boolean(), default=False, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean(), default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    
    groups: Mapped[List["Group"]] = relationship("Group", secondary="user_groups", back_populates="members")
    membership: Mapped[List["UserGroups"]] = relationship("UserGroups", back_populates="user", overlaps="groups")
USER_MODEL_EOF


# Groups Model
cat >> backend/app/models/groups.py << 'GROUPS_MODEL_EOF'
"""Groups Model"""
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List
import enum

from .base import Base


class Group(Base):
    __tablename__ = "groups"
    name: Mapped[str] = mapped_column(sa.String(50), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(sa.Text, default="")
    members: Mapped[List["User"]] = relationship("User", secondary="user_groups", back_populates="groups")


class RoleEnum(enum.Enum):
    admin = "admin"
    sub_admin = "sub_admin"
    manager = "manager"
    staff = "staff"
    member = "member"


class UserGroups(Base):
    __tablename__ = "user_groups"
    role: Mapped[str] = mapped_column(sa.Enum(RoleEnum), default=RoleEnum.member.value, nullable=False)
    add_members: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    view_members: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    remove_members: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    edit_members: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    edit_roles: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    buy_subscription: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    edit_subscription: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    view_subscription: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    edit_group: Mapped[bool] = mapped_column(sa.Boolean(), default=False, nullable=False)
    user_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("users.id", ondelete='CASCADE'), nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="membership", overlaps="groups,members")
    group_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("groups.id", ondelete='CASCADE'), nullable=False)
    group: Mapped["Group"] = relationship("Group", overlaps="groups,members")
GROUPS_MODEL_EOF

# Task History Model
cat >> backend/app/models/task_history.py << 'TASK_HISTORY_EOF'
"""Task History Model"""
from datetime import datetime
from sqlalchemy import String, DateTime, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship, column_property, backref

from app.models.base import Base 


class TaskHistory(Base):
    __tablename__ = "task_history"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    task_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=True)
    result: Mapped[str] = mapped_column(Text, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=True)
TASK_HISTORY_EOF

# User Schemas
cat >> backend/app/schemas/user.py << 'USER_SCHEMA_EOF'
"""User Schemas"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    is_active: bool = True


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=72)


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8, max_length=72)
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    id: int
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserResponse(UserInDB):
    pass


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
USER_SCHEMA_EOF

# Model's INIT
cat >> backend/app/models/__init__.py << 'MODELS_INIT_MODEL_EOF'
"""Model's INIT"""

from app.models.base import Base
from app.models.user import User
from app.models.groups import Group, UserGroups, RoleEnum

__all__ = ["Base", "User", "Group", "UserGroups", "RoleEnum"]
MODELS_INIT_MODEL_EOF


# User Repository
cat >> backend/app/repositories/user_repository.py << 'USER_REPO_EOF'
"""User Repository"""
from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User
from app.core.security import security_service


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        result = await self.session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()
    
    async def get_multi(self, skip: int = 0, limit: int = 100) -> List[User]:
        result = await self.session.execute(select(User).offset(skip).limit(limit))
        return list(result.scalars().all())
    
    async def get_inactive_users(self, days: int = 90) -> List[User]:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(User).where(
                User.last_login < cutoff_date,
                User.is_active == True
            )
        )
        return list(result.scalars().all())
    
    async def create(self, email: str, username: str, password: str, **kwargs) -> User:
        hashed_password = security_service.get_password_hash(password)
        user = User(email=email, username=username, hashed_password=hashed_password, **kwargs)
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user
    
    async def update(self, user: User, **kwargs) -> User:
        for key, value in kwargs.items():
            if value is not None:
                if key == "password":
                    user.hashed_password = security_service.get_password_hash(value)
                else:
                    setattr(user, key, value)
        await self.session.flush()
        await self.session.refresh(user)
        return user
    
    async def update_last_login(self, user: User) -> User:
        user.last_login = datetime.utcnow()
        await self.session.flush()
        await self.session.refresh(user)
        return user
    
    async def delete(self, user: User) -> None:
        await self.session.delete(user)
        await self.session.flush()
USER_REPO_EOF

# User Service
cat >> backend/app/services/user_service.py << 'USER_SERVICE_EOF'
"""User Service"""
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserUpdate
from app.models.user import User


class UserService:
    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)
    
    async def authenticate(self, email: str, password: str) -> Optional[User]:
        user = await self.repository.get_by_email(email)
        if not user:
            return None
        from app.core.security import security_service
        if not security_service.verify_password(password, user.hashed_password):
            return None
        await self.repository.update_last_login(user)
        return user
    
    async def create_user(self, user_in: UserCreate) -> User:
        return await self.repository.create(
            email=user_in.email,
            username=user_in.username,
            password=user_in.password,
            full_name=user_in.full_name,
            is_active=user_in.is_active,
        )
    
    async def get_user(self, user_id: int) -> Optional[User]:
        return await self.repository.get_by_id(user_id)
    
    async def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return await self.repository.get_multi(skip=skip, limit=limit)
    
    async def update_user(self, user: User, user_in: UserUpdate) -> User:
        update_data = user_in.model_dump(exclude_unset=True)
        return await self.repository.update(user, **update_data)
    
    async def delete_user(self, user: User) -> None:
        await self.repository.delete(user)
USER_SERVICE_EOF

# Auth Dependencies
cat >> backend/app/api/v1/dependencies/auth.py << 'AUTH_DEP_EOF'
"""Authentication Dependencies"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import security_service
from app.models.user import User
from app.repositories.user_repository import UserRepository

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_db)
) -> User:
    token = credentials.credentials
    payload = security_service.decode_token(token)
    
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    user_id: Optional[int] = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    repository = UserRepository(session)
    user = await repository.get_by_id(int(user_id))
    
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    
    return user


async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    return current_user
AUTH_DEP_EOF

# Auth Endpoints
cat >> backend/app/api/v1/endpoints/auth.py << 'AUTH_ENDPOINTS_EOF'
"""Authentication Endpoints"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import security_service
from app.schemas.user import UserCreate, UserResponse, Token
from app.services.user_service import UserService
from pydantic import BaseModel

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, session: AsyncSession = Depends(get_db)):
    service = UserService(session)
    user = await service.create_user(user_in)
    return user


@router.post("/login", response_model=Token)
async def login(login_data: LoginRequest, session: AsyncSession = Depends(get_db)):
    service = UserService(session)
    user = await service.authenticate(login_data.email, login_data.password)
    
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    
    access_token = security_service.create_access_token(data={"sub": str(user.id)})
    refresh_token = security_service.create_refresh_token(data={"sub": str(user.id)})
    
    return Token(access_token=access_token, refresh_token=refresh_token)
AUTH_ENDPOINTS_EOF

# Users Endpoints
cat >> backend/app/api/v1/endpoints/users.py << 'USERS_ENDPOINTS_EOF'
"""Users Endpoints"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.user import UserResponse, UserUpdate
from app.services.user_service import UserService
from app.api.v1.dependencies.auth import get_current_user, get_current_superuser
from app.models.user import User

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def read_user_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_me(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    service = UserService(session)
    return await service.update_user(current_user, user_in)


@router.get("/", response_model=List[UserResponse])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser),
    session: AsyncSession = Depends(get_db)
):
    service = UserService(session)
    return await service.get_users(skip=skip, limit=limit)


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    session: AsyncSession = Depends(get_db)
):
    service = UserService(session)
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user
USERS_ENDPOINTS_EOF

# API Router
cat >> backend/app/api/v1/__init__.py << 'API_ROUTER_EOF'
"""API V1 Router"""
from fastapi import APIRouter
from app.api.v1.endpoints import auth, users

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
API_ROUTER_EOF

# Rate Limit Middleware
cat >> backend/app/middleware/rate_limit.py << 'MIDDLEWARE_EOF'
"""Rate Limiting Middleware"""
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.cache = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        self.cache[client_ip] = [t for t in self.cache[client_ip] if now - t < self.period]
        
        if len(self.cache[client_ip]) >= self.calls:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")
        
        self.cache[client_ip].append(now)
        response = await call_next(request)
        return response
MIDDLEWARE_EOF

# Main Application
cat >> backend/app/main.py << 'MAIN_APP_EOF'
"""Main FastAPI Application"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.core.database import engine
from app.models.base import Base
from app.models.user import User
from app.models.groups import Group, UserGroups
from app.api.v1 import api_router
from app.middleware.rate_limit import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Security Middleware
if settings.IS_PRODUCTION:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[settings.DOMAIN, f"*.{settings.DOMAIN}"]
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate Limiting
app.add_middleware(RateLimitMiddleware, calls=settings.RATE_LIMIT_PER_MINUTE, period=60)

# Metrics (Internal only)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# API Routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }
MAIN_APP_EOF

# Celery Configuration
cat >> backend/app/tasks/celery_app.py << 'CELERY_APP_EOF'
"""Celery Application with Beat Schedule"""
from celery import Celery
from celery.schedules import crontab
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.tasks", "app.tasks.scheduled_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)

# Celery Beat Schedule
celery_app.conf.beat_schedule = {
    'health-check-every-5-minutes': {
        'task': 'app.tasks.scheduled_tasks.system_health_check',
        'schedule': 300.0,
    },
    'cleanup-inactive-users-daily': {
        'task': 'app.tasks.scheduled_tasks.cleanup_inactive_users',
        'schedule': crontab(hour=2, minute=0),
    },
    'generate-daily-report': {
        'task': 'app.tasks.scheduled_tasks.generate_daily_report',
        'schedule': crontab(hour=settings.TASK_REPORT_HOUR, minute=0),
    },
    'cleanup-task-history-weekly': {
        'task': 'app.tasks.scheduled_tasks.cleanup_task_history',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),
    },
    'database-maintenance-monthly': {
        'task': 'app.tasks.scheduled_tasks.database_maintenance',
        'schedule': crontab(hour=4, minute=0, day_of_month=1),
    },
    'cache-warming-hourly': {
        'task': 'app.tasks.scheduled_tasks.warm_cache',
        'schedule': crontab(minute=0),
    },
}
CELERY_APP_EOF

# Regular Tasks
cat >> backend/app/tasks/tasks.py << 'TASKS_EOF'
"""Regular Celery Tasks"""
from app.tasks.celery_app import celery_app
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def send_welcome_email(self, email: str):
    try:
        logger.info(f"Sending welcome email to {email}")
        return {"status": "sent", "email": email}
    except Exception as exc:
        logger.error(f"Failed to send email: {exc}")
        raise self.retry(exc=exc, countdown=60)
TASKS_EOF

# Scheduled Tasks
cat >> backend/app/tasks/scheduled_tasks.py << 'SCHEDULED_TASKS_EOF'
"""Scheduled Tasks for Celery Beat"""
from datetime import datetime, timedelta
import logging
from app.tasks.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name='app.tasks.scheduled_tasks.system_health_check')
def system_health_check():
    try:
        logger.info("Running system health check...")
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": "healthy",
            "redis": "healthy",
            "status": "ok"
        }
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise


@celery_app.task(name='app.tasks.scheduled_tasks.cleanup_inactive_users')
def cleanup_inactive_users():
    try:
        logger.info("Running inactive user cleanup...")
        return {"timestamp": datetime.utcnow().isoformat(), "users_processed": 0}
    except Exception as e:
        logger.error(f"User cleanup failed: {e}")
        raise


@celery_app.task(name='app.tasks.scheduled_tasks.generate_daily_report')
def generate_daily_report():
    try:
        logger.info("Generating daily report...")
        return {"timestamp": datetime.utcnow().isoformat(), "status": "generated"}
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


@celery_app.task(name='app.tasks.scheduled_tasks.cleanup_task_history')
def cleanup_task_history():
    try:
        logger.info("Cleaning up old task history...")
        return {"timestamp": datetime.utcnow().isoformat(), "records_deleted": 0}
    except Exception as e:
        logger.error(f"Task history cleanup failed: {e}")
        raise


@celery_app.task(name='app.tasks.scheduled_tasks.database_maintenance')
def database_maintenance():
    try:
        logger.info("Running database maintenance...")
        return {"timestamp": datetime.utcnow().isoformat(), "status": "completed"}
    except Exception as e:
        logger.error(f"Database maintenance failed: {e}")
        raise


@celery_app.task(name='app.tasks.scheduled_tasks.warm_cache')
def warm_cache():
    try:
        logger.info("Warming cache...")
        return {"timestamp": datetime.utcnow().isoformat(), "cache_entries": 0}
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        raise
SCHEDULED_TASKS_EOF

echo "✅ Models, Services, API & Celery created"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 2 Complete - Models, Services & Celery"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Continue with Part 3 for Frontend & Docker configuration..."


# Alembic Configuration
cat >> backend/alembic.ini << 'ALEMBIC_INI_EOF'
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = postgresql+asyncpg://admin:admin123@postgres:5432/fastadmin

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
ALEMBIC_INI_EOF

cat >> backend/alembic/env.py << 'ALEMBIC_ENV_EOF'
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
import asyncio
import os

# from app.core.database import Base
from app.models.base import Base
from app.models.user import User
from app.models.task_history import TaskHistory

config = context.config

# Override with environment variable if available
db_url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
ALEMBIC_ENV_EOF

cat >> backend/alembic/script.py.mako << 'ALEMBIC_MAKO_EOF'
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
ALEMBIC_MAKO_EOF

# Backend Dockerfile
cat >> backend/Dockerfile << 'BACKEND_DOCKERFILE_EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Copy dependencies
COPY pyproject.toml ./

# Copy application
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY startup.sh ./

RUN uv pip install --system --no-cache -r pyproject.toml
RUN chmod +x startup.sh

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
CMD ["./startup.sh"]
BACKEND_DOCKERFILE_EOF

cat >> backend/.dockerignore << 'BACKEND_DOCKERIGNORE_EOF'
__pycache__
*.pyc
.Python
venv/
.env
.env.*
.git
.DS_Store
*.db
celerybeat-schedule*
*.log
BACKEND_DOCKERIGNORE_EOF

echo "✅ Backend Docker & Alembic created"

#==============================================================================
# FRONTEND STRUCTURE
#==============================================================================
echo "🎨 Creating frontend with hot reload support..."

mkdir -p frontend/src/components/ui
mkdir -p frontend/src/components/layout
mkdir -p frontend/src/pages
mkdir -p frontend/src/services
mkdir -p frontend/src/hooks
mkdir -p frontend/src/utils
mkdir -p frontend/src/types
mkdir -p frontend/public

# Frontend package.json
cat >> frontend/package.json << 'PACKAGE_EOF'
{
  "name": "fast-react-admin-frontend",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 0.0.0.0",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.1",
    "@tanstack/react-query": "^5.17.19",
    "axios": "^1.6.5",
    "zod": "^3.22.4",
    "react-hook-form": "^7.49.3",
    "@hookform/resolvers": "^3.3.4",
    "lucide-react": "^0.303.0",
    "date-fns": "^3.2.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0",
    "tailwindcss-animate": "^1.0.7",
    "class-variance-authority": "^0.7.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.48",
    "@types/react-dom": "^18.2.18",
    "@vitejs/plugin-react-swc": "^3.5.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.33",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.11"
  }
}
PACKAGE_EOF

# TypeScript configs
cat >> frontend/tsconfig.json << 'TSCONFIG_EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
TSCONFIG_EOF

cat >> frontend/tsconfig.node.json << 'TSCONFIG_NODE_EOF'
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
TSCONFIG_NODE_EOF

# Vite config
cat >> frontend/vite.config.ts << 'VITE_EOF'
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  base: '/admin/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    watch: {
      usePolling: true,
    },
    proxy: {
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'esbuild',
  },
});
VITE_EOF

# Tailwind & PostCSS
cat >> frontend/tailwind.config.js << 'TAILWIND_EOF'
export default {
  darkMode: ["class"],
  content: ['./src/**/*.{ts,tsx}', './index.html'],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {"2xl": "1400px"},
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
TAILWIND_EOF

cat >> frontend/postcss.config.js << 'POSTCSS_EOF'
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
POSTCSS_EOF

# CSS
cat >> frontend/src/index.css << 'CSS_EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --primary: 221.2 83.2% 53.3%;
    --secondary: 210 40% 96.1%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96.1%;
    --destructive: 0 84.2% 60.2%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 221.2 83.2% 53.3%;
    --radius: 0.5rem;
  }
  
  body {
    @apply bg-background text-foreground;
  }
}
CSS_EOF

# HTML
cat >> frontend/index.html << 'HTML_EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Fast React Admin</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
HTML_EOF

# Types
cat >> frontend/src/types/index.ts << 'TYPES_EOF'
export interface User {
  id: number;
  email: string;
  username: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at: string;
  last_login?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  full_name?: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}
TYPES_EOF

# API Service
cat >> frontend/src/services/api.ts << 'API_EOF'
import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: {'Content-Type': 'application/json'},
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      window.location.href = '/admin/login';
    }
    return Promise.reject(error);
  }
);

export default api;
API_EOF

# Auth Service
cat >> frontend/src/services/auth.service.ts << 'AUTH_SERVICE_EOF'
import api from './api';
import { LoginRequest, RegisterRequest, TokenResponse, User } from '@/types';

class AuthService {
  async login(credentials: LoginRequest): Promise<TokenResponse> {
    const response = await api.post<TokenResponse>('/auth/login', credentials);
    const { access_token, refresh_token } = response.data;
    localStorage.setItem('access_token', access_token);
    localStorage.setItem('refresh_token', refresh_token);
    return response.data;
  }

  async register(userData: RegisterRequest): Promise<User> {
    const response = await api.post<User>('/auth/register', userData);
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await api.get<User>('/users/me');
    return response.data;
  }

  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  }
}

export const authService = new AuthService();
AUTH_SERVICE_EOF

# User Service
cat >> frontend/src/services/user.service.ts << 'USER_SERVICE_EOF'
import api from './api';
import { User } from '@/types';

class UserService {
  async getUsers(skip = 0, limit = 100): Promise<User[]> {
    const response = await api.get<User[]>('/users/', {params: { skip, limit }});
    return response.data;
  }

  async getUser(userId: number): Promise<User> {
    const response = await api.get<User>(`/users/${userId}`);
    return response.data;
  }
}

export const userService = new UserService();
USER_SERVICE_EOF

# Auth Hook
cat >> frontend/src/hooks/useAuth.ts << 'USEAUTH_EOF'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { authService } from '@/services/auth.service';
import { LoginRequest, RegisterRequest } from '@/types';
import { useNavigate } from 'react-router-dom';

export const useAuth = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const { data: user, isLoading } = useQuery({
    queryKey: ['currentUser'],
    queryFn: authService.getCurrentUser,
    enabled: authService.isAuthenticated(),
    retry: false,
  });

  const loginMutation = useMutation({
    mutationFn: (credentials: LoginRequest) => authService.login(credentials),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['currentUser'] });
      navigate('/dashboard');
    },
  });

  const registerMutation = useMutation({
    mutationFn: (userData: RegisterRequest) => authService.register(userData),
    onSuccess: () => navigate('/login'),
  });

  const logout = () => {
    authService.logout();
    queryClient.clear();
    navigate('/login');
  };

  return {
    user,
    isLoading,
    isAuthenticated: authService.isAuthenticated(),
    login: loginMutation.mutate,
    register: registerMutation.mutate,
    logout,
    loginError: loginMutation.error,
    registerError: registerMutation.error,
  };
};
USEAUTH_EOF

# Utils
cat >> frontend/src/utils/cn.ts << 'CN_EOF'
import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
CN_EOF

echo "✅ Frontend base structure created"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 3 Complete - Frontend & Alembic"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Continue with Part 4 for UI Components..."


# UI Components - Button
cat >> frontend/src/components/ui/Button.tsx << 'BUTTON_EOF'
import { ButtonHTMLAttributes, forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/utils/cn';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 rounded-md px-3',
        lg: 'h-11 rounded-md px-8',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {variant: 'default', size: 'default'},
  }
);

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return <button className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  }
);

Button.displayName = 'Button';

export { Button, buttonVariants };
BUTTON_EOF

# UI Components - Input
cat >> frontend/src/components/ui/Input.tsx << 'INPUT_EOF'
import { InputHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/utils/cn';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          'flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);

Input.displayName = 'Input';

export { Input };
INPUT_EOF

# UI Components - Card
cat >> frontend/src/components/ui/Card.tsx << 'CARD_EOF'
import { HTMLAttributes, forwardRef } from 'react';
import { cn } from '@/utils/cn';

const Card = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('rounded-lg border bg-card text-card-foreground shadow-sm', className)} {...props} />
  )
);
Card.displayName = 'Card';

const CardHeader = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('flex flex-col space-y-1.5 p-6', className)} {...props} />
  )
);
CardHeader.displayName = 'CardHeader';

const CardTitle = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3 ref={ref} className={cn('text-2xl font-semibold leading-none tracking-tight', className)} {...props} />
  )
);
CardTitle.displayName = 'CardTitle';

const CardContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('p-6 pt-0', className)} {...props} />
  )
);
CardContent.displayName = 'CardContent';

export { Card, CardHeader, CardTitle, CardContent };
CARD_EOF

# Pages - Login
cat >> frontend/src/pages/Login.tsx << 'LOGIN_EOF'
import { useState } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';

export const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login, loginError } = useAuth();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    login({ email, password });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-center">Fast React Admin</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="text-sm font-medium">Email</label>
              <Input 
                type="email" 
                value={email} 
                onChange={(e) => setEmail(e.target.value)} 
                placeholder="admin@gmail.com" 
                required 
              />
            </div>
            <div>
              <label className="text-sm font-medium">Password</label>
              <Input 
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                placeholder="••••••••" 
                required 
              />
            </div>
            {loginError && <p className="text-sm text-red-500">Invalid credentials</p>}
            <Button type="submit" className="w-full">Login</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};
LOGIN_EOF

# Pages - Dashboard
cat >> frontend/src/pages/Dashboard.tsx << 'DASH_EOF'
import { useAuth } from '@/hooks/useAuth';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';

export const Dashboard = () => {
  const { user } = useAuth();
  
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-gray-500 mt-1">Welcome back, {user?.username}!</p>
      </div>
      
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">System Status</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-green-600 font-semibold">All systems operational</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Active Users</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">1</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Environment</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">Development Mode</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
DASH_EOF

# Pages - Users
cat >> frontend/src/pages/Users.tsx << 'USERS_EOF'
import { useQuery } from '@tanstack/react-query';
import { userService } from '@/services/user.service';
import { Card } from '@/components/ui/Card';
import { format } from 'date-fns';

export const Users = () => {
  const { data: users, isLoading } = useQuery({
    queryKey: ['users'], 
    queryFn: () => userService.getUsers()
  });
  
  if (isLoading) return <div className="p-6">Loading...</div>;
  
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Users</h1>
        <p className="text-gray-500 mt-1">Manage system users</p>
      </div>
      
      <div className="grid gap-4">
        {users?.map((user) => (
          <Card key={user.id} className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-bold text-lg">{user.username}</h3>
                <p className="text-sm text-gray-500">{user.email}</p>
                <p className="text-xs text-gray-400 mt-1">
                  {user.full_name || 'No name provided'}
                </p>
              </div>
              <div className="text-right">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  user.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {user.is_active ? 'Active' : 'Inactive'}
                </span>
                {user.is_superuser && (
                  <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    Admin
                  </span>
                )}
                <p className="text-xs text-gray-400 mt-2">
                  Joined {format(new Date(user.created_at), 'MMM d, yyyy')}
                </p>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};
USERS_EOF

# Layout
cat >> frontend/src/components/layout/AdminLayout.tsx << 'LAYOUT_EOF'
import { Outlet, NavLink } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { Button } from '@/components/ui/Button';
import { LayoutDashboard, Users, LogOut } from 'lucide-react';

export const AdminLayout = () => {
  const { logout, user } = useAuth();
  
  const nav = [
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Users', href: '/users', icon: Users },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <aside className="fixed inset-y-0 left-0 w-64 bg-white border-r">
        <div className="flex flex-col h-full">
          <div className="p-4 border-b">
            <h2 className="text-xl font-bold text-primary">Fast React Admin</h2>
          </div>
          
          <nav className="flex-1 p-4 space-y-2">
            {nav.map((item) => (
              <NavLink 
                key={item.name} 
                to={item.href} 
                className={({ isActive }) => `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive 
                    ? 'bg-primary text-white' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.name}</span>
              </NavLink>
            ))}
          </nav>
          
          <div className="p-4 border-t">
            <div className="flex items-center justify-between">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium truncate">{user?.username}</p>
                <p className="text-xs text-gray-500 truncate">{user?.email}</p>
              </div>
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={logout}
                className="flex-shrink-0"
              >
                <LogOut className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </aside>
      
      <div className="ml-64">
        <main>
          <Outlet />
        </main>
      </div>
    </div>
  );
};
LAYOUT_EOF

# App
cat >> frontend/src/App.tsx << 'APP_EOF'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AdminLayout } from '@/components/layout/AdminLayout';
import { Login } from '@/pages/Login';
import { Dashboard } from '@/pages/Dashboard';
import { Users } from '@/pages/Users';
import { useAuth } from '@/hooks/useAuth';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading } = useAuth();
  
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }
  
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
};

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<ProtectedRoute><AdminLayout /></ProtectedRoute>}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="users" element={<Users />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
APP_EOF

# Main
cat >> frontend/src/main.tsx << 'MAIN_EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
MAIN_EOF

# Frontend Dockerfile - Development
cat >> frontend/Dockerfile.dev << 'FRONTEND_DOCKERFILE_DEV_EOF'
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./

# Install dependencies
RUN if [ -f yarn.lock ]; then yarn install --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then pnpm install --frozen-lockfile; \
    else npm install; fi

# Copy source
COPY . .

EXPOSE 3000

# Run dev server with hot reload
CMD ["npm", "run", "dev"]
FRONTEND_DOCKERFILE_DEV_EOF

# Frontend Dockerfile - Production
cat >> frontend/Dockerfile << 'FRONTEND_DOCKERFILE_PROD_EOF'
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./

RUN if [ -f yarn.lock ]; then yarn install --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then pnpm install --frozen-lockfile; \
    else npm install; fi

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
FRONTEND_DOCKERFILE_PROD_EOF

# Frontend nginx config
cat >> frontend/nginx.conf << 'FRONTEND_NGINX_EOF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location = /index.html {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
        add_header Expires "0";
    }
    
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
FRONTEND_NGINX_EOF

cat >> frontend/.dockerignore << 'FRONTEND_DOCKERIGNORE_EOF'
node_modules
.git
.DS_Store
dist
*.log
.env
.env.*
FRONTEND_DOCKERIGNORE_EOF

echo "✅ Frontend UI, Pages & Dockerfiles created"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 4 Complete - Frontend Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Continue with Part 5 for Docker Compose & Nginx with SSL..."


#==============================================================================
# NGINX CONFIGURATION
#==============================================================================
echo "🌐 Creating Nginx configurations..."

mkdir -p nginx/ssl
mkdir -p nginx/conf.d

# Development Nginx (Hot Reload Support)
cat >> nginx/nginx.dev.conf << 'NGINX_DEV_EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name localhost;

        client_max_body_size 100M;

        # API routes (internal only in production)
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Backend docs (development only)
        location ~ ^/(docs|redoc|openapi.json|health|metrics) {
            proxy_pass http://backend;
            proxy_set_header Host $host;
        }

        # Admin panel with SPA routing
        location /admin/ {
            proxy_pass http://frontend/admin/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # WebSocket for HMR
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Admin static assets
        location /admin/assets/ {
            proxy_pass http://frontend/admin/assets/;
            proxy_set_header Host $host;
        }

        # Vite HMR WebSocket
        location /admin/@vite/ {
            proxy_pass http://frontend/@vite/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        # Vite HMR
        location /@vite/client {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location = /admin {
            return 301 /admin/;
        }

        location = / {
            return 301 /admin/;
        }

        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
NGINX_DEV_EOF

# Production Nginx with SSL
cat >> nginx/nginx.prod.conf << 'NGINX_PROD_EOF'
events {
    worker_connections 2048;
}

http {
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=addr:10m;
    
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:80;
    }

    # HTTP - Redirect to HTTPS
    server {
        listen 80;
        server_name ${DOMAIN};

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 301 https://$server_name$request_uri;
        }
    }

    # HTTPS - Main Server
    server {
        listen 443 ssl http2;
        server_name ${DOMAIN};

        # SSL Configuration
        ssl_certificate /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
        
        # SSL Settings
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # HSTS
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Security Headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;

        client_max_body_size 100M;
        
        # Rate limiting
        limit_req zone=general burst=20 nodelay;
        limit_conn addr 10;

        # Internal API (NOT publicly accessible)
        location /api/ {
            # Block external access to API
            # Only allow through admin interface
            deny all;
        }

        # Block docs in production
        location ~ ^/(docs|redoc|openapi.json) {
            deny all;
        }

        # Health check (internal monitoring only)
        location = /health {
            proxy_pass http://backend;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }

        # Admin Panel (PUBLIC - Frontend Only)
        location /admin {
            rewrite ^/admin/(.*)$ /$1 break;
            rewrite ^/admin$ / break;
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Frontend calls API internally through Docker network
            # API responses proxied through frontend
        }

        location = / {
            return 301 https://$server_name/admin;
        }

        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
NGINX_PROD_EOF

# Nginx Dockerfile
cat >> nginx/Dockerfile << 'NGINX_DOCKERFILE_EOF'
FROM nginx:alpine

# Install envsubst for environment variable substitution
RUN apk add --no-cache gettext

# Copy nginx configs
COPY nginx.dev.conf /etc/nginx/templates/nginx.dev.conf
COPY nginx.prod.conf /etc/nginx/templates/nginx.prod.conf

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
NGINX_DOCKERFILE_EOF

# Nginx entrypoint
cat >> nginx/docker-entrypoint.sh << 'NGINX_ENTRYPOINT_EOF'
#!/bin/sh
set -e

# Select config based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Using production nginx configuration"
    envsubst '${DOMAIN}' < /etc/nginx/templates/nginx.prod.conf > /etc/nginx/nginx.conf
else
    echo "Using development nginx configuration"
    cp /etc/nginx/templates/nginx.dev.conf /etc/nginx/nginx.conf
fi

# Test configuration
nginx -t

# Execute CMD
exec "$@"
NGINX_ENTRYPOINT_EOF

echo "✅ Nginx configurations created"

#==============================================================================
# DOCKER COMPOSE - DEVELOPMENT
#==============================================================================
echo "🐳 Creating Docker Compose files..."

cat >> docker-compose.dev.yml << 'COMPOSE_DEV_EOF'
# version: '3.8'
services:
  postgres:
    image: postgres:16-alpine
    container_name: ${PROJECT_NAME:-fastadmin}_postgres
    environment:
      POSTGRES_USER: ${DB_USER:-admin}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-admin123}
      POSTGRES_DB: ${DB_NAME:-fastadmin}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-admin} -d ${DB_NAME:-fastadmin}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - app_network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: ${PROJECT_NAME:-fastadmin}_redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "${REDIS_PORT:-6379}:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - app_network
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_backend
    environment:
      ENVIRONMENT: ${ENVIRONMENT:-development}
      DATABASE_URL: ${DATABASE_URL:-postgresql+asyncpg://admin:admin123@postgres:5432/fastadmin}
      REDIS_URL: ${REDIS_URL:-redis://redis:6379/0}
      CELERY_BROKER_URL: ${CELERY_BROKER_URL:-redis://redis:6379/1}
      CELERY_RESULT_BACKEND: ${CELERY_RESULT_BACKEND:-redis://redis:6379/2}
      SECRET_KEY: ${SECRET_KEY:-dev-secret-key}
      DEBUG: ${DEBUG:-true}
      BACKEND_CORS_ORIGINS: ${BACKEND_CORS_ORIGINS:-["http://localhost:8001","http://localhost:3000"]}
    volumes:
      - ./backend/app:/app/app:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "${BACKEND_PORT:-8000}:8000"
    networks:
      - app_network
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    restart: unless-stopped

  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_celery_worker
    command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=2
    environment:
      ENVIRONMENT: ${ENVIRONMENT:-development}
      DATABASE_URL: ${DATABASE_URL:-postgresql+asyncpg://admin:admin123@postgres:5432/fastadmin}
      REDIS_URL: ${REDIS_URL:-redis://redis:6379/0}
      CELERY_BROKER_URL: ${CELERY_BROKER_URL:-redis://redis:6379/1}
      CELERY_RESULT_BACKEND: ${CELERY_RESULT_BACKEND:-redis://redis:6379/2}
    volumes:
      - ./backend/app:/app/app:ro
    depends_on:
      - redis
      - postgres
      - backend
    networks:
      - app_network
    restart: unless-stopped

  celery_beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_celery_beat
    command: celery -A app.tasks.celery_app beat --loglevel=info
    environment:
      ENVIRONMENT: ${ENVIRONMENT:-development}
      DATABASE_URL: ${DATABASE_URL:-postgresql+asyncpg://admin:admin123@postgres:5432/fastadmin}
      REDIS_URL: ${REDIS_URL:-redis://redis:6379/0}
      CELERY_BROKER_URL: ${CELERY_BROKER_URL:-redis://redis:6379/1}
      CELERY_RESULT_BACKEND: ${CELERY_RESULT_BACKEND:-redis://redis:6379/2}
    volumes:
      - ./backend/app:/app/app:ro
      - celery_beat_data:/tmp
    depends_on:
      - redis
      - postgres
      - backend
    networks:
      - app_network
    restart: unless-stopped

  flower:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_flower
    command: celery -A app.tasks.celery_app flower --port=5555 --basic_auth=${FLOWER_USER:-admin}:${FLOWER_PASSWORD:-admin123}
    environment:
      CELERY_BROKER_URL: ${CELERY_BROKER_URL:-redis://redis:6379/1}
      CELERY_RESULT_BACKEND: ${CELERY_RESULT_BACKEND:-redis://redis:6379/2}
    ports:
      - "${FLOWER_PORT:-5555}:5555"
    depends_on:
      - redis
      - celery_worker
    networks:
      - app_network
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    container_name: ${PROJECT_NAME:-fastadmin}_frontend_dev
    volumes:
      - ./frontend:/app:cached
      - /app/node_modules
    ports:
      - "${FRONTEND_DEV_PORT:-3000}:3000"
    environment:
      - CHOKIDAR_USEPOLLING=true
      - WATCHPACK_POLLING=true
    depends_on:
      - backend
    networks:
      - app_network
    stdin_open: true
    tty: true
    restart: unless-stopped

  nginx:
    build:
      context: ./nginx
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_nginx
    environment:
      ENVIRONMENT: ${ENVIRONMENT:-development}
      DOMAIN: ${DOMAIN:-localhost}
    ports:
      - "${NGINX_HTTP_PORT:-8001}:80"
    depends_on:
      - backend
      - frontend
    networks:
      - app_network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  celery_beat_data:

networks:
  app_network:
    driver: bridge
COMPOSE_DEV_EOF

#==============================================================================
# DOCKER COMPOSE - PRODUCTION with SSL
#==============================================================================

cat >> docker-compose.prod.yml << 'COMPOSE_PROD_EOF'
# version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: ${PROJECT_NAME:-fastadmin}_postgres
    env_file: .env.prod
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - app_network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: ${PROJECT_NAME:-fastadmin}_redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - app_network
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_backend
    env_file: .env.prod
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - app_network
    restart: unless-stopped

  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_celery_worker
    command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=4
    env_file: .env.prod
    depends_on:
      - redis
      - postgres
      - backend
    networks:
      - app_network
    restart: unless-stopped

  celery_beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_celery_beat
    command: celery -A app.tasks.celery_app beat --loglevel=info
    env_file: .env.prod
    volumes:
      - celery_beat_data:/tmp
    depends_on:
      - redis
      - postgres
      - backend
    networks:
      - app_network
    restart: unless-stopped

  flower:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_flower
    command: celery -A app.tasks.celery_app flower --port=5555 --basic_auth=${FLOWER_USER}:${FLOWER_PASSWORD}
    env_file: .env.prod
    depends_on:
      - redis
      - celery_worker
    networks:
      - app_network
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_frontend
    depends_on:
      - backend
    networks:
      - app_network
    restart: unless-stopped

  nginx:
    build:
      context: ./nginx
      dockerfile: Dockerfile
    container_name: ${PROJECT_NAME:-fastadmin}_nginx
    env_file: .env.prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/ssl:/etc/letsencrypt:ro
      - certbot_www:/var/www/certbot:ro
    depends_on:
      - backend
      - frontend
    networks:
      - app_network
    restart: unless-stopped

  certbot:
    image: certbot/certbot:latest
    container_name: ${PROJECT_NAME:-fastadmin}_certbot
    volumes:
      - ./nginx/ssl:/etc/letsencrypt
      - certbot_www:/var/www/certbot
    entrypoint: "/bin/sh -c 'trap exit TERM; while :; do certbot renew; sleep 12h & wait $${!}; done;'"
    networks:
      - app_network

volumes:
  postgres_data:
  redis_data:
  celery_beat_data:
  certbot_www:

networks:
  app_network:
    driver: bridge
COMPOSE_PROD_EOF

echo "✅ Docker Compose files created"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 5 Complete - Docker Compose & Nginx with SSL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Continue with Part 6 for Management Scripts & SSL setup..."


#==============================================================================
# SSL SETUP SCRIPT
#==============================================================================
echo "🔐 Creating SSL setup script..."

cat >> setup-ssl.sh << 'SSL_SETUP_EOF'
#!/bin/bash
set -e

# Load environment variables
if [ -f .env.prod ]; then
    export $(cat .env.prod | grep -v '^#' | xargs)
fi

DOMAIN=${DOMAIN:-admin.example.com}
EMAIL=${ADMIN_EMAIL:-admin@gmail.com}
STAGING=${SSL_PRODUCTION:-false}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 SSL Certificate Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo "Mode: $([ "$STAGING" = "true" ] && echo "PRODUCTION" || echo "STAGING/TEST")"
echo ""

# Check if domain is reachable
echo "Checking if domain is accessible..."
if ! ping -c 1 "$DOMAIN" &> /dev/null; then
    echo "❌ Domain $DOMAIN is not reachable!"
    echo "Please ensure:"
    echo "  1. Domain DNS is configured"
    echo "  2. Domain points to this server's IP"
    echo "  3. Ports 80 and 443 are open"
    exit 1
fi

echo "✅ Domain is reachable"
echo ""

# Create SSL directory
mkdir -p nginx/ssl
mkdir -p nginx/certbot/www

# Initial certificate generation
echo "Obtaining SSL certificate..."

if [ "$STAGING" = "false" ]; then
    echo "⚠️  Using Let's Encrypt STAGING server (for testing)"
    STAGING_FLAG="--staging"
else
    echo "✅ Using Let's Encrypt PRODUCTION server"
    STAGING_FLAG=""
fi

docker compose -f docker-compose.prod.yml run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    $STAGING_FLAG \
    -d "$DOMAIN"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SSL certificate obtained successfully!"
    echo ""
    echo "Certificate location: nginx/ssl/live/$DOMAIN/"
    echo ""
    echo "🔄 Certificate will auto-renew every 12 hours via certbot container"
else
    echo ""
    echo "❌ Failed to obtain SSL certificate"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure domain points to this server"
    echo "  2. Check firewall allows ports 80/443"
    echo "  3. Try staging mode first: SSL_PRODUCTION=false"
    exit 1
fi
SSL_SETUP_EOF

chmod +x setup-ssl.sh

#==============================================================================
# INITIALIZATION SCRIPT
#==============================================================================
echo "🎯 Creating initialization script..."

cat >> init.sh << 'INIT_EOF'
#!/bin/bash
set -e

# Determine environment
if [ -f .env.prod ] && docker compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    COMPOSE_FILE="docker-compose.prod.yml"
    ENV_FILE=".env.prod"
    echo "🔧 Production environment detected"
elif [ -f .env ] && docker compose -f docker-compose.dev.yml ps | grep -q "Up"; then
    COMPOSE_FILE="docker-compose.dev.yml"
    ENV_FILE=".env"
    echo "🔧 Development environment detected"
else
    echo "❌ No running environment detected. Please start services first."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Initializing Application"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 15

echo "🗃  Running database migrations..."
docker compose -f "$COMPOSE_FILE" exec -T backend alembic upgrade head

# Create superuser (only if doesn't exist)
echo "👤 Checking superuser..."
docker compose -f "$COMPOSE_FILE" exec -T backend python -c "

import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.core.security import security_service

async def ensure_superuser():
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(select(User).where(User.email == 'admin@gmail.com'))
            if result.scalar_one_or_none():
                print('ℹ️  Superuser already exists')
                return
            
            password = 'admin123'
            
            user = User(
                email='admin@gmail.com',
                username='admin',
                hashed_password=security_service.get_password_hash(password),
                full_name='System Administrator',
                is_active=True,
                is_superuser=True
            )
            session.add(user)
            await session.commit()
            print('✅ Superuser created successfully!')
            print('   Email: admin@gmail.com')
            print('   Password: admin123')
            print('')
            print('⚠️  IMPORTANT: Change the default password immediately!')
        except Exception as e:
            print(f'❌ Error creating superuser: {e}')
            await session.rollback()
            raise

asyncio.run(ensure_superuser())
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Initialization Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Load environment
export $(cat "$ENV_FILE" | grep -v '^#' | xargs)

if [ "$COMPOSE_FILE" = "docker-compose.dev.yml" ]; then
    echo "🌐 Development URLs:"
    echo "   Admin:  http://localhost:${NGINX_HTTP_PORT:-8001}/admin"
    echo "   API:    http://localhost:${NGINX_HTTP_PORT:-8001}/docs"
    echo "   React:  http://localhost:${FRONTEND_DEV_PORT:-3000}"
    echo "   Flower: http://localhost:${FLOWER_PORT:-5555}"
    echo ""
    echo "🔥 Hot Reload: Edit files and see instant changes!"
else
    echo "🌐 Production URLs:"
    echo "   Admin:  https://${DOMAIN}/admin"
    echo "   Flower: Internal only (Docker network)"
    echo ""
    echo "🔒 Security: API is NOT publicly accessible"
fi

echo ""
echo "👤 Login Credentials:"
echo "   Email: admin@gmail.com"
echo "   Password: admin123"
echo ""
INIT_EOF

chmod +x init.sh

#==============================================================================
# STARTUP CLI
#==============================================================================
echo "🛠  Creating Startup CLI..."

cat >> backend/startup.sh << 'MANAGE_EOF'
#!/bin/bash
set -e

echo "⏳ Waiting for database..."
until pg_isready -h postgres -U admin; do
  sleep 2
done

echo "🗃 Running migrations..."
alembic upgrade head

echo "🚀 Starting server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
MANAGE_EOF

#==============================================================================
# MANAGEMENT CLI
#==============================================================================
echo "🛠  Creating management CLI..."

cat >> manage.sh << 'MANAGE_EOF'
#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_header() { echo -e "${BLUE}═══ $1 ═══${NC}"; }
print_mode() { echo -e "${CYAN}🔧 $1${NC}"; }

# Detect environment
detect_env() {
    if docker compose -f docker-compose.prod.yml ps 2>/dev/null | grep -q "Up"; then
        echo "prod"
    elif docker compose -f docker-compose.dev.yml ps 2>/dev/null | grep -q "Up"; then
        echo "dev"
    else
        echo "none"
    fi
}

get_compose_file() {
    local env=$1
    if [ "$env" = "prod" ]; then
        echo "docker-compose.prod.yml"
    else
        echo "docker-compose.dev.yml"
    fi
}

get_env_file() {
    local env=$1
    if [ "$env" = "prod" ]; then
        echo ".env.prod"
    else
        echo ".env"
    fi
}

show_help() {
    echo ""
    print_header "Fast React Admin - Management CLI"
    echo ""
    echo "Usage: ./manage.sh <command> [mode]"
    echo ""
    echo "Commands:"
    echo "  dev                Daily dev workflow (start + migrate) (hot reload)"
    echo "  prod               Start PRODUCTION mode (SSL + security)"
    echo "  build              Build Docker images [dev|prod]"
    echo "  start              Start services [dev|prod]"
    echo "  stop               Stop all services"
    echo "  restart            Restart services"
    echo "  logs               View logs [-f to follow] [service]"
    echo "  status             Show service status"
    echo "  shell              Backend shell"
    echo "  frontend           Frontend shell"
    echo "  install backend    Install backend package"
    echo "  install frontend   Install frontend package"
    echo "  db-shell           Database shell"
    echo "  migrate            Run database migrations"
    echo "  tasks              View Celery scheduled tasks"
    echo "  flower             Show Flower monitoring info"
    echo "  ssl-setup          Setup SSL certificates (production)"
    echo "  ssl-renew          Manually renew SSL certificates"
    echo "  backup             Backup database"
    echo "  restore            Restore database from backup"
    echo "  clean              Remove containers and volumes"
    echo "  help               Show this help"
    echo ""
    echo "Examples:"
    echo "  ./manage.sh dev              # Start development"
    echo "  ./manage.sh prod             # Start production"
    echo "  ./manage.sh logs dev -f      # Follow dev logs"
    echo "  ./manage.sh ssl-setup        # Setup SSL"
    echo ""
}

case "$1" in
    dev)
        print_mode "DEVELOPMENT MODE"
        print_info "Starting with hot reload..."
        
        if [ ! -f .env ]; then
            print_error ".env not found!"
            print_info "Creating .env from .env.example..."
            if [ -f .env.example ]; then
                cp .env.example .env
                print_success ".env created! Please review and edit if needed."
            else
                print_error ".env.example not found either!"
                exit 1
            fi
        fi

        # Export environment variables for Docker Compose
        export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
        
        docker compose -f docker-compose.dev.yml up -d
        sleep 5

        print_info "Running migrations..."
        docker compose -f docker-compose.dev.yml exec backend alembic upgrade head || true

        docker compose -f docker-compose.dev.yml ps
        
        print_success "Development started!"
        echo ""
        print_info "🔥 Hot reload enabled - edit and see instant changes!"
        echo ""
        echo "🌐 dev url : http://localhost:8001/admin"

        # Check if init needed
        if ! docker compose -f docker-compose.dev.yml exec -T backend python -c "import sys; sys.exit(0)" 2>/dev/null; then
            print_info "Backend not ready yet, waiting..."
            sleep 10
        fi

        ./init.sh
        ;;
    
    prod)
        print_mode "PRODUCTION MODE"
        print_info "Starting production environment..."
        
        if [ ! -f .env.prod ]; then
            print_error ".env.prod not found!"
            print_info "Creating .env.prod from .env.example..."
            if [ -f .env.example ]; then
                cp .env.example .env.prod
                print_success ".env.prod created!"
                print_error "IMPORTANT: Edit .env.prod with production settings!"
                print_info "Required changes:"
                echo "  - DOMAIN=your-domain.com"
                echo "  - Change ALL passwords"
                echo "  - SSL_PRODUCTION=true"
                exit 1
            else
                print_error ".env.example not found!"
                exit 1
            fi
        fi

        # Export environment variables
        export $(cat .env.prod | grep -v '^#' | grep -v '^$' | xargs)
        
        # Check SSL
        if [ ! -d "nginx/ssl/live/$DOMAIN" ]; then
            print_error "SSL certificates not found for domain: $DOMAIN"
            print_info "Run: ./manage.sh ssl-setup"
            exit 1
        fi
        
        docker compose -f docker-compose.prod.yml up -d
        sleep 5
        docker compose -f docker-compose.prod.yml ps
        
        print_success "Production started!"
        echo ""
        ./init.sh
        ;;
    
    build)
        MODE="${2:-dev}"
        COMPOSE_FILE=$(get_compose_file "$MODE")
        ENV_FILE=$(get_env_file "$MODE")
        
        if [ ! -f "$ENV_FILE" ]; then
            print_error "$ENV_FILE not found!"
            exit 1
        fi
        
        print_mode "Building for $MODE"
        
        # Export environment variables
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
        
        docker compose -f "$COMPOSE_FILE" build
        print_success "Build complete!"
        ;;
    
    start)
        MODE="${2:-dev}"
        COMPOSE_FILE=$(get_compose_file "$MODE")
        ENV_FILE=$(get_env_file "$MODE")
        
        if [ ! -f "$ENV_FILE" ]; then
            print_error "$ENV_FILE not found!"
            exit 1
        fi
        
        print_info "Starting services..."
        
        # Export environment variables
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
        
        docker compose -f "$COMPOSE_FILE" up -d
        sleep 5
        docker compose -f "$COMPOSE_FILE" ps
        print_success "Services started!"
        ;;
    
    stop)
        print_info "Stopping all services..."
        docker compose -f docker-compose.dev.yml down 2>/dev/null || true
        docker compose -f docker-compose.prod.yml down 2>/dev/null || true
        print_success "Services stopped!"
        ;;
    
    restart)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        
        # Export environment variables
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)

        docker compose -f "$COMPOSE_FILE" restart
        print_success "Services restarted!"
        ;;
    
    logs)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        
        # Export environment variables
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
        
        if [ "$2" = "-f" ]; then
            docker compose -f "$COMPOSE_FILE" logs -f ${3:-}
        else
            docker compose -f "$COMPOSE_FILE" logs --tail=100 ${2:-}
        fi
        ;;
    
    status)
        CURRENT_ENV=$(detect_env)
        
        echo ""
        print_header "Service Status"
        echo ""
        
        if [ "$CURRENT_ENV" = "dev" ]; then
            print_mode "DEVELOPMENT"
            export $(cat .env | grep -v '^#' | grep -v '^$' | xargs) 2>/dev/null || true
            docker compose -f docker-compose.dev.yml ps
        elif [ "$CURRENT_ENV" = "prod" ]; then
            print_mode "PRODUCTION"
            export $(cat .env.prod | grep -v '^#' | grep -v '^$' | xargs) 2>/dev/null || true
            docker compose -f docker-compose.prod.yml ps
        else
            print_info "No services running"
        fi
        ;;
    
    shell)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")

        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)

        docker compose -f "$COMPOSE_FILE" exec backend /bin/bash
        ;;
    
    frontend)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")

        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)

        docker compose -f "$COMPOSE_FILE" exec frontend /bin/sh
        ;;
    
    install)
        if [ "$2" == "backend" ]; then
          if [ -z "$3" ]; then
            echo "Usage: ./manage.sh install backend <package-name> [version]"
            exit 1
          fi
          PACKAGE="$3"
          VERSION="${4:-latest}"
          
          print_info "Installing backend package: $PACKAGE"
          docker compose exec backend uv pip install "$PACKAGE" --break-system-packages
          
          # Add to pyproject.toml
          print_info "Updating pyproject.toml..."
          if [ "$VERSION" == "latest" ]; then
            ENTRY="\"$PACKAGE\","
          else
            ENTRY="\"$PACKAGE>=$VERSION\","
          fi
          
          # Insert into dependencies array (before closing bracket)
          sed -i "/dependencies = \[/a\\    $ENTRY" backend/pyproject.toml
          
          print_success "Package installed and pyproject.toml updated!"
          
        elif [ "$2" == "frontend" ]; then
          if [ -z "$3" ]; then
            echo "Usage: ./manage.sh install frontend <package-name>"
            exit 1
          fi
          
          print_info "Installing frontend package: $3"
          docker compose exec frontend yarn add "$3"
          
          # Copy updated files from container
          print_info "Syncing package.json and yarn.lock..."
          docker compose cp frontend:/app/package.json frontend/package.json
          docker compose cp frontend:/app/yarn.lock frontend/yarn.lock
          
          print_success "Package installed and files synced!"
          
        else
          echo "Usage: ./manage.sh install [backend|frontend] <package-name> [version]"
          exit 1
        fi
        ;;
    
    db-shell)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
        
        docker compose -f "$COMPOSE_FILE" exec postgres psql -U "$DB_USER" -d "$DB_NAME"
        ;;
    
    migrate)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)

        print_info "Running migrations..."
        docker compose -f "$COMPOSE_FILE" exec backend alembic upgrade head
        print_success "Migrations complete!"
        ;;
    
    tasks)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
        
        print_info "Celery Beat Schedule:"
        docker compose -f "$COMPOSE_FILE" exec celery_beat celery -A app.tasks.celery_app inspect scheduled
        ;;
    
    flower)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            CURRENT_ENV="dev"
        fi
        
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        export $(cat "$ENV_FILE" | grep FLOWER_PORT | xargs) 2>/dev/null || export FLOWER_PORT=5555
        
        echo ""
        print_info "Flower Monitoring:"
        if [ "$CURRENT_ENV" = "dev" ]; then
            echo "   URL: http://localhost:${FLOWER_PORT:-5555}"
        else
            echo "   URL: Internal only (Docker network)"
        fi
        echo "   Login: Check $ENV_FILE for credentials"
        ;;
    
    ssl-setup)
        if [ ! -f .env.prod ]; then
            print_error ".env.prod not found!"
            exit 1
        fi
        
        print_info "Setting up SSL certificates..."
        export $(cat .env.prod | grep -v '^#' | grep -v '^$' | xargs)
        ./setup-ssl.sh
        ;;
    
    ssl-renew)
        if [ ! -f .env.prod ]; then
            print_error ".env.prod not found!"
            exit 1
        fi
        
        export $(cat .env.prod | grep -v '^#' | grep -v '^$' | xargs)
        
        print_info "Renewing SSL certificates..."
        docker compose -f docker-compose.prod.yml exec certbot certbot renew
        docker compose -f docker-compose.prod.yml restart nginx
        print_success "SSL certificates renewed!"
        ;;
    
    backup)
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
        
        BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
        print_info "Creating backup: $BACKUP_FILE"
        
        docker compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"
        print_success "Backup created: $BACKUP_FILE"
        ;;
    
    restore)
        BACKUP_FILE="$2"
        if [ -z "$BACKUP_FILE" ]; then
            print_error "Usage: ./manage.sh restore <backup_file>"
            exit 1
        fi
        
        if [ ! -f "$BACKUP_FILE" ]; then
            print_error "Backup file not found: $BACKUP_FILE"
            exit 1
        fi
        
        CURRENT_ENV=$(detect_env)
        if [ "$CURRENT_ENV" = "none" ]; then
            print_error "No services running"
            exit 1
        fi
        
        COMPOSE_FILE=$(get_compose_file "$CURRENT_ENV")
        ENV_FILE=$(get_env_file "$CURRENT_ENV")
        export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
        
        print_info "Restoring from: $BACKUP_FILE"
        docker compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$DB_USER" "$DB_NAME" < "$BACKUP_FILE"
        print_success "Restore complete!"
        ;;
    
    clean)
        read -p "Remove all containers and volumes? (y/N) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker compose -f docker-compose.dev.yml down -v 2>/dev/null || true
            docker compose -f docker-compose.prod.yml down -v 2>/dev/null || true
            print_success "Cleanup complete!"
        fi
        ;;
    
    help|*)
        show_help
        ;;
esac
MANAGE_EOF

chmod +x manage.sh

echo "✅ Management scripts created"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Part 6 Complete - All Scripts Created!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Creating final README and .gitignore..."


# ============================================================================
# ENTERPRISE FAST REACT ADMIN GENERATOR - COMPLETE
# ============================================================================
# Production-Ready Full Stack Admin Boilerplate
# 
# Features:
# ✅ Hot Reload Development Mode
# ✅ Production with SSL/TLS Auto-Renewal
# ✅ Domain-based Security (Frontend Only Public)
# ✅ Environment-based Configuration (.env)
# ✅ Celery Beat + Flower Monitoring
# ✅ Security Hardening
# ✅ Plug & Play Deployment
#
# Usage: ./enterprise_admin_generator.sh [project_name]
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FEATURES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📦 Project: $PROJECT_NAME"
echo "✨ Full Stack: FastAPI + React + PostgreSQL + Redis + Celery"
echo "🔐 Security: SSL + Domain Mapping + API Protection"
echo ""

# Note: In production, you would cat all the parts here or source them
# For now, this is a template showing the structure

# This script would include all content from parts 1-6

# Instead of repeating all code, create comprehensive README

cat > README.md << 'README_EOF'
# Fast React Admin - Enterprise Edition

## 🚀 Complete Production-Ready Full Stack Admin

Enterprise-grade admin boilerplate with hot reload development and production SSL deployment.

### ✨ Features

#### Development
- 🔥 **Hot Reload** - Instant React/FastAPI changes
- 🐳 **Docker Dev** - Complete containerized development
- 🔍 **API Docs** - Auto-generated Swagger/ReDoc
- 🌸 **Flower** - Celery task monitoring

#### Production
- 🔒 **SSL/TLS** - Auto Let's Encrypt with renewal
- 🛡️ **Security** - API not publicly accessible
- 🌐 **Domain** - Frontend-only public access
- 📊 **Monitoring** - Health checks + metrics
- ⚡ **Performance** - Optimized builds + caching

#### Backend
- ⚡ FastAPI with async SQLAlchemy
- 🐘 PostgreSQL database
- 📮 Redis caching
- 📅 Celery Beat scheduled tasks
- 🌸 Flower monitoring UI
- 🔐 JWT authentication
- 📝 Alembic migrations

#### Frontend
- ⚛️ React 18 with TypeScript
- 🎨 Tailwind CSS
- 📱 Responsive design
- 🔄 React Query
- 🎯 Type-safe API calls

---

## 📋 Quick Start

### Development Mode (Hot Reload)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your settings

# 2. Start development
./manage.sh dev

# 3. Access applications
# Admin:  http://localhost:8001/admin
# API:    http://localhost:8001/docs
# React:  http://localhost:3000
# Flower: http://localhost:5555

# Login: admin@gmail.com / admin123
```

**Now edit any React file and see instant changes!** 🔥

### Production Mode (SSL + Security)

```bash
# 1. Configure production
cp .env.example .env.prod

# 2. Update .env.prod:
#    - DOMAIN=your-domain.com
#    - ADMIN_EMAIL=your@email.com
#    - Change ALL passwords!
#    - SSL_PRODUCTION=true

# 3. Point domain DNS to server IP

# 4. Setup SSL certificates
./manage.sh ssl-setup

# 5. Start production
./manage.sh prod

# 6. Access via your domain
# https://your-domain.com/admin
```

---

## 🏗️ Architecture

### Development Architecture
```
┌─────────────────────────────────────────┐
│     Nginx (localhost:8001)              │
│     Reverse Proxy with HMR              │
└──────┬──────────────────────────┬───────┘
       │                          │
┌──────▼────────┐        ┌────────▼────────┐
│   Frontend    │        │    Backend      │
│  Vite Dev     │◄──────►│    FastAPI      │
│  HOT RELOAD   │        │   --reload      │
│  Port 3000    │        │   Port 8000     │
└───────────────┘        └─────────┬───────┘
                                   │
        ┌──────────────────────────┼────────┐
        │                          │        │
┌───────▼────────┐  ┌──────────────▼──┐  ┌─▼─────────┐
│ Celery Worker  │  │  Celery Beat    │  │PostgreSQL │
│  + Flower      │  │  (Scheduler)    │  │           │
└───────┬────────┘  └──────────┬──────┘  └───────────┘
        └──────────┬───────────┘
                   │
            ┌──────▼──────┐
            │    Redis    │
            └─────────────┘
```

### Production Architecture (Secure)
```
                    Internet
                       │
                  Port 443 (HTTPS)
                       │
         ┌─────────────▼──────────────┐
         │  Nginx + Let's Encrypt     │
         │  SSL Termination           │
         └─────────────┬──────────────┘
                       │
         ┌─────────────┴──────────────┐
         │  PUBLIC: Frontend Only     │
         │  BLOCKED: Direct API Access│
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │   Frontend Container       │
         │   (Calls API internally)   │
         └─────────────┬──────────────┘
                       │ Docker Network
         ┌─────────────▼──────────────┐
         │   Backend Container        │
         │   (NOT Public)             │
         └────────────────────────────┘
```

**Security**: Only the frontend admin panel is publicly accessible. API is blocked from external access and only accessible through Docker internal network.

---

## 🛠️ Management Commands

### Essential Commands

```bash
# Development
./manage.sh dev              # Start dev with hot reload
./manage.sh logs dev -f      # Watch logs
./manage.sh shell            # Backend shell
./manage.sh frontend         # Frontend shell

# Production
./manage.sh prod             # Start production
./manage.sh ssl-setup        # Setup SSL
./manage.sh ssl-renew        # Renew SSL

# Database
./manage.sh migrate          # Run migrations
./manage.sh db-shell         # PostgreSQL shell
./manage.sh backup           # Backup database
./manage.sh restore file.sql # Restore database

# Monitoring
./manage.sh status           # Service status
./manage.sh tasks            # View scheduled tasks
./manage.sh flower           # Flower info

# Maintenance
./manage.sh restart          # Restart services
./manage.sh clean            # Remove all
```

---

## 📁 Project Structure

```
project/
├── .env                  # Development config
├── .env.prod                 # Production config
├── manage.sh                 # Management CLI ⭐
├── init.sh                   # Initialize DB
├── setup-ssl.sh              # SSL setup
│
├── backend/
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── core/            # Config, security, DB
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   ├── repositories/    # Data access
│   │   ├── tasks/           # Celery tasks
│   │   │   ├── celery_app.py       # Celery config
│   │   │   ├── tasks.py            # Event tasks
│   │   │   └── scheduled_tasks.py  # Cron tasks
│   │   └── middleware/      # Custom middleware
│   ├── alembic/             # Migrations
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API services
│   │   ├── hooks/           # Custom hooks
│   │   └── types/           # TypeScript types
│   ├── Dockerfile.dev       # Dev (hot reload)
│   └── Dockerfile           # Production
│
├── nginx/
│   ├── nginx.dev.conf       # Development
│   ├── nginx.prod.conf      # Production + SSL
│   ├── Dockerfile
│   └── ssl/                 # SSL certificates
│
├── docker-compose.dev.yml   # Dev environment
└── docker-compose.prod.yml  # Prod environment
```

---

## 🔐 Security Features

### Production Security

1. **SSL/TLS**
   - Automatic Let's Encrypt certificates
   - Auto-renewal every 12 hours
   - HSTS enabled
   - TLS 1.2/1.3 only

2. **API Protection**
   - Backend NOT publicly accessible
   - Only internal Docker network access
   - API docs disabled in production

3. **Headers**
   - X-Frame-Options: SAMEORIGIN
   - X-Content-Type-Options: nosniff
   - X-XSS-Protection enabled
   - Strict-Transport-Security

4. **Rate Limiting**
   - 10 req/s general
   - 30 req/s API
   - Connection limiting

5. **Authentication**
   - JWT tokens
   - bcrypt password hashing
   - Token expiration
   - Refresh tokens

---

## 📅 Celery Scheduled Tasks

Automatic background tasks:

| Task | Schedule | Purpose |
|------|----------|---------|
| System Health Check | Every 5 min | Monitor DB, Redis, disk |
| Cleanup Inactive Users | Daily 2 AM | Deactivate old accounts |
| Daily Reports | Daily 9 AM | Generate analytics |
| Task History Cleanup | Weekly Sun 3 AM | Remove old logs |
| Database Maintenance | Monthly 1st 4 AM | VACUUM, optimize |
| Cache Warming | Hourly | Pre-populate cache |

View tasks: `./manage.sh tasks`

---

## 🌐 Environment Configuration

### .env (Development)

```bash
ENVIRONMENT=development
DEBUG=true

DOMAIN=localhost
ADMIN_EMAIL=admin@gmail.com

# Weak passwords OK for dev
DB_PASSWORD=admin123
SECRET_KEY=dev-secret-key

# Ports
NGINX_HTTP_PORT=8001
FRONTEND_DEV_PORT=3000
```

### .env.prod (Production)

```bash
ENVIRONMENT=production
DEBUG=false

# YOUR DOMAIN
DOMAIN=admin.yourdomain.com
ADMIN_EMAIL=admin@yourdomain.com

# SSL
SSL_PRODUCTION=true  # false for staging/testing

# STRONG PASSWORDS REQUIRED
DB_PASSWORD=CHANGE-TO-STRONG-PASSWORD-64-CHARS
SECRET_KEY=CHANGE-TO-RANDOM-STRING-64-CHARS
REDIS_PASSWORD=CHANGE-TO-STRONG-PASSWORD
FLOWER_PASSWORD=CHANGE-TO-STRONG-PASSWORD

# Security
ACCESS_TOKEN_EXPIRE_MINUTES=15
RATE_LIMIT_PER_MINUTE=60
```

---

## 🔄 Development Workflow

### Hot Reload in Action

```bash
# 1. Start development
./manage.sh dev

# 2. Edit React component
nano frontend/src/pages/Dashboard.tsx
# Save file → See changes INSTANTLY! ⚡

# 3. Edit Python endpoint
nano backend/app/api/v1/endpoints/users.py
# Save file → Server restarts in 2s ⚡

# 4. Add npm package
./manage.sh frontend
npm install react-icons
exit

# 5. Add python package
nano backend/pyproject.toml
# Add package, then:
./manage.sh build dev
./manage.sh restart
```

### No Docker Rebuilds Needed!

Traditional: Edit → Build (5 min) → Restart → Test
**Now**: Edit → See instantly! 🔥

---

## 🚀 Production Deployment

### Step-by-Step Production Deployment

#### 1. Server Setup

```bash
# Update server
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Clone/Upload project
git clone your-repo.git
cd fast_react_admin
```

#### 2. Configure Environment

```bash
# Copy and edit production config
cp .env.example .env.prod

# CRITICAL: Update these in .env.prod
# - DOMAIN=your-domain.com
# - All passwords with strong random strings
# - SSL_PRODUCTION=true (or false for testing)
```

#### 3. DNS Configuration

Point your domain to server:
```
A Record: @ → your-server-ip
A Record: admin → your-server-ip (if subdomain)
```

Wait for DNS propagation (check: `nslookup your-domain.com`)

#### 4. SSL Setup

```bash
# This will:
# - Verify domain is reachable
# - Obtain Let's Encrypt certificate
# - Setup auto-renewal
./manage.sh ssl-setup

# For testing first, use staging in .env.prod:
# SSL_PRODUCTION=false
```

#### 5. Deploy

```bash
# Build production images
./manage.sh build prod

# Start production
./manage.sh prod

# Check status
./manage.sh status
```

#### 6. Verify

```bash
# Check SSL
curl -I https://your-domain.com/admin

# Check services
./manage.sh status

# View logs
./manage.sh logs prod -f
```

---

## 🔧 Troubleshooting

### Hot Reload Not Working

```bash
# Check if dev mode
./manage.sh status

# Restart dev services
./manage.sh restart

# Check volumes
docker inspect fastadmin_frontend_dev | grep Mounts
```

### SSL Certificate Failed

```bash
# 1. Check DNS
nslookup your-domain.com

# 2. Check ports
sudo netstat -tlnp | grep -E ':(80|443)'

# 3. Try staging first
# In .env.prod: SSL_PRODUCTION=false
./manage.sh ssl-setup

# 4. Check logs
./manage.sh logs prod certbot
```

### Database Issues

```bash
# Check database
./manage.sh db-shell

# Run migrations
./manage.sh migrate

# View logs
./manage.sh logs prod postgres
```

### Backend Not Accessible

**This is correct for production!** Backend is NOT publicly accessible for security.

Frontend calls API through Docker internal network.

For development access:
```bash
# Development mode has API docs at:
http://localhost:8001/docs
```

---

## 📊 Monitoring & Maintenance

### Health Monitoring

```bash
# Service status
./manage.sh status

# View logs
./manage.sh logs prod -f

# Health endpoint (internal)
curl http://backend:8000/health
```

### Scheduled Tasks

```bash
# View schedule
./manage.sh tasks

# Flower UI (dev)
http://localhost:5555
```

### Database Backups

```bash
# Create backup
./manage.sh backup

# Restore backup
./manage.sh restore backup_20250101_120000.sql

# Automated backups (add to crontab)
0 2 * * * cd /path/to/project && ./manage.sh backup
```

---

## 🎓 Adding Features

### Add New API Endpoint

1. Create schema in `backend/app/schemas/`
2. Create model in `backend/app/models/`
3. Create repository in `backend/app/repositories/`
4. Create service in `backend/app/services/`
5. Create endpoint in `backend/app/api/v1/endpoints/`
6. Register in `backend/app/api/v1/__init__.py`

### Add New Frontend Page

1. Create page in `frontend/src/pages/`
2. Add route in `frontend/src/App.tsx`
3. Add nav link in `frontend/src/components/layout/AdminLayout.tsx`

### Add Scheduled Task

1. Add task in `backend/app/tasks/scheduled_tasks.py`
2. Add schedule in `backend/app/tasks/celery_app.py`
3. Restart: `./manage.sh restart`

---

## 📚 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend API | FastAPI 0.109+ | Async REST API |
| Database | PostgreSQL 16 | Primary database |
| Cache | Redis 7 | Caching + broker |
| Task Queue | Celery 5.3+ | Async tasks |
| Scheduler | Celery Beat | Cron jobs |
| Monitoring | Flower 2.0+ | Task monitoring |
| Frontend | React 18 | UI framework |
| Language | TypeScript | Type safety |
| Styling | Tailwind CSS | Utility CSS |
| State | React Query | Server state |
| Gateway | Nginx | Reverse proxy |
| SSL | Let's Encrypt | Free SSL certs |
| Container | Docker | Containerization |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test in dev mode
5. Submit pull request

---

## 📄 License

MIT License - feel free to use for any project!

---

## 🆘 Support

### Quick Help

```bash
# View all commands
./manage.sh help

# Check documentation
cat README.md

# View service logs
./manage.sh logs -f [service]
```

### Common Issues

1. **Port already in use**: Change ports in .env file
2. **SSL failed**: Ensure DNS points to server
3. **Hot reload not working**: Ensure dev mode active
4. **Cannot access API**: Correct - API not public in prod

---

**Built for Production-Ready Enterprise Applications** 🚀

Generate with: `./enterprise_admin_generator.sh my_project`
README_EOF

# Create .gitignore
cat > .gitignore << 'GITIGNORE_EOF'
# Environment
.env
.env.dev
.env.prod
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/

# Node
node_modules/
dist/
build/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
*.pid

# Celery
celerybeat-schedule*
celerybeat.pid

# Backups
backup_*.sql

# SSL
nginx/ssl/*
!nginx/ssl/.gitkeep

# Logs
*.log
logs/

# Database
*.db
*.sqlite
GITIGNORE_EOF

touch nginx/ssl/.gitkeep

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PROJECT GENERATED SUCCESSFULLY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Project: $PROJECT_NAME"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "   cd $PROJECT_NAME"
echo ""
echo "   # For Development (Hot Reload):"
echo "   ./manage.sh dev"
echo ""
echo "   # For Production (SSL + Security):"
echo "   # 1. Edit .env.prod with your domain"
echo "   # 2. ./manage.sh ssl-setup"
echo "   # 3. ./manage.sh prod"
echo ""
echo "📖 Read README.md for complete documentation"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

