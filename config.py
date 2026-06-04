"""
config.py - Application Configuration
=====================================
Loads all settings from the .env file and makes them available
throughout the application. Uses Pydantic for validation so you
get clear error messages if something is missing or wrong.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load the .env file from the same directory as this config file
# This makes the app work regardless of where you run it from
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)


class Settings(BaseSettings):
    """
    All application settings loaded from environment variables / .env file.
    Each field has a default value so the app still runs during development
    even if some settings are missing.
    """

    # -------------------------------------------------------------------------
    # OpenAI Settings
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(
        default="sk-proj-YOUR_KEY_HERE",
        description="Your OpenAI API key"
    )
    openai_chat_model: str = Field(
        default="gpt-4-turbo-preview",
        description="GPT model used for generating answers"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model used for creating text embeddings"
    )

    # -------------------------------------------------------------------------
    # Google Drive Settings
    # -------------------------------------------------------------------------
    google_drive_credentials_path: str = Field(
        default="credentials.json",
        description="Path to Google Drive OAuth credentials file"
    )
    google_drive_folder_name: str = Field(
        default="HR Policies",
        description="Name of the Google Drive folder with HR documents"
    )

    # -------------------------------------------------------------------------
    # Database Settings
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="sqlite:///./hr_policies.db",
        description="Database connection string"
    )

    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    app_name: str = Field(default="HR Policy Chatbot")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(
        default="AI-powered HR Policy Assistant using RAG"
    )
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # -------------------------------------------------------------------------
    # RAG Pipeline Settings
    # -------------------------------------------------------------------------
    top_k_results: int = Field(
        default=5,
        description="Number of document chunks to retrieve per query"
    )
    chunk_size: int = Field(
        default=500,
        description="Characters per text chunk"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlapping characters between chunks"
    )
    min_confidence_score: float = Field(
        default=0.3,
        description="Minimum similarity score to consider a result relevant"
    )
    max_context_tokens: int = Field(
        default=4000,
        description="Maximum tokens in the context window for GPT"
    )

    # -------------------------------------------------------------------------
    # CORS Settings
    # -------------------------------------------------------------------------
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins"
    )

    class Config:
        # Tell Pydantic to read from environment variables (case-insensitive)
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore any extra env vars we don't define here

    @property
    def allowed_origins_list(self) -> list[str]:
        """Convert the comma-separated origins string into a Python list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def is_openai_configured(self) -> bool:
        """Check if a real OpenAI API key has been provided."""
        return (
            self.openai_api_key != "sk-proj-YOUR_KEY_HERE"
            and len(self.openai_api_key) > 20
        )

    @property
    def credentials_file_path(self) -> Path:
        """Return the full path to the Google credentials file."""
        path = Path(self.google_drive_credentials_path)
        if not path.is_absolute():
            # If relative path, look next to this config file
            path = BASE_DIR / path
        return path


# Create a single shared instance used throughout the application
# Import this in other files: from config import settings
settings = Settings()


def print_config_summary():
    """Print a summary of current configuration (hides secrets)."""
    print("=" * 50)
    print(f"  {settings.app_name} v{settings.app_version}")
    print("=" * 50)
    print(f"  OpenAI configured : {settings.is_openai_configured}")
    print(f"  Chat model        : {settings.openai_chat_model}")
    print(f"  Embedding model   : {settings.openai_embedding_model}")
    print(f"  Drive folder      : {settings.google_drive_folder_name}")
    print(f"  Credentials path  : {settings.google_drive_credentials_path}")
    print(f"  Chunk size        : {settings.chunk_size}")
    print(f"  Top K results     : {settings.top_k_results}")
    print(f"  Debug mode        : {settings.debug}")
    print("=" * 50)
