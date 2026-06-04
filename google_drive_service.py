"""
google_drive_service.py - Google Drive Integration
===================================================
Handles all communication with Google Drive:
  - Authenticating with your Google account
  - Finding the "HR Policies" folder
  - Listing documents inside that folder
  - Downloading documents to a local temp folder
  - Syncing all documents at once

HOW GOOGLE DRIVE AUTH WORKS (simple explanation):
  1. You download a credentials.json file from Google Cloud Console
  2. First time: a browser window opens, you log in & grant permission
  3. A token.json file is saved so you don't need to log in again
  4. After that: fully automatic!
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional

# Google API libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from config import settings

# Set up logging so we can see what's happening
logger = logging.getLogger(__name__)

# These are the permissions we ask Google for.
# "readonly" means we can only READ files, never modify them - safer!
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Supported file types and their Google Drive MIME types
SUPPORTED_MIME_TYPES = {
    # --- Documents ---
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",

    # --- PowerPoint ---
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".pptx",

    # --- Excel / CSV ---
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xlsx",
    "text/csv": ".csv",

    # --- Video (transcribed via OpenAI Whisper) ---
    "video/mp4": ".mp4",
    "video/quicktime": ".mp4",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
    "video/mpeg": ".mpeg",
    "video/x-ms-wmv": ".wmv",

    # --- Audio (transcribed via OpenAI Whisper) ---
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",

    # --- Google Workspace (auto-exported) ---
    "application/vnd.google-apps.document": ".docx",       # Google Docs → DOCX
    "application/vnd.google-apps.presentation": ".pptx",   # Google Slides → PPTX
    "application/vnd.google-apps.spreadsheet": ".xlsx",    # Google Sheets → XLSX
}

# Where to save downloaded files temporarily
DOWNLOAD_DIR = Path("./temp_documents")


class GoogleDriveService:
    """
    Service class for all Google Drive operations.

    Usage:
        service = GoogleDriveService()
        if service.authenticate():
            documents = service.sync_all_documents()
    """

    def __init__(self):
        """Initialize the service (does NOT connect yet - call authenticate() first)."""
        self.drive_service = None  # Will hold the Google Drive API client
        self.is_authenticated = False
        self._hr_folder_id = None  # Cached folder ID to avoid repeated lookups

        # Create the temp download folder if it doesn't exist
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("GoogleDriveService initialized. Call authenticate() to connect.")

    # =========================================================================
    # AUTHENTICATION
    # =========================================================================

    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive using OAuth2.

        First run: Opens browser for you to log in.
        Subsequent runs: Uses saved token.json automatically.

        Returns:
            True if authentication succeeded, False otherwise.
        """
        logger.info("Starting Google Drive authentication...")

        credentials_path = settings.credentials_file_path
        token_path = credentials_path.parent / "token.json"

        # Check if the credentials file exists
        if not credentials_path.exists():
            logger.error(
                f"credentials.json not found at: {credentials_path}\n"
                "Please download it from Google Cloud Console.\n"
                "See setup instructions for details."
            )
            return False

        creds = None

        # Try to load previously saved login token
        if token_path.exists():
            logger.info("Found saved token.json - loading existing credentials...")
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_path), SCOPES
                )
            except Exception as e:
                logger.warning(f"Could not load token.json: {e}. Will re-authenticate.")

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # Token expired but can be refreshed automatically
                logger.info("Token expired - refreshing automatically...")
                try:
                    creds.refresh(Request())
                    logger.info("Token refreshed successfully!")
                except Exception as e:
                    logger.warning(f"Token refresh failed: {e}. Need to re-authenticate.")
                    creds = None

            if not creds:
                # Need to do the full OAuth flow (opens browser)
                logger.info(
                    "Opening browser for Google login...\n"
                    "Please log in and grant permission to access your Drive."
                )
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_path), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    logger.info("Browser authentication successful!")
                except Exception as e:
                    logger.error(f"Browser authentication failed: {e}")
                    return False

            # Save the credentials for next time (so browser doesn't open again)
            try:
                token_path.write_text(creds.to_json())
                logger.info(f"Saved credentials to {token_path}")
            except Exception as e:
                logger.warning(f"Could not save token.json: {e}")

        # Build the Google Drive API client
        try:
            self.drive_service = build("drive", "v3", credentials=creds)
            self.is_authenticated = True
            logger.info("Successfully connected to Google Drive!")
            return True
        except Exception as e:
            logger.error(f"Failed to build Drive service: {e}")
            return False

    # =========================================================================
    # FOLDER OPERATIONS
    # =========================================================================

    def find_hr_folder(self) -> Optional[str]:
        """
        Find the "HR Policies" folder in Google Drive and return its ID.

        Returns:
            Folder ID string if found, None if not found.
        """
        if not self.is_authenticated:
            logger.error("Not authenticated. Call authenticate() first.")
            return None

        # Use cached folder ID if we already found it
        if self._hr_folder_id:
            return self._hr_folder_id

        folder_name = settings.google_drive_folder_name
        logger.info(f"Searching for folder: '{folder_name}'...")

        try:
            # Search for folders with the exact name
            query = (
                f"name = '{folder_name}' "
                f"and mimeType = 'application/vnd.google-apps.folder' "
                f"and trashed = false"
            )
            results = self.drive_service.files().list(
                q=query,
                fields="files(id, name, createdTime)",
                pageSize=5
            ).execute()

            folders = results.get("files", [])

            if not folders:
                logger.error(
                    f"Folder '{folder_name}' not found in Google Drive!\n"
                    f"Please create a folder named exactly '{folder_name}' "
                    f"and add your HR policy documents to it."
                )
                return None

            # Use the first matching folder
            folder = folders[0]
            self._hr_folder_id = folder["id"]
            logger.info(
                f"Found folder '{folder['name']}' (ID: {self._hr_folder_id})"
            )

            if len(folders) > 1:
                logger.warning(
                    f"Found {len(folders)} folders named '{folder_name}'. "
                    f"Using the first one. Rename duplicates to avoid confusion."
                )

            return self._hr_folder_id

        except HttpError as e:
            logger.error(f"Google Drive API error while searching for folder: {e}")
            return None

    # =========================================================================
    # FILE LISTING
    # =========================================================================

    def list_documents(self) -> list[dict]:
        """
        List all supported documents in the HR Policies folder.

        Returns:
            List of dicts, each with keys: id, name, mimeType, size, modifiedTime
        """
        folder_id = self.find_hr_folder()
        if not folder_id:
            return []

        logger.info("Listing documents in HR Policies folder...")
        documents = []

        try:
            # Build the query to find supported file types
            mime_types = list(SUPPORTED_MIME_TYPES.keys())
            mime_query = " or ".join(
                [f"mimeType = '{mt}'" for mt in mime_types]
            )
            query = (
                f"'{folder_id}' in parents "
                f"and ({mime_query}) "
                f"and trashed = false"
            )

            # Handle pagination (Google Drive returns max 100 files per page)
            page_token = None
            while True:
                response = self.drive_service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageSize=100,
                    pageToken=page_token
                ).execute()

                batch = response.get("files", [])
                documents.extend(batch)

                page_token = response.get("nextPageToken")
                if not page_token:
                    break  # No more pages

            logger.info(f"Found {len(documents)} document(s) in HR Policies folder.")
            for doc in documents:
                size_kb = int(doc.get("size", 0)) // 1024
                logger.info(f"  - {doc['name']} ({size_kb} KB, {doc['mimeType']})")

            return documents

        except HttpError as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    # =========================================================================
    # FILE DOWNLOADING
    # =========================================================================

    def download_document(self, file_id: str, file_name: str, mime_type: str) -> Optional[Path]:
        """
        Download a single document from Google Drive to the local temp folder.

        Args:
            file_id: Google Drive file ID
            file_name: Display name of the file
            mime_type: MIME type of the file

        Returns:
            Path to the downloaded file, or None if download failed.
        """
        logger.info(f"Downloading: {file_name}...")

        # Determine the correct file extension
        extension = SUPPORTED_MIME_TYPES.get(mime_type, "")
        if not extension:
            logger.warning(f"Unsupported file type '{mime_type}' for {file_name}. Skipping.")
            return None

        # Clean the filename for use as a local path
        safe_name = "".join(c for c in file_name if c.isalnum() or c in " ._-")
        safe_name = safe_name.strip()

        # Make sure it ends with the right extension
        if not safe_name.lower().endswith(extension):
            safe_name = safe_name + extension

        local_path = DOWNLOAD_DIR / safe_name

        try:
            # Google Workspace files must be exported (they have no binary content)
            GOOGLE_EXPORT_MAP = {
                "application/vnd.google-apps.document":     "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.google-apps.spreadsheet":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }

            if mime_type in GOOGLE_EXPORT_MAP:
                export_mime = GOOGLE_EXPORT_MAP[mime_type]
                request = self.drive_service.files().export_media(
                    fileId=file_id, mimeType=export_mime
                )
            else:
                # Download regular files (PDF, DOCX, PPTX, video, etc.) directly
                request = self.drive_service.files().get_media(fileId=file_id)

            # Stream the download to a file (handles large files efficiently)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.debug(f"  Downloading {file_name}: {progress}%")

            # Write the downloaded bytes to disk
            local_path.write_bytes(buffer.getvalue())
            logger.info(f"  Saved to: {local_path} ({local_path.stat().st_size // 1024} KB)")
            return local_path

        except HttpError as e:
            logger.error(f"Failed to download {file_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {file_name}: {e}")
            return None

    # =========================================================================
    # SYNC ALL DOCUMENTS
    # =========================================================================

    def sync_all_documents(self) -> list[dict]:
        """
        Main method: Download all HR policy documents from Google Drive.

        This is the method called by the API when syncing.

        Returns:
            List of dicts with 'name', 'path', 'mime_type' for each downloaded file.
        """
        logger.info("=" * 50)
        logger.info("Starting full document sync from Google Drive...")
        logger.info("=" * 50)

        if not self.is_authenticated:
            logger.error("Not authenticated. Call authenticate() first.")
            return []

        # Step 1: Get list of all documents
        documents = self.list_documents()
        if not documents:
            logger.warning(
                "No documents found to sync. "
                "Make sure the HR Policies folder exists and has documents."
            )
            return []

        # Step 2: Download each document
        downloaded = []
        failed = []

        for doc in documents:
            file_id = doc["id"]
            file_name = doc["name"]
            mime_type = doc["mimeType"]

            local_path = self.download_document(file_id, file_name, mime_type)

            if local_path:
                downloaded.append({
                    "name": file_name,
                    "path": str(local_path),
                    "mime_type": mime_type,
                    "drive_file_id": file_id,
                })
            else:
                failed.append(file_name)

        # Step 3: Report results
        logger.info("=" * 50)
        logger.info(f"Sync complete!")
        logger.info(f"  Successfully downloaded: {len(downloaded)} file(s)")
        if failed:
            logger.warning(f"  Failed: {len(failed)} file(s): {', '.join(failed)}")
        logger.info("=" * 50)

        return downloaded


# =============================================================================
# QUICK TEST - Run this file directly to test the connection
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("Testing Google Drive connection...")
    service = GoogleDriveService()

    if service.authenticate():
        print("Authentication successful!")
        docs = service.list_documents()
        print(f"Found {len(docs)} documents.")
    else:
        print("Authentication failed. Check credentials.json.")
