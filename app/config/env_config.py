import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
  """Load and expose environment settings for the application."""

  def __init__(self) -> None:
    self.ENV: str = os.getenv("ENV")
    self.PROJECT_NAME: str = os.getenv("PROJECT_NAME")
    self.PROJECT_VERSION: str = os.getenv("PROJECT_VERSION")
    self.PROJECT_DESCRIPTION: str = os.getenv("PROJECT_DESCRIPTION")

    self.DB_FILE_PATH: str = os.getenv("DB_FILE_PATH")
    self.SQLALCHEMY_DATABASE_URL: str = os.getenv("SQLALCHEMY_DATABASE_URL")

    self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    working_dir = os.path.abspath(os.getenv("WORKING_DIR", ".").strip() or ".")
    self.WORKING_PROJECT_DIR: str = os.path.join(working_dir,self.PROJECT_NAME)
    self.UPLOAD_DIR: str = str(BASE_DIR / "static" / "upload")

    self.HF_TOKEN: str = os.getenv("HF_TOKEN")
    self.VECTOR_DB_TYPE: str = os.getenv('VECTOR_DB_TYPE')
    self.VECTOR_PERSIST_DIR: str = os.getenv("VECTOR_PERSIST_DIR")

settings = Settings()