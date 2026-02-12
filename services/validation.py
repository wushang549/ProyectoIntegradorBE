import csv
from pathlib import Path

ALLOWED_EXTENSIONS = {".csv"}

def validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Only CSV files are supported at this stage")

def validate_text_column(file_path: Path) -> None:
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if not header:
        raise ValueError("CSV file is empty or missing header")

    columns = [c.strip() for c in header]
    if "text" not in columns:
        raise ValueError("Missing required column: text")
