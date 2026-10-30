from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.processing.spreadsheet import DataError, read_table


async def read_upload(file: UploadFile):
    limit = get_settings().max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"arquivo excede {get_settings().max_upload_mb} MB")
    try:
        return read_table(file.filename or "", content)
    except DataError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
