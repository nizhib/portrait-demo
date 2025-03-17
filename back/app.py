"""
Made by @nizhib
"""

import base64
import io
import logging
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import UJSONResponse as JSONResponse
from imageio import imsave
from PIL import Image
from pydantic import BaseModel

from api import Segmenter

LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "[%(asctime)s] %(name)s:%(lineno)d: %(message)s"

logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)

segmenter = Segmenter()

app = FastAPI(root_path="/api")
logger = logging.getLogger(__file__)


class ImageOrUrl(BaseModel):
    image: str | None = None
    url: str | None = None


@dataclass
class MaskData:
    mask: str


@dataclass
class ResponseData:
    success: bool = False
    data: MaskData | None = None
    message: str | None = None
    total: float | None = None


@app.post("/segment")
def segment(image: ImageOrUrl) -> JSONResponse:
    start: float = time.process_time()
    status: HTTPStatus = HTTPStatus.OK
    result: ResponseData = ResponseData()

    try:
        if image.image:
            blob = io.BytesIO(base64.b64decode(image.image))
        elif image.url:
            r = httpx.get(image.url, timeout=5)
            r.raise_for_status()
            blob = io.BytesIO(r.content)
        else:
            raise ValueError('Either "image" or "url" must be specified')
        img = Image.open(blob).convert("RGB")

        mask = segmenter.predict(img)
        mask = (mask * 255).astype(np.uint8)

        buffer = io.BytesIO()
        imsave(buffer, mask, "png")  # type: ignore
        buffer.seek(0)
        mask64 = base64.b64encode(buffer.read()).decode("utf-8")

        result.data = MaskData(mask64)
        result.success = True
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        result.message = str(e)
        status = HTTPStatus.BAD_REQUEST
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        result.message = str(e)
        status = HTTPStatus.INTERNAL_SERVER_ERROR

    result.total = time.process_time() - start

    return JSONResponse(asdict(result), status_code=status)
