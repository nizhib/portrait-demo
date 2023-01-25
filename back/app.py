#!/usr/bin/env python3

"""
Made by @nizhib
"""

import base64
import io
import logging
import sys
import time
from dataclasses import dataclass, asdict
from http import HTTPStatus
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import UJSONResponse as JSONResponse
from imageio import imsave
from PIL import Image
from pydantic import BaseModel

from api import Segmentator

LOGGING_LEVEL = "DEBUG"
LOGGING_FORMAT = "[%(asctime)s] %(name)s:%(lineno)d: %(message)s"

logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)

segmentator = Segmentator()

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
    start = time.process_time()
    status = HTTPStatus.OK
    result = ResponseData()

    try:
        if image.image:
            blob = io.BytesIO(base64.b64decode(image.image))
        elif image.url:
            r = httpx.get(image.url, timeout=5)
            blob = io.BytesIO(r.content)
        else:
            raise ValueError('Either "image" or "url" must be specified')
        img = Image.open(blob).convert("RGB")

        mask = segmentator.predict(img)
        mask = (mask * 255).astype(np.uint8)

        fmem = io.BytesIO()
        imsave(fmem, mask, "png")  # type: ignore
        fmem.seek(0)
        mask64 = base64.b64encode(fmem.read()).decode("utf-8")

        result.data = MaskData(mask64)
        result.success = True
    except Exception as e:
        logger.exception(e)
        result.message = str(e)
        status = HTTPStatus.INTERNAL_SERVER_ERROR

    result.total = time.process_time() - start

    return JSONResponse(asdict(result), status_code=status)


if __name__ == "__main__":
    name = Path(__file__).stem
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 5000

    uvicorn.run(f"{name}:app", host="0.0.0.0", port=port, log_level="debug")
