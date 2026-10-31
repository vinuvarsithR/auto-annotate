"""
errors.py - every error leaves the API as {"error": {"code": ..., "message": ...}}.

Raise ApiError(404, "not_found", "image 99 not found") anywhere in a route.
FastAPI's own errors (bad JSON body, unknown path, ...) are converted to the same shape.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("app")

_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed",
          409: "conflict", 500: "internal_error"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def not_found(what: str, ident: int) -> ApiError:
    return ApiError(404, "not_found", f"{what} {ident} not found")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        parts = []
        for e in exc.errors():
            where = ".".join(str(p) for p in e.get("loc", ()) if p != "body")
            parts.append(f"{where}: {e.get('msg', 'invalid')}" if where else e.get("msg", "invalid"))
        return JSONResponse(error_body("bad_request", "; ".join(parts) or "invalid request"),
                            status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = _CODES.get(exc.status_code, "error")
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(error_body("internal_error", type(exc).__name__), status_code=500)
