"""Authenticated CSV-first local review, independent of scientific admission."""

from typing import Annotated

from fastapi import APIRouter, Depends

from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    PrivateResultRoute,
    require_session,
    require_unsafe_session,
)
from biomodals.service.protein_optimization.contracts import (
    OptimizationReview,
    OptimizationReviewRequest,
    ProteinOptimizationOptions,
)
from biomodals.service.protein_optimization.review import review_inputs


def create_review_router() -> APIRouter:
    """Expose options and explicit review without a Job or provider invocation."""
    router = APIRouter(
        prefix="/api/v1/protein-optimization",
        tags=["protein-optimization"],
        route_class=PrivateResultRoute,
    )

    @router.get("/options", response_model=ProteinOptimizationOptions)
    def options(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> ProteinOptimizationOptions:
        return ProteinOptimizationOptions()

    @router.post(
        "/review",
        response_model=OptimizationReview,
        responses={422: {"model": CodedErrorResponse}},
    )
    def review(
        body: OptimizationReviewRequest,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> OptimizationReview:
        try:
            return review_inputs(body)
        except ValueError as exc:
            raise CodedAPIError(422, "invalid_measurements", str(exc)) from exc

    return router
