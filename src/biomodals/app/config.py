"""Compatibility helpers for Biomodals app configuration."""

from __future__ import annotations

from functools import cached_property
from pathlib import PurePosixPath

from modal import CloudBucketMount, Volume
from pydantic import ConfigDict, computed_field

from biomodals.schema.app import AppConfig as SchemaAppConfig


class AppConfig(SchemaAppConfig):
    """App configuration with Modal-specific compatibility helpers."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @computed_field
    @cached_property
    def output_volume_name(self) -> str:
        """Name of the output volume."""
        return f"{self.name}-outputs"

    @computed_field
    @cached_property
    def output_volume(self) -> Volume:
        """Volume for storing outputs."""
        return Volume.from_name(
            self.output_volume_name, create_if_missing=True, version=2
        )

    def mounts(
        self,
        output_volume: bool = False,
        model_volume: bool = False,
        *,
        model_ro: bool = True,
        model_mount_subdir: bool = True,
        is_huggingface: bool = False,
    ) -> dict[str | PurePosixPath, Volume | CloudBucketMount]:
        """Generate the volume mountpoints for modal.Function definitions.

        Args:
            output_volume: Whether to mount the output volume.
            model_volume: Whether to mount the model volume for storing checkpoints.
                `self.model_volume_mountpoint` will be used as the mount point.
            model_ro: Whether to mount the model volume as read-only.
            model_mount_subdir: If True, only mount a subdirectory of the volume
                for isolation from other apps. Otherwise, mount the full volume.
            is_huggingface: Whether the model is managed by HuggingFace.

        Returns:
            A dictionary mapping volume mount points to volumes.
        """
        volumes = {}
        if output_volume:
            volumes[self.output_volume_mountpoint] = self.output_volume
        if model_volume:
            from biomodals.helper.constant import MODEL_VOLUME

            if model_mount_subdir:
                sub_path = (
                    "/huggingface" if is_huggingface else self.model_volume_subdir
                )
            else:
                sub_path = None

            volumes[self.model_volume_mountpoint] = MODEL_VOLUME.with_mount_options(
                read_only=model_ro, sub_path=sub_path
            )
        return volumes


__all__ = ["AppConfig"]
