"""Provider adapters. Kept import-light: modal_app is imported inside Modal containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_agent.providers.base import Provider

PROVIDERS = ("modal", "fake")


def get_provider(name: str) -> Provider:
    if name == "modal":
        from gpu_agent.providers.modal import ModalProvider

        return ModalProvider()
    if name == "fake":
        from gpu_agent.providers.fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"unknown provider {name!r}; known: {', '.join(PROVIDERS)}")
