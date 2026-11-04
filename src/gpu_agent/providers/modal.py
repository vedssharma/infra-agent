"""Modal adapter.

- Launch: deploy `modal_app` (cached after the first image build), look the function up by name,
  set GPU and TTL with `with_options`, and `spawn` it. Returns at once; the run survives the CLI.
- TTL: Modal's own `timeout=`, so teardown never depends on this machine.
- Status/logs: the in-container runner writes state and log chunks to the `gpu-agent-runs`
  modal.Dict; the FunctionCall result tells us about kills (timeout, cancel) the runner cannot report.
- Stop: `FunctionCall.cancel(terminate_containers=True)`.

Auth: Modal token from `modal token new` (~/.modal.toml) or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET.
Hub pushes need a Modal secret named `gpu-agent-hf` holding HF_TOKEN.
"""

from __future__ import annotations

from typing import Any

from gpu_agent import pricing
from gpu_agent.models import JobSpec
from gpu_agent.providers.base import CallPoll, CallStatus, ProviderError, Store, StoreBackedProvider


class ModalProvider(StoreBackedProvider):
    name = "modal"
    price_as_of = pricing.MODAL_PRICES_AS_OF
    usd_per_hour = pricing.MODAL_USD_PER_HOUR

    def __init__(self, store: Store | None = None, modal_module: Any = None, deploy: bool = True):
        """`modal_module` and `store` exist for tests; normally both come from the real SDK."""
        if modal_module is None:
            import modal as modal_module
        self.modal = modal_module
        self.deploy = deploy
        self._deployed = False
        if store is None:
            from gpu_agent.providers.modal_app import STORE_NAME

            store = _DictStore(self.modal.Dict.from_name(STORE_NAME, create_if_missing=True))
        super().__init__(store)

    def normalize_gpu(self, gpu: str) -> str:
        return pricing.normalize_modal_gpu(gpu)

    def _function(self) -> Any:
        from gpu_agent.providers import modal_app

        if self.deploy and not self._deployed:
            with self.modal.enable_output():
                modal_app.app.deploy(name=modal_app.APP_NAME)
            self._deployed = True
        return self.modal.Function.from_name(modal_app.APP_NAME, modal_app.FUNCTION_NAME)

    def _spawn(self, spec: JobSpec, run_id: str, gpu: str, timeout_s: int) -> str:
        try:
            fn = self._function().with_options(gpu=gpu, timeout=timeout_s)
            call = fn.spawn(spec.to_train_json(), run_id)
        except Exception as e:  # auth, missing secret, image build failure, capacity...
            raise ProviderError(f"Modal launch failed: {type(e).__name__}: {e}") from e
        return call.object_id

    def _poll(self, ref: str) -> CallPoll:
        exc = self.modal.exception
        try:
            result = self.modal.FunctionCall.from_id(ref).get(timeout=0)
        except exc.FunctionTimeoutError:  # subclass of modal TimeoutError: check first
            return CallPoll(CallStatus.TIMED_OUT, error="Modal function timeout (TTL) reached")
        except exc.OutputExpiredError:
            return CallPoll(CallStatus.FAILED, error="Modal result expired")
        except exc.TimeoutError:
            return CallPoll(CallStatus.RUNNING)
        except exc.InputCancellation:
            return CallPoll(CallStatus.CANCELLED)
        except Exception as e:
            return CallPoll(CallStatus.FAILED, error=f"{type(e).__name__}: {e}")
        return CallPoll(CallStatus.SUCCEEDED, result=result if isinstance(result, dict) else None)

    def _cancel(self, ref: str) -> None:
        self.modal.FunctionCall.from_id(ref).cancel(terminate_containers=True)


class _DictStore:
    """Adapts modal.Dict to the Store protocol (it already has get/put/keys; this pins the API)."""

    def __init__(self, d: Any):
        self.d = d

    def get(self, key: Any, default: Any = None) -> Any:
        return self.d.get(key, default)

    def put(self, key: Any, value: Any) -> Any:
        return self.d.put(key, value)

    def keys(self) -> Any:
        return self.d.keys()
