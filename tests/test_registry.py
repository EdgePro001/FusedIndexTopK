from __future__ import annotations

import sys
import types

import pytest

from index_topk_perflab.api import VariantDescriptor
from index_topk_perflab.registry import available_variants, load_variant


def test_builtin_baseline_load_is_cuda_lazy() -> None:
    sys.modules.pop("deep_gemm", None)
    plugin = load_variant("deepgemm_torch_unfused", options={"sorted": False})
    assert plugin.descriptor.plugin_id == "deepgemm_torch_unfused"
    assert plugin.descriptor.exact_topk is True
    assert "deep_gemm" not in sys.modules


def test_explicit_module_factory_can_be_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("test_external_variant")

    class Plugin:
        descriptor = VariantDescriptor(
            plugin_id="external_test",
            display_name="external",
            api_version="1.0",
            implementation_version="1",
            mode="fused",
            description="test",
            implementation="python",
            exact_topk=True,
            supported_arches=(),
        )

        def supports(self, case: object) -> bool:
            return True

        def fingerprint_metadata(self) -> dict[str, bool]:
            return {"test": True}

        def prepare(self, case: object, inputs: object, **kwargs: object) -> object:
            return object()

    def create_variant(options=None):
        return Plugin()

    module.create_variant = create_variant
    monkeypatch.setitem(sys.modules, "test_external_variant", module)
    plugin = load_variant("test_external_variant:create_variant")
    assert plugin.descriptor.plugin_id == "external_test"


def test_unknown_variant_error_lists_available_names() -> None:
    with pytest.raises(KeyError, match="unknown variant"):
        load_variant("does_not_exist")


def test_available_variants_contains_baseline() -> None:
    variants = available_variants()
    assert variants["deepgemm_torch_unfused"].endswith(":create_variant")
    assert variants["deepgemm_flashinfer_topk_auto"].endswith(":create_auto_variant")
    assert variants["fused_index_topk"].endswith(":create_variant")
    assert "deepgemm_raft_select_k_auto" not in variants
