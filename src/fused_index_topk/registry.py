"""Lazy plugin loading through config aliases, entry points, or module factories."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from collections.abc import Mapping
from typing import Any, Callable

from .api import VariantDescriptor, VariantPlugin

ENTRY_POINT_GROUP = "fused_index_topk.variants"
BUILTIN_VARIANTS = {
    "deepgemm_torch_unfused": "fused_index_topk.variants.deepgemm_torch:create_variant",
    "deepgemm_deepselect_topk": (
        "fused_index_topk.variants.deepgemm_deepselect.plugin:create_variant"
    ),
    "fused_index_topk": "fused_index_topk.kernel.plugin:create_variant",
}


def _load_factory(reference: str) -> Callable[..., Any]:
    if ":" not in reference:
        raise ValueError(f"variant factory must use module:object syntax: {reference!r}")
    module_name, object_name = reference.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(f"failed to import variant module {module_name!r}") from error
    try:
        factory = getattr(module, object_name)
    except AttributeError as error:
        raise RuntimeError(
            f"variant factory {object_name!r} does not exist in module {module_name!r}"
        ) from error
    if not callable(factory):
        raise TypeError(f"variant factory {reference!r} is not callable")
    return factory


def installed_entry_points() -> dict[str, str]:
    points = importlib.metadata.entry_points()
    selected = (
        points.select(group=ENTRY_POINT_GROUP)
        if hasattr(points, "select")
        else points.get(ENTRY_POINT_GROUP, ())
    )
    return {point.name: point.value for point in sorted(selected, key=lambda item: item.name)}


def available_variants() -> dict[str, str]:
    result = dict(BUILTIN_VARIANTS)
    result.update(installed_entry_points())
    return dict(sorted(result.items()))


def load_variant(
    reference: str,
    *,
    options: Mapping[str, Any] | None = None,
) -> VariantPlugin:
    references = available_variants()
    factory_reference = reference if ":" in reference else references.get(reference)
    if factory_reference is None:
        raise KeyError(f"unknown variant {reference!r}; available={sorted(references)}")
    factory = _load_factory(factory_reference)
    signature = inspect.signature(factory)
    accepts_options = "options" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_options:
        plugin = factory(options=dict(options or {}))
    else:
        if options:
            raise TypeError(
                f"variant factory {factory_reference!r} does not accept configured options"
            )
        plugin = factory()
    required = ("descriptor", "supports", "prepare", "fingerprint_metadata")
    missing = [name for name in required if not hasattr(plugin, name)]
    if missing:
        raise TypeError(
            f"factory {factory_reference!r} returned an invalid plugin; missing={missing}"
        )
    if not isinstance(plugin.descriptor, VariantDescriptor):
        raise TypeError(f"factory {factory_reference!r} returned an invalid descriptor")
    if plugin.descriptor.api_version.split(".", 1)[0] != "1":
        raise TypeError(
            f"variant {plugin.descriptor.plugin_id!r} uses incompatible API "
            f"{plugin.descriptor.api_version!r}"
        )
    if plugin.descriptor.mode not in {"unfused", "partially_fused", "fused"}:
        raise TypeError(
            f"variant {plugin.descriptor.plugin_id!r} has invalid mode "
            f"{plugin.descriptor.mode!r}"
        )
    if not plugin.descriptor.exact_topk:
        raise TypeError("FusedIndexTopK accepts only exact TopK variants")
    return plugin
