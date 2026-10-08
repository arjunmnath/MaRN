"""Stub and design documentation for lazy parameter tree streaming.

Current status
--------------
``LazyParameterTree`` is **not yet implemented**.  This module documents the
design constraints and the conditions under which true on-demand streaming
would be safe, so the API surface can be stabilised in a later milestone.

Why true streaming is hard
--------------------------
:func:`torch.func.functional_call` requires a complete mapping of
``{name: tensor}`` to be passed at call time.  It cannot accept a generator or
iterator; the target module's ``__call__`` inspects the parameter dict for
every named parameter before executing any computation.

A future implementation could intercept ``functional_call`` at a lower level
(e.g. by subclassing :class:`~torch.nn.Module` and overriding
``_call_impl``) to feed tensors into the forward graph one at a time.  However
this would:

* Break ``torch.compile`` full-graph capture (dynamic iteration = graph break).
* Require target models to expose a layer-by-layer execution interface.
* Complicate gradient accumulation across discarded intermediates.

Safe use cases for future ``LazyParameterTree``
------------------------------------------------
* Targets whose layers are executed sequentially (e.g. plain ``nn.Sequential``
  with no skip connections).
* Inference-only (no gradient) use cases where tensors can be freed after each
  layer's forward.
* Eager-mode execution only (no ``torch.compile``).

For the current milestone, :class:`~marn.generators.lazy.LazyLayerwiseGenerator`
provides a soft-lazy alternative that is compatible with the existing
``functional_call``-based execution path.
"""

from __future__ import annotations


class LazyParameterTree:
    """Placeholder for a future on-demand parameter streaming interface.

    .. warning::
        This class is **not implemented**.  Constructing it raises
        :exc:`NotImplementedError` with a pointer to the design constraints
        documented in this module.

    See the module docstring for the conditions required before this can be
    safely implemented.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "LazyParameterTree is not yet implemented.  "
            "See marn.runtime.lazy for design constraints, or use "
            "LazyLayerwiseGenerator for the current soft-lazy alternative."
        )
