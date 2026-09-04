# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""ComfyUI nodes taking one monocular clip to a streamable ``.sogst`` 4D asset.

The chain is: 4DAnyone generates a synchronized ring of novel views, the ring
becomes a D-NeRF style 4D Gaussian Splatting dataset, a rotor 4DGS model trains
on it, and the trained model is baked into ``.sogst``.

Everything runs in ComfyUI's own environment. Ring generation and training run
as child processes of the same interpreter -- not of a different environment --
because both need to configure the CUDA allocator before their first allocation
and both want the whole device. See ``cumuli_bridge/settings.py``.
"""

from .cumuli_bridge.nodes import comfy_entrypoint

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
