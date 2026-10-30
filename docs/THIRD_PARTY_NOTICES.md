<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->
<!-- Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio) -->

# Third-party notices

This pack's own code is PolyForm-Noncommercial-1.0.0 (a commercial licence is available
separately). It drives, or adapts code from, the projects below, each of which keeps its own
terms. This is a summary of licence texts as we read them, **not legal advice**; read the
originals before shipping anything commercially.

Status key: **read** = we read the licence file or page; **claimed** = stated by the project's own
documentation and not independently checked; **unverified** = not yet checked.

## Code adapted into this pack

### ComfyUI-SplatKit (MIT)

`cumuli_bridge/pose.py` is adapted from `core/four_d_anyone/pose.py` in
[ComfyUI-SplatKit](https://github.com/mickmumpitz/ComfyUI-SplatKit) by mickmumpitz (MIT, **read**).
The pose-handover design in our 4DAnyone fork also comes from that project's vendored 4DAnyone fork.

```
MIT License

Copyright (c) 2026 mickmumpitz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Models and projects the pipeline runs

| Component | Role | Licence | Commercial use |
|---|---|---|---|
| 4DAnyone (our fork of ant-research/4DAnyone) code and checkpoint | Ring generation | Apache-2.0 (**read**) | Yes. Training-data provenance of the checkpoint is **unverified**. |
| Wan2.2-TI2V-5B base, VAE, UMT5 | Backbone | Apache-2.0 (**read**) | Yes |
| Wan2.2 TI2V 5B **Turbo LoRA** | 4-step denoising (`enable_turbo`) | CC BY-NC-SA 4.0 (**read**, quanhaol/Wan2.2-TI2V-5B-Turbo `LICENSE.md`) | **No.** Off by default. |
| SAM 3D Body (Meta) | Body pose | SAM License, 2025-11-19 (**read** from the GitHub copy; the Hugging Face copy was not read) | Yes, with conditions below |
| MHR (Meta) | Body rig / keypoint layout | Apache-2.0 (**read**) | Yes. The terms of downloadable model assets are **unverified**. |
| Sapiens2 (Meta) | MHR70 keypoint names and order only | Sapiens2 licence (**read**) | Conditional, see below |
| BiRefNet | Foreground masks | MIT (**read**) | Yes |
| HLOC, LightGlue, ALIKED, pycolmap/COLMAP | Real-capture rig solve | Apache-2.0, Apache-2.0, BSD-3, BSD-3 (**read**) | Yes |
| SuperPoint (weights and code) | Optional `feature_type` | Magic Leap: academic / non-profit non-commercial research only (**read**) | **No.** Not the default. |
| cumuli-trainer, cumuli-core (Solipsist Studios) | 4DGS trainer and the format library it shares with the bake | PolyForm-Noncommercial-1.0.0, the same as this pack (**read**: each repository's `LICENSE.md`); a commercial licence is available separately. Each repository's `PROVENANCE.md` records what it was written from. | Under that commercial licence |
| gsplat (nerfstudio-project) | The trainer's rasterizer, installed unmodified; its CUDA kernels are compiled on first use | Apache-2.0 (**read**: the installed distribution's `LICENSE`) | Yes |
| lpips | The trainer's perceptual metric | BSD-style: copyright notice and "all rights reserved" terms (**read**: the installed distribution's `LICENSE`) | Yes, keeping the notice |
| GVHMR, SMPL-X, ultralytics | Former pose path (removed) | Research-only; non-commercial; AGPL-3.0 | Not used by this pack any more |
| Rerun web viewer (`@rerun-io/web-viewer` 0.37.1) and `rerun-sdk` | Optional ring viewer node: the viewer's JS/WASM is fetched at install time, not shipped here | MIT, with Apache-2.0 also listed (**read**: the package's `package.json` and registry metadata) | Yes. The viewer's own requests to fonts, GitHub and telemetry are blocked by the page's Content-Security-Policy, not by a viewer setting. |
| ComfyUI | Host process | GPL-3.0 (**read**) | The pack imports `comfy.*`; how that sits with a non-GPL licence is **unverified**, ask counsel |

### SAM 3D Body: conditions that matter

Read from the SAM License file. It is a royalty-free licence with no non-commercial clause, and:

- Distribution of the materials or derivatives must carry the agreement.
- You must not use the materials for ITAR-controlled activities or end uses prohibited by trade
  controls, including military or warfare purposes, nuclear industries, espionage, or the
  development of weapons.
- You may not reverse engineer or help others do so.
- Meta may amend the agreement from time to time, effective immediately.
- Suing Meta over the materials, outputs or results ends the licence and obliges you to indemnify Meta.
- It asks for acknowledgement when results are submitted for research publication.

### Sapiens2: the keypoint schema

4DAnyone's skeleton conditioning uses keypoint names and ordering that its authors conservatively treat
as Sapiens2-derived. The Sapiens2 licence prohibits, among other things, surveillance, biometric
processing, identification or re-identification, deepfakes and deceptive content, and gives Meta audit
and amendment rights. Whether a human-video generator is affected by "biometric processing" is a legal
judgement, not something this file can settle.

## Open items before commercial use

1. Read Meta's SAM License as shipped with the weights you actually download.
2. Counsel's view on Sapiens2's use restrictions, and on ComfyUI's GPL-3.0 with this pack.
3. Provenance of the 4DAnyone checkpoint's training data (its SMPL-X-to-MHR70 regressor was trained on
   DNA-Rendering, which requires a separate signed agreement; this pack no longer uses that regressor).
