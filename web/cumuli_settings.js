// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
// Registers the bridge's user-tweakable options in ComfyUI's own Settings
// dialog (stored as cumuli.* keys in comfy.settings.json, which the python
// side re-reads at every queue -- no restart needed).
import { app } from "../../scripts/app.js";

const SETTINGS = [
  {
    id: "cumuli.work_root",
    name: "Work root (heavy per-run intermediates)",
    type: "text",
    defaultValue: "",
    tooltip: "Flipbook staging, 4DGS datasets and trainer checkpoints land under " +
             "<work_root>/<run_name>/ (~20 GB per run). Point it at a large drive. " +
             "Empty falls back to ComfyUI's temp and output directories.",
  },
  {
    id: "cumuli.dataset_roots",
    name: "Dataset roots (comma separated)",
    type: "text",
    defaultValue: "",
    tooltip: "Extra directories the Load 4DGS Dataset dropdown scans, on top of the work root.",
  },
  {
    id: "cumuli.flipbook_roots",
    name: "Flipbook roots (comma separated)",
    type: "text",
    defaultValue: "",
    tooltip: "Extra directories the Load Flipbook dropdown scans, on top of the work root.",
  },
  {
    id: "cumuli.min_free_vram_gb",
    name: "Minimum free VRAM to start a ring (GB)",
    type: "number",
    defaultValue: 30,
    tooltip: "Generate Ring refuses to start below this floor. The measured RCP " +
             "peak on a 32 GB card is 29.0 GB reserved.",
  },
  {
    id: "cumuli.fdanyone_root",
    name: "4DAnyone checkout",
    type: "text",
    defaultValue: "",
    tooltip: "Path to the 4DAnyone repository. Empty keeps the built-in default.",
  },
  {
    id: "cumuli.trainer_root",
    name: "OMG4 trainer checkout",
    type: "text",
    defaultValue: "",
    tooltip: "Path to the OMG4 rotor-4DGS trainer. Empty keeps the built-in default.",
  },
];

app.registerExtension({
  name: "cumuli.settings",
  settings: SETTINGS.map((s) => ({ ...s, category: ["Cumuli", "Bridge", s.name] })),
});
