/* ==========================================================================
   LM3 — THE 17 MODULES, once.
   --------------------------------------------------------------------------
   Canonical order + device/exec facts, mirroring pipeline.STAGE_ORDER and
   progress_api.module_table(). Held statically HERE, in its own module, because
   TWO surfaces render this list and they must never disagree:

     * topbar.js  — the global stage bar, in RUN order (`order`)
     * tabs/settings.js — the settings rail, in READING order (`railOrder`),
       grouped by `phase`

   It is static because the status snapshot returns `modules: []` until a run has
   touched the ledger, and an empty track above the tabs would tell a first-time
   user nothing. Live data is merged over this table by key, and any module the
   server reports that we do not know about is appended rather than dropped.

   `short` exists because 17 full names do not fit across a 1400 px window at
   10.5 px; the full name is always in the tooltip.

   ADDING A MODULE is three edits that a test enforces (tests/test_settings_ui.py):
   an entry here, an entry in pipeline.STAGE_ORDER, and a `_sections` entry in
   settings_meta.json carrying the matching `stage`. Bilateral Symmetry shipped
   with only the first two for months and its five settings silently filed
   themselves under "Project" — that is the failure this table plus that test
   exists to make impossible.
   ========================================================================== */

/**
 * Reading-order phases for the settings rail.
 *
 * NOT strictly run order: ECT is stage 17 but is a leaf-shape measurement, so it
 * reads under LEAF next to the other per-leaf modules rather than orphaned after
 * the Reporter. Every entry keeps its true `order`, and the rail prints it as a
 * badge, so the run-order fact survives the regrouping.
 */
export const PHASES = [
  { id: "setup", label: "Setup", blurb: "Where the images come from, where the output goes, and what the machine may use." },
  { id: "sheet", label: "Sheet", blurb: "Everything read off the whole herbarium sheet." },
  { id: "scale", label: "Scale", blurb: "Turning pixels into centimeters." },
  { id: "leaf", label: "Leaf", blurb: "Per-leaf segmentation, landmarks and shape." },
  { id: "output", label: "Output", blurb: "Grounding the measurements and writing the files." },
];

export const MODULES = [
  { key: "mp_conversion_factor",  name: "MP Conversion Factor",    short: "MP CF",       order: 1,  phase: "sheet",  railOrder: 1,  device: "cpu", exec: "thread",  depends: [] },
  { key: "archival_detector",     name: "Archival Detector",       short: "Archival",    order: 2,  phase: "sheet",  railOrder: 2,  device: "gpu", exec: "gpu",     depends: [] },
  { key: "plant_detector",        name: "Plant Detector",          short: "Plant",       order: 3,  phase: "sheet",  railOrder: 3,  device: "gpu", exec: "gpu",     depends: [] },
  { key: "specimen_segmenter",    name: "Specimen Segmenter",      short: "Specimen",    order: 4,  phase: "sheet",  railOrder: 4,  device: "gpu", exec: "gpu",     depends: [] },
  { key: "phenology_detector",    name: "Phenology Detector",      short: "Phenology",   order: 5,  phase: "sheet",  railOrder: 5,  device: "cpu", exec: "thread",  depends: ["plant_detector"] },
  { key: "ruler_classifier",      name: "Ruler Classifier",        short: "Ruler Cls",   order: 6,  phase: "scale",  railOrder: 6,  device: "gpu", exec: "gpu",     depends: ["archival_detector"] },
  { key: "ruler_cf",              name: "Ruler Conversion Factor", short: "Ruler CF",    order: 7,  phase: "scale",  railOrder: 7,  device: "cpu", exec: "process", depends: ["archival_detector", "ruler_classifier"] },
  { key: "leaf_segmenter",        name: "Leaf Segmenter",          short: "Leaf Seg",    order: 8,  phase: "leaf",   railOrder: 8,  device: "gpu", exec: "gpu",     depends: ["plant_detector"] },
  { key: "morphology",            name: "Morphology",              short: "Morphology",  order: 9,  phase: "leaf",   railOrder: 9,  device: "cpu", exec: "thread",  depends: ["leaf_segmenter"] },
  { key: "landmark_detector",     name: "Landmark Detector",       short: "Landmarks",   order: 10, phase: "leaf",   railOrder: 10, device: "gpu", exec: "gpu",     depends: ["plant_detector"] },
  { key: "landmark_measurements", name: "Landmark Measurements",   short: "LM Measure",  order: 11, phase: "leaf",   railOrder: 11, device: "cpu", exec: "thread",  depends: ["landmark_detector"] },
  { key: "leaf_orientation",      name: "Leaf Orientation",        short: "Orientation", order: 12, phase: "leaf",   railOrder: 12, device: "cpu", exec: "thread",  depends: ["morphology", "landmark_detector"] },
  { key: "petiole_width",         name: "Petiole Width",           short: "Petiole",     order: 13, phase: "leaf",   railOrder: 13, device: "cpu", exec: "thread",  depends: ["leaf_segmenter", "landmark_detector", "morphology"] },
  { key: "bilateral_symmetry",    name: "Bilateral Symmetry",      short: "Symmetry",    order: 14, phase: "leaf",   railOrder: 14, device: "cpu", exec: "process", depends: ["leaf_segmenter", "landmark_detector", "morphology", "leaf_orientation"] },
  { key: "ect",                   name: "Shape (ECT)",             short: "ECT",         order: 17, phase: "leaf",   railOrder: 15, device: "cpu", exec: "process", depends: ["reporter"] },
  { key: "metric_grounding",      name: "Metric Grounding",        short: "Grounding",   order: 15, phase: "output", railOrder: 16, device: "cpu", exec: "thread",  depends: ["ruler_cf", "leaf_segmenter", "petiole_width"] },
  { key: "reporter",              name: "Reporter",                short: "Reporter",    order: 16, phase: "output", railOrder: 17, device: "cpu", exec: "thread",  depends: ["archival_detector", "plant_detector", "specimen_segmenter", "phenology_detector", "ruler_classifier", "ruler_cf", "leaf_segmenter", "morphology", "landmark_detector", "landmark_measurements", "leaf_orientation", "petiole_width", "metric_grounding"] },
];

/** key -> module record. */
export const MODULE_BY_KEY = new Map(MODULES.map((m) => [m.key, m]));

/** The stage bar's order: exactly pipeline.STAGE_ORDER. */
export const MODULES_BY_RUN_ORDER = MODULES.slice().sort((a, b) => a.order - b.order);

/** The settings rail's order: phase-grouped, ECT read with the leaf modules. */
export const MODULES_BY_RAIL_ORDER = MODULES.slice().sort((a, b) => a.railOrder - b.railOrder);

export default MODULES;
