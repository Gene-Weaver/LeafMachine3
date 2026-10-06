/**
 * Models tab — the full "Models from Hugging Face" card (every module, its install state and the
 * install button) at full width. The Live Status tab keeps the compact banner that only appears
 * while a model is missing or outdated; both are the same panel from ../models.js.
 */
import { el } from "../api.js";
import { mountModelsPanel, refreshModelsStatus } from "../models.js";

export function initModelsTab(root) {
  if (!root) throw new Error("initModelsTab(root): no mount element");
  root.replaceChildren();
  const host = el("div", { style: { maxWidth: "none" } });
  root.appendChild(host);
  const panel = mountModelsPanel(host, { compact: false, fullWidth: true });
  refreshModelsStatus();                       // always re-read when the tab is opened
  return { refresh: refreshModelsStatus, panel };
}
