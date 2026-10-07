/* Preload bridge.
 *
 * The UI is a plain web app that talks to the LM3 server over HTTP, so it needs almost nothing from
 * Electron. Only a few read-only facts are exposed, and contextIsolation stays on -- there is no
 * reason to hand the renderer Node access.
 *
 * Nothing here is a capability over a run or a server. `quit` closes the WINDOW; per invariant 9 a
 * running LM3 job survives that, and per invariant 10 a server this app did not start survives it
 * too. Stopping a run is a separate, explicit action in the UI.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("lm3desktop", {
  isElectron: true,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
  },
  /** Open a produced file (image, timing report, STL) in the OS default application. */
  openExternal: (url) => ipcRenderer.invoke("lm3:open-external", String(url)),
  /** Which deployment this window belongs to, and whether this app started the server.
   *  Descriptive only (section 2.1 / invariant 12): none of it authorizes an action. */
  identity: () => ipcRenderer.invoke("lm3:identity"),
  /** Close the desktop app. `running` only shapes the confirmation text -- the job is not stopped
   *  here, and the UI must no longer stop it before calling (invariant 9). */
  quit: (opts) => ipcRenderer.invoke("lm3:quit", opts || {}),
});
