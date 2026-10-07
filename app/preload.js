/* Preload bridge.
 *
 * The UI is a plain web app that talks to the LM3 server over HTTP, so it needs almost nothing from
 * Electron. Only a couple of read-only facts are exposed, and contextIsolation stays on -- there is
 * no reason to hand the renderer Node access.
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
  /** Quit the desktop app. The UI stops any in-flight LM3 run BEFORE calling this. */
  quit: (opts) => ipcRenderer.invoke("lm3:quit", opts || {}),
});
