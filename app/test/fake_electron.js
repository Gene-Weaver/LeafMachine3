/* An injectable stand-in for the `electron` module, so app/main.js's boot() can be exercised in
 * plain Node.
 *
 * There is no way to assert "requestSingleInstanceLock happens before IPC registration and before
 * any server is contacted" from the outside of a running GUI, and that ORDER is exactly what
 * section 2.11 requires. So the order is recorded here, by a double that answers the same calls the
 * real module does. A GUI is never launched.
 */
const path = require("node:path");
const Module = require("node:module");

/** Put `exports` in front of `require("electron")` for anything resolving from `fromDir`. */
function installFakeElectron(fromDir, exports) {
  const resolved = Module.createRequire(path.join(fromDir, "main.js")).resolve("electron");
  require.cache[resolved] = {
    id: resolved, filename: resolved, path: path.dirname(resolved),
    loaded: true, children: [], paths: [], exports,
  };
  return () => { delete require.cache[resolved]; };
}

function makeFakeElectron({ lock = true, appData = "/tmp/lm3-fake-appdata", messageBoxResponse = 1 } = {}) {
  const rec = {
    order: [],            // the sequence that section 2.11 constrains
    setPath: [],
    lockData: null,
    lockCalls: 0,
    handlers: {},         // ipcMain channel -> handler
    listeners: {},        // app event -> [handler]
    quitCalls: 0,
    exitCodes: [],
    errorBoxes: [],
    windows: [],
    openedExternal: [],
  };
  let resolveReady;
  const readyPromise = new Promise((r) => { resolveReady = r; });

  class BrowserWindow {
    constructor(opts) {
      this.opts = opts || {};
      this.destroyed = false;
      this.minimized = false;
      this.shown = false;
      this.focused = false;
      this.loadedURL = null;
      this.webContents = { setWindowOpenHandler: (fn) => { this.openHandler = fn; },
                           toggleDevTools: () => {} };
      rec.windows.push(this);
      BrowserWindow._all.push(this);
    }
    once(event, fn) { if (event === "ready-to-show") this._readyToShow = fn; }
    async loadURL(url) { this.loadedURL = url; }
    isDestroyed() { return this.destroyed; }
    isMinimized() { return this.minimized; }
    restore() { this.minimized = false; }
    show() { this.shown = true; }
    focus() { this.focused = true; }
    hide() { this.shown = false; }
    reload() {}
  }
  BrowserWindow._all = [];
  BrowserWindow.getAllWindows = () => BrowserWindow._all.filter((w) => !w.destroyed);

  const electron = {
    app: {
      isPackaged: false,
      isQuiting: false,
      getPath: (key) => path.join(appData, key),
      setPath: (key, value) => { rec.setPath.push([key, value]); },
      requestSingleInstanceLock: (data) => {
        rec.order.push("requestSingleInstanceLock");
        rec.lockCalls += 1;
        rec.lockData = data;
        return lock;
      },
      whenReady: () => { rec.order.push("whenReady"); return readyPromise; },
      on: (event, fn) => { (rec.listeners[event] = rec.listeners[event] || []).push(fn); },
      quit: () => { rec.quitCalls += 1; },
      exit: (code) => { rec.exitCodes.push(code); },
    },
    BrowserWindow,
    Menu: { setApplicationMenu: () => {}, buildFromTemplate: (t) => t },
    shell: {
      openExternal: (u) => { rec.openedExternal.push(u); return true; },
      openPath: (p) => { rec.openedExternal.push(p); return ""; },
    },
    dialog: {
      showErrorBox: (title, body) => { rec.errorBoxes.push([title, body]); },
      showMessageBox: async () => ({ response: messageBoxResponse }),
    },
    ipcMain: {
      handle: (channel, fn) => { rec.order.push(`ipc:${channel}`); rec.handlers[channel] = fn; },
    },
  };

  return { electron, rec, ready: () => { resolveReady(); return readyPromise; } };
}

module.exports = { installFakeElectron, makeFakeElectron };
