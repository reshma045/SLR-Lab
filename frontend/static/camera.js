// Webcam + WebSocket client shared by the Live, Record and Self-test pages.
// Usage: const cam = new LiveCamera({stage, page: "live", onResult(msg){...}, onStatus(s){...}}); cam.start();
class LiveCamera {
  constructor({ stage, page, onResult = () => {}, onStatus = () => {} }) {
    this.stage = stage; this.page = page; this.onResult = onResult; this.onStatus = onStatus;
    this.video = stage.querySelector("video"); this.img = stage.querySelector("img");
    this.msg = stage.querySelector(".overlay.cam"); this.recon = stage.querySelector(".overlay.recon");
    this.canvas = document.createElement("canvas"); this.canvas.width = 640; this.canvas.height = 480;
    this.ctx = this.canvas.getContext("2d");
    this.ws = null; this.attempt = 0; this.inflight = false; this.lastSent = 0; this.sentAt = 0;
    this.fps = 18; this.quality = 0.7; this.replies = [];
    this.backoff = [1, 2, 4, 8];
  }

  async start() {
    try {
      const s = (await JSL.settings()).settings;
      this.fps = s.camera.send_fps; this.quality = s.camera.jpeg_quality;
      this.stage.classList.toggle("mirror", !!s.camera.mirror);
    } catch (e) {}
    this.connect();
    await this.openCamera();
    requestAnimationFrame(t => this.loop(t));
  }

  async openCamera() {
    let deviceId = null;
    try { deviceId = localStorage.getItem("jsl.camera"); } catch (e) {}
    const video = deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 };
    try {
      let stream;
      try { stream = await navigator.mediaDevices.getUserMedia({ video, audio: false }); }
      catch (e) { if (deviceId) stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false }); else throw e; }
      this.video.srcObject = stream; await this.video.play();
      this.msg && this.msg.classList.remove("show");
      this.status();
    } catch (err) {
      if (!this.msg) return;
      this.msg.innerHTML = "";
      const s = document.createElement("strong"); s.textContent = "Camera unavailable";
      const t = document.createElement("span");
      t.textContent = err && err.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access for this site in the address bar, then reload."
        : `No camera could be opened (${(err && err.name) || "error"}). Check Settings → Camera, or close other apps using it.`;
      this.msg.append(s, t); this.msg.classList.add("show");
    }
  }

  connect() {
    this.status("connecting");
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = this.ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => {
      this.attempt = 0; this.inflight = false; ws.send(JSON.stringify({ type: "hello", page: this.page }));
      this.recon && this.recon.classList.remove("show"); this.status("connected");
    };
    ws.onmessage = ev => {
      this.inflight = false;
      const now = performance.now(); this.replies.push(now);
      while (this.replies.length && now - this.replies[0] > 1000) this.replies.shift();
      const msg = JSON.parse(ev.data);
      if (msg.type === "result" && msg.frame && this.img) this.img.src = "data:image/jpeg;base64," + msg.frame;
      this.onResult(msg);
    };
    ws.onclose = () => {
      this.status("disconnected"); this.inflight = false;
      const wait = this.backoff[Math.min(this.attempt, this.backoff.length - 1)]; this.attempt++;
      if (this.recon) { this.recon.querySelector("span").textContent = `Retrying in ${wait}s (attempt ${this.attempt})`; this.recon.classList.add("show"); }
      setTimeout(() => this.connect(), wait * 1000);
    };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }

  send(obj) { if (this.ws && this.ws.readyState === 1) { this.ws.send(JSON.stringify(obj)); return true; } return false; }
  get processedFps() { return this.replies.length; }
  status(ws) { if (ws) this.wsState = ws; this.onStatus({ ws: this.wsState, camera: !!(this.video && this.video.videoWidth) }); }

  loop(now) {
    requestAnimationFrame(t => this.loop(t));
    if (!this.ws || this.ws.readyState !== 1 || !this.video.videoWidth) return;
    if (this.inflight && now - this.sentAt < 2000) return;       // adaptive: wait for the previous reply
    if (now - this.lastSent < 1000 / this.fps) return;
    this.ctx.drawImage(this.video, 0, 0, 640, 480);
    const data = this.canvas.toDataURL("image/jpeg", this.quality).split(",")[1];
    this.ws.send(JSON.stringify({ type: "frame", data }));
    this.inflight = true; this.lastSent = this.sentAt = now;
  }
}
window.LiveCamera = LiveCamera;
