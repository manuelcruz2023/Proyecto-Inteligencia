(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = {
    video: $("video"), overlay: $("overlay"), capture: $("capture"), wrap: $("video-wrap"),
    msg: $("video-msg"), banner: $("alarm-banner"),
    start: $("btn-start"), stop: $("btn-stop"), silence: $("btn-silence"), status: $("status-line"),
    stateLabel: $("state-label"), stateDesc: $("state-desc"),
    warm: $("warmup"), warmBar: $("warmup-bar"), warmTxt: $("warmup-text"), timer: $("drowsy-timer"),
    probList: $("prob-list"), latency: $("latency"), log: $("event-log"),
    secs: $("opt-seconds"), secsVal: $("opt-seconds-val"),
    thr: $("opt-threshold"), thrVal: $("opt-threshold-val"),
    vol: $("opt-volume"), volVal: $("opt-volume-val"),
    modelBanner: $("model-banner"), modelBadge: $("model-badge"),
  };

  const STATE_TEXT = {
    "Alert": "Conductor atento.",
    "Low Vigilance": "Vigilancia reducida: posibles signos tempranos de fatiga.",
    "Drowsy": "Somnolencia: riesgo elevado. Considere detenerse a descansar.",
  };
  const CAPTURE_WIDTH = 320, JPEG_QUALITY = 0.8;

  let cfg = { class_names: [], sample_fps: 2, seq_len: 16 };
  let stream = null, running = false, timerId = null;
  let sessionId = newId();
  let drowsySince = null, alarmActive = false, silenced = false, lastLabel = null, busy = false;

  // ---------------------------- Utilidades ----------------------------
  function newId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "s-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
  function setStatus(text, isError = false) {
    el.status.textContent = text;
    el.status.classList.toggle("error", isError);
  }
  function log(text, isAlarm = false) {
    const li = document.createElement("li");
    li.textContent = `${new Date().toLocaleTimeString()} · ${text}`;
    if (isAlarm) li.classList.add("alarm");
    el.log.prepend(li);
    while (el.log.children.length > 50) el.log.removeChild(el.log.lastChild);
  }
  const fillClass = (name) => (name === "Low Vigilance" ? "Low" : name);

  // ---------------------------- Alarma sonora (Web Audio) ----------------------------
  const alarm = {
    ctx: null, osc: null, gain: null, timer: null, active: false,
    unlock() {   // debe llamarse dentro de un gesto del usuario (clic)
      if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.ctx.state === "suspended") this.ctx.resume();
    },
    start() {
      if (this.active || !this.ctx) return;
      this.osc = this.ctx.createOscillator();
      this.gain = this.ctx.createGain();
      this.osc.type = "square";
      this.osc.frequency.value = 880;
      this.gain.gain.value = 0;
      this.osc.connect(this.gain).connect(this.ctx.destination);
      this.osc.start();
      let on = false;
      const beep = () => {
        on = !on;
        const v = (Number(el.vol.value) / 100) * 0.5;
        this.gain.gain.setTargetAtTime(on ? v : 0, this.ctx.currentTime, 0.01);
        this.osc.frequency.setValueAtTime(on ? 880 : 660, this.ctx.currentTime);
      };
      beep();
      this.timer = setInterval(beep, 350);
      this.active = true;
    },
    stop() {
      if (!this.active) return;
      clearInterval(this.timer);
      try { this.osc.stop(); } catch (_) {}
      this.osc.disconnect(); this.gain.disconnect();
      this.active = false;
    },
  };

  function triggerAlarm() {
    if (alarmActive) return;
    alarmActive = true;
    document.body.classList.add("alarm-on");
    el.banner.classList.remove("hidden");
    el.silence.disabled = false;
    if (!silenced) alarm.start();
    log("ALARMA: somnolencia sostenida detectada", true);
  }
  function clearAlarm() {
    if (!alarmActive) return;
    alarmActive = false; silenced = false;
    alarm.stop();
    document.body.classList.remove("alarm-on");
    el.banner.classList.add("hidden");
    el.silence.disabled = true;
    log("Alarma desactivada (el estado ya no es Drowsy)");
  }

  function evaluateAlarm(result, now) {
    const secs = Number(el.secs.value), thr = Number(el.thr.value) / 100;
    const isDrowsy = result.ready && result.label === "Drowsy" && result.probs["Drowsy"] >= thr;
    if (isDrowsy) {
      if (drowsySince === null) drowsySince = now;
      const elapsed = (now - drowsySince) / 1000;
      el.timer.textContent = `${elapsed.toFixed(1)} s / ${secs} s`;
      if (elapsed >= secs) triggerAlarm();
    } else {
      drowsySince = null;
      el.timer.textContent = "—";
      clearAlarm();
    }
  }

  // ---------------------------- UI de resultados ----------------------------
  function buildProbBars() {
    el.probList.innerHTML = "";
    cfg.class_names.forEach((name) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `<span>${name}</span><div class="bar"><div class="fill ${fillClass(name)}" data-class="${name}"></div></div><span class="pct" data-pct="${name}">0%</span>`;
      el.probList.appendChild(row);
    });
  }
  function setProbs(probs) {
    cfg.class_names.forEach((name) => {
      const p = probs ? probs[name] : 0;
      el.probList.querySelector(`[data-class="${name}"]`).style.width = `${(p * 100).toFixed(1)}%`;
      el.probList.querySelector(`[data-pct="${name}"]`).textContent = `${(p * 100).toFixed(0)}%`;
    });
  }
  function drawBox(result) {
    const c = el.overlay, ctx = c.getContext("2d");
    c.width = el.video.clientWidth; c.height = el.video.clientHeight;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!result.bbox) return;
    const [fw, fh] = result.frame_size;
    const sx = c.width / fw, sy = c.height / fh;
    const [x, y, w, h] = result.bbox;
    const colors = { "Alert": "#2ea043", "Low Vigilance": "#f0a500", "Drowsy": "#f04747" };
    ctx.lineWidth = 3;
    ctx.strokeStyle = result.ready ? colors[result.label] : "#ffffff";
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
  }

  function handleResult(result) {
    const now = performance.now();
    drawBox(result);
    el.latency.textContent = result.latency_ms != null ? `${result.latency_ms} ms` : "—";

    if (!result.face_detected) {
      el.stateLabel.dataset.state = "idle";
      el.stateLabel.textContent = "Sin rostro";
      el.stateDesc.textContent = "Colóquese frente a la cámara con buena iluminación.";
      el.warm.classList.add("hidden");
      setProbs(null);
      evaluateAlarm({ ready: false }, now);
      lastLabel = null;
      return;
    }
    if (!result.ready) {
      el.stateLabel.dataset.state = "idle";
      el.stateLabel.textContent = "Analizando…";
      el.stateDesc.textContent = "Recopilando la secuencia temporal inicial.";
      el.warm.classList.remove("hidden");
      el.warmBar.style.width = `${(result.progress * 100).toFixed(0)}%`;
      el.warmTxt.textContent = `Frames: ${Math.round(result.progress * cfg.seq_len)}/${cfg.seq_len} (≈ ${(cfg.seq_len / cfg.sample_fps).toFixed(0)} s)`;
      evaluateAlarm({ ready: false }, now);
      return;
    }
    el.warm.classList.add("hidden");
    el.stateLabel.dataset.state = result.label;
    el.stateLabel.textContent = `${result.label} · ${(result.confidence * 100).toFixed(0)}%`;
    el.stateDesc.textContent = STATE_TEXT[result.label] || "";
    setProbs(result.probs);
    if (result.label !== lastLabel) {
      log(`Estado: ${result.label} (${(result.confidence * 100).toFixed(0)}%)`);
      lastLabel = result.label;
    }
    evaluateAlarm(result, now);
  }

  // ---------------------------- Bucle de captura ----------------------------
  async function tick() {
    if (!running) return;
    const t0 = performance.now();
    if (!busy && el.video.videoWidth) {
      busy = true;
      try {
        const vw = el.video.videoWidth, vh = el.video.videoHeight;
        el.capture.width = CAPTURE_WIDTH;
        el.capture.height = Math.round(CAPTURE_WIDTH * vh / vw);
        el.capture.getContext("2d").drawImage(el.video, 0, 0, el.capture.width, el.capture.height);
        const image = el.capture.toDataURL("image/jpeg", JPEG_QUALITY);
        const resp = await fetch("/api/predict", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, image }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        if (running) { handleResult(data); setStatus("Monitoreo activo."); }
      } catch (err) {
        setStatus(`Error de comunicación: ${err.message}`, true);
      } finally { busy = false; }
    }
    const wait = Math.max(0, 1000 / cfg.sample_fps - (performance.now() - t0));
    timerId = setTimeout(tick, wait);
  }

  async function start() {
    alarm.unlock();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
    } catch (err) {
      setStatus(`No se pudo acceder a la cámara: ${err.message}`, true);
      return;
    }
    el.video.srcObject = stream;
    await el.video.play();
    el.wrap.style.aspectRatio = `${el.video.videoWidth} / ${el.video.videoHeight}`;
    el.msg.classList.add("hidden");
    sessionId = newId();
    running = true; drowsySince = null; lastLabel = null;
    el.start.disabled = true; el.stop.disabled = false;
    log("Monitoreo iniciado");
    setStatus("Iniciando análisis…");
    tick();
  }

  function stop() {
    running = false;
    clearTimeout(timerId);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; el.video.srcObject = null;
    clearAlarm();
    el.overlay.getContext("2d").clearRect(0, 0, el.overlay.width, el.overlay.height);
    el.msg.classList.remove("hidden");
    el.start.disabled = false; el.stop.disabled = true;
    el.stateLabel.dataset.state = "idle"; el.stateLabel.textContent = "En espera";
    el.stateDesc.textContent = "Monitoreo detenido."; el.timer.textContent = "—";
    el.warm.classList.add("hidden"); setProbs(null);
    fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ session_id: sessionId }), keepalive: true }).catch(() => {});
    log("Monitoreo detenido");
    setStatus("Monitoreo detenido.");
  }

  // ---------------------------- Inicialización ----------------------------
  [["secs", "secsVal"], ["thr", "thrVal"], ["vol", "volVal"]].forEach(([i, o]) =>
    el[i].addEventListener("input", () => { el[o].textContent = el[i].value; }));
  el.start.addEventListener("click", start);
  el.stop.addEventListener("click", stop);
  el.silence.addEventListener("click", () => { silenced = true; alarm.stop(); log("Alarma sonora silenciada"); });
  window.addEventListener("beforeunload", () => { if (stream) stream.getTracks().forEach((t) => t.stop()); });

  (async function init() {
    try {
      const h = await (await fetch("/api/health")).json();
      if (!h.model_loaded) {
        el.modelBanner.textContent = `Modelo no disponible: ${h.error || "entrene el modelo (scripts/03_train.py)."}`;
        el.modelBanner.classList.remove("hidden");
        el.modelBadge.textContent = "Modelo: no cargado"; el.modelBadge.classList.add("err");
        return;
      }
      cfg = h;
      el.modelBadge.textContent = "Modelo: CNN+LSTM cargado"; el.modelBadge.classList.add("ok");
      buildProbBars();
      el.start.disabled = false;
    } catch (err) {
      setStatus(`No se pudo contactar con el servidor: ${err.message}`, true);
    }
  })();
})();