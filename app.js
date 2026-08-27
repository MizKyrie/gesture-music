import { HandLandmarker, FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ============================================================
//  DOM 引用
// ============================================================
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");

const chordDisplayEl = document.getElementById("chordDisplay");
const qualityDisplayEl = document.getElementById("qualityDisplay");
const filterDisplayEl = document.getElementById("filterDisplay");
const startOverlayEl = document.getElementById("startOverlay");
const gestureGuideEl = document.getElementById("gestureGuide");
const helpButton = document.getElementById("helpButton");
const helpModal = document.getElementById("helpModal");
const closeHelp = document.getElementById("closeHelp");
const pianoEl = document.getElementById("piano");
const volumeSlider = document.getElementById("volumeSlider");
const degreeBtns = Array.from(document.querySelectorAll(".degree-btn"));
const accidentalHintEl = document.getElementById("accidentalHint");
const headVolumeToggleEl = document.getElementById("headVolumeToggle");
const headVolumeMeterEl = document.getElementById("headVolumeMeter");
const volumeBarFillEl = document.getElementById("volumeBarFill");

// 头部音量控制状态
let headVolumeEnabled = false;   // 开关
let faceWidthSmooth = null;      // 平滑后的脸宽（归一化）
let lastAppliedVolume = -1;      // 上次应用的音量（避免每帧 ramp）

// 就近衔接（voice leading）状态
let voiceLeadEnabled = false;    // 开关：切换和弦时各声部就近移动
let lastChordFreqs = null;       // 上一次实际演奏的频率（作为声部锚点）

// ============================================================
//  手指关键点（MediaPipe Hands 21 点）
//  伸直判断：回到最初准确版 —— 指尖在 PIP 上方（tip.y < pip.y）
//  这是手心朝摄像头时最准的判断；6 级（手指朝下）单独用 isFingerExtendedDown
// ============================================================
const FINGERS = {
  index:  { mcp: 5,  pip: 6,  tip: 8  },
  middle: { mcp: 9,  pip: 10, tip: 12 },
  ring:   { mcp: 13, pip: 14, tip: 16 },
  pinky:  { mcp: 17, pip: 18, tip: 20 },
};

function isFingerExtended(landmarks, name) {
  const { pip, tip } = FINGERS[name];
  return landmarks[tip].y < landmarks[pip].y;
}
// 朝下伸直（用于 6 级）：指尖在 PIP 下方 + 指尖到手腕距离 > PIP 到手腕 × 1.2（伸直非弯曲）
function isFingerExtendedDown(landmarks, name) {
  const { pip, tip } = FINGERS[name];
  if (landmarks[tip].y <= landmarks[pip].y) return false;
  const w = landmarks[0];
  const dTip = Math.hypot(landmarks[tip].x - w.x, landmarks[tip].y - w.y);
  const dPip = Math.hypot(landmarks[pip].x - w.x, landmarks[pip].y - w.y);
  return dTip > dPip * 1.2;
}
// 拇指伸展判断：带滞回（左右手独立状态）
// 参考点取中指根部（手掌中心）：拇指弯曲时贴掌心 → 距离小；伸出时远离 → 距离大
// 滞回：收→伸需 > TH_OPEN，伸→收需 < TH_CLOSE，中间区间保持上一状态 → 吃掉临界抖动
const THUMB_OPEN = 0.085, THUMB_CLOSE = 0.060;
const thumbStates = { left: false, right: false };
function isThumbExtended(landmarks, side) {
  const t = landmarks[4], palm = landmarks[9];
  const d = Math.hypot(t.x - palm.x, t.y - palm.y);
  const st = thumbStates[side] ?? false;
  if (st && d > THUMB_CLOSE) return true;       // 已伸：跌过下限才收
  if (!st && d < THUMB_OPEN) return false;      // 已收：涨过上限才伸
  thumbStates[side] = d > (THUMB_OPEN + THUMB_CLOSE) / 2;
  return thumbStates[side];
}
function allFingersUp(landmarks, side) {
  return isFingerExtended(landmarks, "index") && isFingerExtended(landmarks, "middle")
      && isFingerExtended(landmarks, "ring") && isFingerExtended(landmarks, "pinky")
      && isThumbExtended(landmarks, side);
}
function fourFingersDown(landmarks) {
  return isFingerExtendedDown(landmarks, "index") && isFingerExtendedDown(landmarks, "middle")
      && isFingerExtendedDown(landmarks, "ring") && isFingerExtendedDown(landmarks, "pinky");
}

// ============================================================
//  调性 / 音名
// ============================================================
const CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const KEY_PC = { A:9, Bb:10, B:11, C:0, Db:1, D:2, Eb:3, E:4, F:5, Gb:6, G:7, Ab:8 };
const DEGREE_SEMITONES = { 1:0, 2:2, 3:4, 4:5, 5:7, 6:9, 7:11 };

const keySelectEl = document.getElementById("keySelect");
let currentTonicFreq = Number(keySelectEl.value);
let currentKeyName = keySelectEl.selectedOptions[0].dataset.note;

const toneSelectEl = document.getElementById("toneSelect");
// ============================================================
//  音色库
//  BASE_TONES：内置基础音色（type=基础波形；custom=自定义谐波 PeriodicWave；dual=双振荡器 detune）
//  userTones：用户音色（基于 base 调效果器参数，localStorage 持久化）
//  效果器参数 FX：低通截止/共振、起音/释音包络、延迟、混响
// ============================================================
const FX_DEFAULTS = { cutoff: 2400, q: 0.7, lowGain: 0, midGain: 0, highGain: 0, attack: 0.09, release: 0.06, delayTime: 0, delayFb: 0, reverb: 0 };
const BASE_TONES = {
  triangle: { name: "暖色合成 Warm", type: "triangle", fx: { cutoff: 1900, q: 0.8, attack: 0.10, release: 0.10, reverb: 0.10 } },
  sine:     { name: "纯净 Sine",     type: "sine",     fx: { cutoff: 5000, q: 0.4, attack: 0.06, release: 0.08, reverb: 0.12 } },
  sawtooth: { name: "明亮 Bright",   type: "sawtooth", fx: { cutoff: 3600, q: 0.9, attack: 0.02, release: 0.09, highGain: 2 } },
  square:   { name: "复古 Retro",    type: "square",   fx: { cutoff: 2100, q: 1.2, attack: 0.03, release: 0.08, lowGain: 2, highGain: -1 } },
  bell:     { name: "铃音 Bell",     type: "custom", harmonics: [1, 0.28, 0.12, 0.07, 0.04, 0.02], fx: { cutoff: 8000, q: 0.6, attack: 0.01, release: 0.35, reverb: 0.30, highGain: 2 } },
  organ:    { name: "风琴 Organ",    type: "custom", harmonics: [1, 0.5, 0.33, 0.25, 0.2, 0.15], fx: { cutoff: 4200, q: 1.1, attack: 0.02, release: 0.05, midGain: 2 } },
  pad:      { name: "铺底 Pad",      type: "sawtooth", dual: true, detune: 12, fx: { cutoff: 3200, q: 0.7, attack: 0.28, release: 0.32, reverb: 0.32, lowGain: 2, highGain: 1 } },
};
let userTones = [];
try { const raw = localStorage.getItem("gs_tones_v1"); if (raw) userTones = JSON.parse(raw); } catch {}
function saveUserTones() { try { localStorage.setItem("gs_tones_v1", JSON.stringify(userTones)); } catch {} }
let currentTone = toneSelectEl.value;   // "triangle" 或 "user:<id>"
function getToneDef(id) {
  if (id && id.startsWith("user:")) {
    const u = userTones.find((t) => t.id === id.slice(5));
    // 用户音色继承基础波形的默认 FX（若有），再覆盖用户保存的 FX
    if (u) return { ...BASE_TONES[u.base], base: u.base, name: u.name, fx: { ...FX_DEFAULTS, ...(BASE_TONES[u.base]?.fx || {}), ...u.fx } };
  }
  const key = id in BASE_TONES ? id : "triangle";
  const b = BASE_TONES[key];
  return { ...b, base: key, fx: { ...FX_DEFAULTS, ...(b.fx || {}) } };
}
let currentToneDef = getToneDef(currentTone);

// 立体声 Tremolo（音量颤音）参数：LFO 调制主输出音量，左右声道反相 → 音量波动 + 左右交替
let tremoloOn = false;      // 开关
let tremoloDepth = 0.5;     // 深度（0~1，调制幅度）
let tremoloRate = 6;        // 速率（Hz，0.5~12）

// 主控三段 EQ（设置面板滑杆，Roland 风格 low/mid/high；独立于音色编辑器里保存的 tone EQ）
let masterEQ = { lowGain: 0, midGain: 0, highGain: 0 };

// ============================================================
//  6 个和弦家族（角度=家族，半径=形态；环数 = 该家族形态数，可自定义增删）
//  ring: { label, intervals, voicing: "close"|"drop2"|"drop3"|"drop24" }
//  编排：Maj(Maj/sus4/drop2) · Maj7(Maj7/Maj9·d2/Maj9·d3) · 7(7/7sus4/13)
//        Min(Min/d2/d3) · m7(m7/m9·d2/m9·d3) · ø7(ø7/d2/d3)
//  13 用 1-3-b7-13 四音 voicing（省略 5、9，吉他常用配置）
// ============================================================
const DEFAULT_CHORD_LIB = {
  maj: {
    label: "Maj", color: "255,107,107",
    rings: [
      { label: "Maj",  intervals: [0, 4, 7],     voicing: "close" },
      { label: "sus4", intervals: [0, 5, 7],     voicing: "close" },
      { label: "Maj",  intervals: [0, 4, 7],     voicing: "drop2" },
    ],
  },
  maj7: {
    label: "Maj7", color: "69,183,209",
    rings: [
      { label: "Maj7", intervals: [0, 4, 7, 11],     voicing: "close" },
      { label: "Maj9", intervals: [0, 4, 7, 11, 14], voicing: "drop2" },
      { label: "Maj9", intervals: [0, 4, 7, 11, 14], voicing: "drop3" },
    ],
  },
  dom7: {
    label: "7", color: "150,206,180",
    rings: [
      { label: "7",     intervals: [0, 4, 7, 10],  voicing: "close" },
      { label: "7sus4", intervals: [0, 5, 7, 10],  voicing: "close" },
      { label: "13",    intervals: [0, 4, 10, 21], voicing: "close" },
    ],
  },
  min: {
    label: "Min", color: "78,205,196",
    rings: [
      { label: "Min", intervals: [0, 3, 7], voicing: "close" },
      { label: "Min", intervals: [0, 3, 7], voicing: "drop2" },
      { label: "Min", intervals: [0, 3, 7], voicing: "drop3" },
    ],
  },
  m7: {
    label: "m7", color: "192,132,252",
    rings: [
      { label: "m7", intervals: [0, 3, 7, 10],     voicing: "close" },
      { label: "m9", intervals: [0, 3, 7, 10, 14], voicing: "drop2" },
      { label: "m9", intervals: [0, 3, 7, 10, 14], voicing: "drop3" },
    ],
  },
  m7b5: {
    label: "ø7", color: "255,215,100",
    rings: [
      { label: "ø7", intervals: [0, 3, 6, 10], voicing: "close" },
      { label: "ø7", intervals: [0, 3, 6, 10], voicing: "drop2" },
      { label: "ø7", intervals: [0, 3, 6, 10], voicing: "drop3" },
    ],
  },
};
const QUALITY_ORDER = ["maj", "maj7", "dom7", "min", "m7", "m7b5"];

// 和弦库：默认 + 用户编辑（localStorage 覆盖；旧格式 drop 字段迁移为 voicing）
function migrateRing(r) {
  if (r.voicing) return r;
  return { ...r, voicing: r.drop === 2 ? "drop2" : r.drop === 3 ? "drop3" : "close" };
}
function loadChordLib() {
  try {
    const raw = localStorage.getItem("gs_chords_v1");
    if (raw) {
      const d = JSON.parse(raw);
      if (d && d.maj && d.maj.rings) {
        for (const k of QUALITY_ORDER) if (d[k]) d[k].rings = d[k].rings.map(migrateRing);
        return d;
      }
    }
  } catch {}
  return JSON.parse(JSON.stringify(DEFAULT_CHORD_LIB));
}
let CHORD_QUALITIES = loadChordLib();
function saveChordLib() { try { localStorage.setItem("gs_chords_v1", JSON.stringify(CHORD_QUALITIES)); } catch {} }

// 形态 label → 和弦名后缀（Maj/Min 特判，其余直接拼 label，支持自定义）
function ringSuffix(label) {
  if (label === "Maj") return "";
  if (label === "Min") return "m";
  return label;
}
// 圆盘统一环数 = 所有家族形态数的最大值（形态不足的家族，空环带以浅色标示）
function maxRings() {
  let m = 1;
  for (const k of QUALITY_ORDER) m = Math.max(m, CHORD_QUALITIES[k].rings.length);
  return m;
}
// 音程字符串 → 升序去重数组。
// 支持半音数字（"0 4 7"）与和弦内音记法（"R b3 5 b7 9"）。
// 和弦内音记法：R=根音, b3=小三度, 3=大三度, 4=纯四度, b5=减五度, 5=纯五度, #5=增五度,
// 6=大六度, b7=小七度, 7=大七度, b9=13, 9=14, #9=15, 11=17, #11=18, b13=20, 13=21。
const CHORD_TONE_TO_SEMI = {
  r: 0, "1": 0,
  "b3": 3, "3": 4, "4": 5, "b5": 6, "5": 7, "#5": 8, "6": 9,
  "b7": 10, "7": 11,
  "b9": 13, "9": 14, "#9": 15, "11": 17, "#11": 18, "b13": 20, "13": 21,
};
const SEMI_TO_CHORD_TONE = {
  0: "R", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "#5", 9: "6",
  10: "b7", 11: "7", 13: "b9", 14: "9", 15: "#9", 17: "11", 18: "#11", 20: "b13", 21: "13",
};
function parseIntervals(str) {
  const out = [];
  for (const raw of str.trim().split(/[\s,，]+/)) {
    const tok = raw.trim().toLowerCase();
    if (!tok) continue;
    let semi = null;
    if (CHORD_TONE_TO_SEMI[tok] !== undefined) semi = CHORD_TONE_TO_SEMI[tok];
    else {
      const n = Number(tok);
      if (Number.isFinite(n) && n >= 0 && n <= 24) semi = n;   // 兼容旧的半音数字输入
    }
    if (semi !== null && !out.includes(semi)) out.push(semi);
  }
  return out.sort((a, b) => a - b);
}
function formatIntervals(intervals) {
  return intervals.map((i) => SEMI_TO_CHORD_TONE[i] ?? String(i)).join(" ");
}
function midiNoteName(m) {
  return CHROMATIC[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
}

// ============================================================
//  左手：手势 → 级数 I~VII（原版"数手指"方案，最直观最准）
//   1=食指  2=食+中  3=食+中+拇指  4=四指(无拇指)  5=五指全伸
//   6=金属礼(食+小指)  7=我爱你(拇+食+小指)  拳头=停
// ============================================================
function getLeftHandDegree(landmarks) {
  const idx   = isFingerExtended(landmarks, "index");
  const mid   = isFingerExtended(landmarks, "middle");
  const ring  = isFingerExtended(landmarks, "ring");
  const pinky = isFingerExtended(landmarks, "pinky");
  const thumb = isThumbExtended(landmarks, "left");

  // 拳头 = 停止
  if (!idx && !mid && !ring && !pinky && !thumb) return 0;

  // 5级 = 五指全伸
  if (allFingersUp(landmarks, "left")) return 5;

  // 7级 = 拇+食+小指（我爱你），中+无名弯曲 —— 必须在 6 级前判断
  if (thumb && idx && pinky && !mid && !ring) return 7;

  // 6级 = 食+小指（金属礼），中+无名+拇弯曲
  if (idx && pinky && !mid && !ring && !thumb) return 6;

  // 4级 = 食+中+无名+小指（不含拇指）
  if (idx && mid && ring && pinky && !thumb) return 4;

  // 3级 = 食+中+拇指
  if (idx && mid && thumb && !ring && !pinky) return 3;

  // 2级 = 食+中
  if (idx && mid && !ring && !pinky && !thumb) return 2;

  // 1级 = 只伸食指
  if (idx && !mid && !ring && !pinky && !thumb) return 1;

  return -1; // 未定义手势 → 不出声
}

function getLeftHandInfo(landmarks) {
  const degree = getLeftHandDegree(landmarks);
  if (degree <= 0) return null;

  // 上下移 → 升降半音（带滞后防闪）
  // 升区 y<0.30、降区 y>0.70，还原带 0.30-0.70（缩小升降区）
  const y = landmarks[0].y;
  const accidental = getAccidentalWithHysteresis(y);

  let tonic = currentTonicFreq;
  if (tonic === 369.99 || tonic === 392.0 || tonic === 415.3) tonic /= 2;
  const semitone = DEGREE_SEMITONES[degree] + accidental;
  const rootFreq = tonic * Math.pow(2, semitone / 12);
  const rootPC = ((KEY_PC[currentKeyName] + semitone) % 12 + 12) % 12;
  const rootName = CHROMATIC[rootPC];

  return { degree, accidental, rootFreq, rootName };
}

// 升降滞后（全局锁定，防边界闪烁）
let lockedAccidental = 0;
function getAccidentalWithHysteresis(y) {
  if (lockedAccidental === 1 && y < 0.45) return 1;         // 锁升，未越过中线 → 保持升
  if (lockedAccidental === -1 && y > 0.55) return -1;       // 锁降，未越过中线 → 保持降
  if (lockedAccidental === 0 && y >= 0.28 && y <= 0.72) return 0; // 锁还原，在还原带 → 保持
  // 超出滞后带 → 切换
  if (y < 0.30) lockedAccidental = 1;
  else if (y > 0.70) lockedAccidental = -1;
  else lockedAccidental = 0;
  return lockedAccidental;
}

// ============================================================
//  右手：平滑后的食指指尖 → 圆盘（像素空间判断，画多大识别多大）
//    R = min(画布宽高) × 0.17（适中大小），触发范围延伸到 R×1.35
//    角度 → 家族（6 扇区）  半径 → 形态（3 环）
// ============================================================
const WHEEL_CX = 0.78;
const WHEEL_CY = 0.55;
const WHEEL_R_RATIO = 0.204;   // 圆盘主半径 = min(cw,ch) × 0.204（扩大 1.2 倍）
const WHEEL_TRIGGER_EXT = 1.35; // 触发范围延伸倍数

let smoothRightTip = null;
let lockedQuality = null;   // 滞后锁定的属性（家族）
let lockedRing = null;      // 滞后锁定的形态（①近圆心 ②中圈 ③外圈）

function qualityFromTip(tip) {
  if (!tip) return null;
  const cw = canvasEl.width, ch = canvasEl.height;
  const center = normToCanvas(WHEEL_CX, WHEEL_CY);
  const tipPx = normToCanvas(1 - tip.x, tip.y);
  const R = Math.min(cw, ch) * WHEEL_R_RATIO;
  const R_TRIGGER = R * WHEEL_TRIGGER_EXT;

  const dx = tipPx.x - center.x;
  const dy = tipPx.y - center.y;
  const L = Math.hypot(dx, dy);
  const inWheel = L >= R * 0.10 && L <= R_TRIGGER;

  if (inWheel) {
    // 角度 → 扇区（带滞后，像素空间=屏幕视觉角度）
    let deg = (Math.atan2(dy, dx) * 180) / Math.PI;
    if (deg < 0) deg += 360;
    const adjusted = (deg - 270 + 360) % 360; // 从正上方起算
    let sector = Math.floor(adjusted / 60);

    if (lockedQuality !== null) {
      const curSector = QUALITY_ORDER.indexOf(lockedQuality);
      const curCenter = curSector * 60 + 30;
      let diff = Math.abs(adjusted - curCenter);
      if (diff > 180) diff = 360 - diff;
      if (diff < 30) sector = curSector;
      else lockedQuality = QUALITY_ORDER[sector];
    } else {
      lockedQuality = QUALITY_ORDER[sector];
    }

    // 半径 → 形态（带滞后；环数 = 所有家族形态数的最大值，圆盘统一几何）
    const nRings = maxRings();
    let ring = Math.min(nRings - 1, Math.max(0, Math.floor((L / R) * nRings)));

    if (lockedRing !== null && lockedRing < nRings) {
      // 中心带保持；超出中心带 → 切换到新环并锁定
      const lo = R * (lockedRing / nRings + 0.08);
      const hi = R * ((lockedRing + 1) / nRings - 0.08);
      if (L < lo || L > hi) lockedRing = ring;
    } else {
      lockedRing = ring;
    }
  }

  if (lockedQuality === null) return null;
  return { quality: lockedQuality, ring: lockedRing ?? 0, active: inWheel };
}

// 和弦名：家族 + 环形态 → 如 "Cmaj9 (drop2)"、"G7sus4"、"A13"
function chordName(rootName, quality, ringIndex) {
  const q = CHORD_QUALITIES[quality];
  if (!q) return rootName;
  const r = q.rings[ringIndex] || q.rings[0];
  const base = rootName + ringSuffix(r.label);
  if (r.voicing === "drop2") return base + " (drop2)";
  else if (r.voicing === "drop3") return base + " (drop3)";
  else if (r.voicing === "drop24") return base + " (drop2&4)";
  else if (r.voicing === "drop23") return base + " (drop2&3)";
  else if (Array.isArray(r.voicing)) return base + " (自定义)";
  return base;
}

// voicing 应用：drop2=第二高音降八度；drop3=第三高音降八度；drop2&4=第二、四高音降八度；
// drop2&3=第二、三高音降八度。若 voicing 为数组，则视为自定义音程排列（相对根音的半音偏移）。
function applyVoicing(intervals, voicing) {
  if (Array.isArray(voicing)) return voicing.slice();
  const n = intervals.length;
  const targets = { drop2: [n - 2], drop3: [n - 3], drop24: [n - 2, n - 4], drop23: [n - 2, n - 3] }[voicing] || [];
  const out = intervals.slice();
  for (const i of targets) if (i >= 0) out[i] -= 12;
  return out;
}
// 按环形态构建频率（支持 voicing 排列）
function buildChordFreqs(rootFreq, quality, ringIndex) {
  const q = CHORD_QUALITIES[quality];
  if (!q) return [];
  const r = q.rings[ringIndex] || q.rings[0];
  return applyVoicing(r.intervals, r.voicing).map((i) => rootFreq * Math.pow(2, i / 12));
}

// ============================================================
//  就近衔接（voice leading）：切换和弦时各声部就近移动
//  目标音级集合 + 上一和弦声部锚点 → 暴力枚举所有分配，最小化总位移
//  防漂移：音域锁定 C3~E6（硬边界）+ 轻量回归中音区（软拉力）
// ============================================================
const VL_LOW = 48, VL_HIGH = 84;   // 声部音域 C3 ~ E6
const VL_CENTER = 60;              // 中音区回归中心（C4）
const VL_REGRESS = 0.3;            // 中心回归权重（防越跑越偏）
const midiToFreq = (m) => 440 * Math.pow(2, (m - 69) / 12);

// ---------- 理论 voicing 候选库 ----------
// 从一个和弦音程集合生成多种爵士/流行常用排列（shape = 相对根音的半音偏移数组）。
// 这些 voicing 同时用于：①和弦编辑器里的 voicing 建议；②就近衔接时从中挑“离上一和弦最近”的排列。
// 理论依据：drop 类（drop2/drop3/drop2&4/drop2&3）来自爵士 big band 编曲的开放排列；
// 转位/inversion 保证低声部旋律化；rootless A/B 来自现代爵士钢琴的左手无根音 voicing；
// spread 用于流行钢琴的开放三和弦/七和弦。
function suggestVoicings(intervals) {
  const n = intervals.length;
  const list = [];
  const add = (key, label, offsets) => {
    const arr = offsets.slice().sort((a, b) => a - b);
    if (!list.some((x) => x.key === key)) list.push({ key, label, offsets: arr });
  };
  const drop = (voicing) => applyVoicing(intervals, voicing);

  if (n === 0) return list;
  add("close", "原位 close", intervals);

  // 根据音程集合判断和弦性质，用于生成对应的 rootless voicing
  const has = (semi) => intervals.includes(semi);
  const kind = has(4) && has(7) && has(11) ? "maj7"
             : has(4) && has(7) && has(10) ? "dom7"
             : has(3) && has(7) && has(10) ? "m7"
             : has(3) && has(6) && has(10) ? "m7b5"
             : has(4) && has(7) ? "maj"
             : has(3) && has(7) ? "min"
             : has(4) && has(10) ? "dom7"
             : has(3) && has(10) ? "m7" : "other";

  if (n === 3) {
    const [a, b, c] = intervals;
    add("drop2", "drop2", drop("drop2"));
    add("inv1", "第一转位", [b, c, a + 12]);
    add("inv2", "第二转位", [c, a + 12, b + 12]);
    add("spread", "开放 spread", [a, b + 12, c]);
    add("add9", kind === "min" ? "min add9" : "add9", [a, b, c, a + 14]);
  } else if (n === 4) {
    add("drop2", "drop2", drop("drop2"));
    add("drop3", "drop3", drop("drop3"));
    add("drop24", "drop2&4", drop("drop24"));
    add("drop23", "drop2&3", drop("drop23"));
    const [a, b, c, d] = intervals;
    add("inv1", "第一转位", [b, c, d, a + 12]);
    add("inv2", "第二转位", [c, d, a + 12, b + 12]);
    add("inv3", "第三转位", [d, a + 12, b + 12, c + 12]);
    add("spread", "开放 spread", [a, c, b + 12, d + 12]);

    // rootless A/B：现代爵士最常用的无根音 voicing（左手用，右手可叠旋律）
    if (kind === "maj7") {
      add("rootlessA", "rootless A (3-5-7-9)", [b, c, d, 14]);
      add("rootlessB", "rootless B (7-9-3-5)", [d, 14, b + 12, c + 12]);
    } else if (kind === "dom7") {
      add("rootlessA", "rootless A (3-5-b7-9)", [b, c, d, 14]);
      add("rootlessB", "rootless B (b7-9-3-5)", [d, 14, b + 12, c + 12]);
      add("rootless13", "rootless 13 (3-6-b7-9)", [b, 9, d, 14]);
    } else if (kind === "m7") {
      add("rootlessA", "rootless A (b3-5-b7-9)", [b, c, d, 14]);
      add("rootlessB", "rootless B (b7-9-b3-5)", [d, 14, b + 12, c + 12]);
      add("rootless11", "rootless 11 (b3-5-b7-11)", [b, c, d, 17]);
    } else if (kind === "m7b5") {
      add("rootlessA", "rootless A (b3-b5-b7-b9)", [b, c, d, 13]);
      add("rootlessB", "rootless B (b7-b9-b3-b5)", [d, 13, b + 12, c + 12]);
      add("rootless11", "rootless 11 (b3-b5-b7-11)", [b, c, d, 17]);
    } else {
      const ext = [14, 9, 21].find((x) => !intervals.includes(x) && x > d) ?? 14;
      add("rootlessA", "rootless A", [b, c, d, ext]);
    }
  } else {
    add("drop2", "drop2", drop("drop2"));
    add("drop3", "drop3", drop("drop3"));
    add("drop24", "drop2&4", drop("drop24"));
    add("drop23", "drop2&3", drop("drop23"));
    add("spread", "开放 spread", intervals.map((v, i) => (i % 2 === 1 ? v + 12 : v)));
    add("inv1", "第一转位", intervals.slice(1).concat([intervals[0] + 12]));
  }
  return list;
}

// 全排列（n ≤ 5，暴力枚举足够快）
function* vlPermutations(arr) {
  if (arr.length <= 1) { yield arr.slice(); return; }
  for (let i = 0; i < arr.length; i++) {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const p of vlPermutations(rest)) yield [arr[i], ...p];
  }
}
// 从 src 中选 k 个的组合（声部数变化时使用）
function* vlCombinations(arr, k, start = 0, cur = []) {
  if (cur.length === k) { yield cur.slice(); return; }
  for (let i = start; i < arr.length; i++) {
    cur.push(arr[i]);
    yield* vlCombinations(arr, k, i + 1, cur);
    cur.pop();
  }
}
// 计算把一个 MIDI 集合安排到另一个 MIDI 集合的最小总位移（每个声部一一对应）。
function vlMinMatchingCost(from, to) {
  const n = to.length;
  let best = Infinity;
  // 上一和弦声部多于目标时，枚举选择哪几个声部保留
  for (const combo of from.length > n ? vlCombinations(from, n) : [from]) {
    const src = combo.length === n ? combo : combo.slice();
    // 上一和弦声部不足时，用目标音补齐（由调用方保证 n 相同）
    if (src.length !== n) continue;
    for (const perm of vlPermutations(src)) {
      let cost = 0;
      for (let i = 0; i < n; i++) cost += Math.abs(perm[i] - to[i]);
      if (cost < best - 1e-9) best = cost;
    }
  }
  return best === Infinity ? 0 : best;
}
// 就近分配：从理论 voicing 库中，选一个与上一和弦总位移最小、且靠近中音区的排列。
// 返回新的频率数组（升序）。若 preferredVoicing 是自定义数组，则优先使用该排列（仍会做八度选择）。
function voiceLeadFreqs(rootFreq, intervals, srcFreqs, preferredVoicing) {
  const rootMidi = freqToMidi(rootFreq);
  const srcAll = srcFreqs.map(freqToMidi).sort((a, b) => a - b);
  const baseN = intervals.length;
  if (baseN === 0) return [];

  const shapes = [];
  if (Array.isArray(preferredVoicing)) shapes.push({ key: "custom", label: "自定义", offsets: preferredVoicing.slice().sort((a, b) => a - b) });
  const lib = suggestVoicings(intervals);
  for (const sh of lib) {
    if (!shapes.some((x) => x.key === sh.key)) shapes.push(sh);
  }

  let bestCost = Infinity, best = null;

  for (const sh of shapes) {
    const offs = sh.offsets;
    const m = offs.length;
    if (m === 0) continue;

    // 目标声部数变化：上一和弦声部不足时，用中音区默认位补齐到当前 voicing 的声部数
    const src = srcAll.slice();
    while (src.length < m) {
      const pc = ((rootMidi + intervals[src.length % baseN]) % 12 + 12) % 12;
      src.push(pc + 12 * Math.round((VL_CENTER - pc) / 12));
    }
    const srcAvg = src.reduce((a, b) => a + b, 0) / src.length;
    const shapeAvg = offs.reduce((a, b) => a + b, 0) / m;
    // 以“形状中心靠近上一和弦中心”为基准，在其上下两个八度内寻找最佳摆放
    const baseShift = 12 * Math.round((srcAvg - rootMidi - shapeAvg) / 12);
    for (let shift = baseShift - 24; shift <= baseShift + 24; shift += 12) {
      const cand = offs.map((o) => rootMidi + o + shift).sort((a, b) => a - b);
      if (cand[0] < VL_LOW || cand[m - 1] > VL_HIGH) continue;
      // 形状匹配 preferredVoicing 时给一点偏好，避免频繁在 voicing 间跳动
      const prefer = Array.isArray(preferredVoicing)
        ? (sh.key === "custom" ? 0.85 : 1)
        : (sh.key === preferredVoicing ? 0.85 : 1);
      const moveCost = vlMinMatchingCost(src, cand);
      const centerCost = cand.reduce((a, c) => a + Math.abs(c - VL_CENTER), 0) * VL_REGRESS;
      const cost = prefer * moveCost + centerCost;
      if (cost < bestCost - 1e-9) {
        bestCost = cost;
        best = cand;
      }
    }
  }
  if (!best) return srcAll.slice(0, baseN).map(midiToFreq);   // 理论兜底
  return best.map(midiToFreq);
}

// ============================================================
//  合成引擎
// ============================================================
class SynthEngine {
  constructor() {
    this.ctx = null; this.filter = null; this.masterGain = null;
    this.voices = [];          // [{oscs, gain}] 当前发声的振荡器组（oscs 可为 1~2 个）
    this.fadingOut = [];       // 正在淡出的振荡器组
    this.currentKey = null; this.volume = 0.35;
    this.waves = {};           // 自定义谐波音色表（PeriodicWave）
    this.attack = 0.09; this.release = 0.06;  // 包络（来自音色 FX）
    this.splitter = null; this.merger = null;  // 立体声 Tremolo 链路
    this.tremL = null; this.tremR = null;
    this.tremLfo = null; this.tremDepthL = null; this.tremDepthR = null;
  }
  ensureContext() {
    if (this.ctx) {
        // 如果已存在但挂起，尝试恢复
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        return;
    }
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    
    // iOS 关键：创建后立即检查并恢复
    if (this.ctx.state === 'suspended') {
        this.ctx.resume();
    }
    // —— 效果器链：voice → toneFilter(低通) → 3段EQ(低/中/高) → [干: masterGain] + [湿: delay / reverb] → masterGain ——
    this.filter = this.ctx.createBiquadFilter();
    this.filter.type = "lowpass";
    this.filter.frequency.value = 2400;
    this.filter.Q.value = 0.7;

    // 三段 EQ（Roland 风格 low/mid/high，默认 0 dB 不染色）
    this.eqLow = this.ctx.createBiquadFilter();
    this.eqLow.type = "lowshelf";
    this.eqLow.frequency.value = 220;
    this.eqLow.gain.value = 0;
    this.eqMid = this.ctx.createBiquadFilter();
    this.eqMid.type = "peaking";
    this.eqMid.frequency.value = 900;
    this.eqMid.Q.value = 0.9;
    this.eqMid.gain.value = 0;
    this.eqHigh = this.ctx.createBiquadFilter();
    this.eqHigh.type = "highshelf";
    this.eqHigh.frequency.value = 2800;
    this.eqHigh.gain.value = 0;

    this.filter.connect(this.eqLow);
    this.eqLow.connect(this.eqMid);
    this.eqMid.connect(this.eqHigh);

    // 主控三段 EQ（设置面板直接调，Roland 风格 low/mid/high，独立于音色编辑里的 tone EQ）
    this.masterLow = this.ctx.createBiquadFilter();
    this.masterLow.type = "lowshelf";
    this.masterLow.frequency.value = 110;
    this.masterLow.gain.value = masterEQ.lowGain;
    this.masterMid = this.ctx.createBiquadFilter();
    this.masterMid.type = "peaking";
    this.masterMid.frequency.value = 800;
    this.masterMid.Q.value = 0.8;
    this.masterMid.gain.value = masterEQ.midGain;
    this.masterHigh = this.ctx.createBiquadFilter();
    this.masterHigh.type = "highshelf";
    this.masterHigh.frequency.value = 3200;
    this.masterHigh.gain.value = masterEQ.highGain;

    this.eqHigh.connect(this.masterLow);
    this.masterLow.connect(this.masterMid);
    this.masterMid.connect(this.masterHigh);

    // 波形可视化探针（2048 点：时间窗足够估算基频，又不会把周期挤太密）
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.masterHigh.connect(this.analyser);

    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = this.volume;
    this.masterHigh.connect(this.masterGain);           // 干路

    // 延迟（带反馈回路）
    this.delay = this.ctx.createDelay(1.2);
    this.delay.delayTime.value = 0;
    this.delayFb = this.ctx.createGain();
    this.delayFb.gain.value = 0;
    this.delayWet = this.ctx.createGain();
    this.delayWet.gain.value = 0;
    this.masterHigh.connect(this.delay);
    this.delay.connect(this.delayFb);
    this.delayFb.connect(this.delay);               // 反馈回路
    this.delay.connect(this.delayWet);
    this.delayWet.connect(this.masterGain);         // 延迟湿路

    // 混响（卷积 + 生成的指数衰减噪声脉冲）
    this.reverb = this.ctx.createConvolver();
    this.reverb.buffer = makeImpulse(this.ctx, 1.8, 4);
    this.reverbWet = this.ctx.createGain();
    this.reverbWet.gain.value = 0;
    this.masterHigh.connect(this.reverb);
    this.reverb.connect(this.reverbWet);
    this.reverbWet.connect(this.masterGain);        // 混响湿路

    // —— 立体声 Tremolo：master → splitter → [L/R 各一 Gain，LFO 反相调制] → merger → destination ——
    this.splitter = this.ctx.createChannelSplitter(2);
    this.merger = this.ctx.createChannelMerger(2);
    this.tremL = this.ctx.createGain(); this.tremL.gain.value = 1;
    this.tremR = this.ctx.createGain(); this.tremR.gain.value = 1;
    this.tremLfo = this.ctx.createOscillator();
    this.tremLfo.frequency.value = tremoloRate;
    this.tremDepthL = this.ctx.createGain(); this.tremDepthL.gain.value = tremoloOn ? tremoloDepth : 0;
    this.tremDepthR = this.ctx.createGain(); this.tremDepthR.gain.value = tremoloOn ? -tremoloDepth : 0;  // 反相
    this.tremLfo.connect(this.tremDepthL); this.tremDepthL.connect(this.tremL.gain);
    this.tremLfo.connect(this.tremDepthR); this.tremDepthR.connect(this.tremR.gain);
    this.tremLfo.start();

    this.masterGain.connect(this.splitter);
    this.splitter.connect(this.tremL, 0, 0); this.tremL.connect(this.merger, 0, 0);
    this.splitter.connect(this.tremR, 1, 0); this.tremR.connect(this.merger, 0, 1);
    this.merger.connect(this.ctx.destination);

    // 自定义谐波音色 → PeriodicWave 表（含纯正弦兜底，防 fallback 时 undefined）
    for (const key of Object.keys(BASE_TONES)) {
      const t = BASE_TONES[key];
      if (t.harmonics) {
        const real = new Float32Array(t.harmonics.length + 1);
        real[0] = 0;
        t.harmonics.forEach((h, i) => { real[i + 1] = h; });
        this.waves[key] = this.ctx.createPeriodicWave(real, new Float32Array(real.length));
      }
    }
    if (!this.waves.triangle) {
      this.waves.triangle = this.ctx.createPeriodicWave(new Float32Array([0, 1]), new Float32Array([0, 0]));
    }

    this.applyToneParams(currentToneDef.fx);   // 应用当前音色的效果器参数
  }
  // 应用音色效果器参数（截止/共振/延迟/混响/包络），可随时调用（编辑时实时试听）
  applyToneParams(fx) {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.filter.frequency.setTargetAtTime(fx.cutoff, now, 0.03);
    this.filter.Q.setTargetAtTime(fx.q, now, 0.03);
    if (this.eqLow) this.eqLow.gain.setTargetAtTime(fx.lowGain ?? 0, now, 0.03);
    if (this.eqMid) this.eqMid.gain.setTargetAtTime(fx.midGain ?? 0, now, 0.03);
    if (this.eqHigh) this.eqHigh.gain.setTargetAtTime(fx.highGain ?? 0, now, 0.03);
    this.delay.delayTime.setTargetAtTime(fx.delayTime, now, 0.03);
    this.delayFb.gain.setTargetAtTime(fx.delayFb, now, 0.03);
    this.delayWet.gain.setTargetAtTime(fx.delayTime > 0.005 ? 0.35 : 0, now, 0.03);
    this.reverbWet.gain.setTargetAtTime(fx.reverb, now, 0.03);
    this.attack = fx.attack; this.release = fx.release;
  }
  // 主控三段 EQ（设置面板滑杆实时调用）
  applyMasterEQ(eq) {
    if (!this.ctx || !this.masterLow) return;
    const now = this.ctx.currentTime;
    this.masterLow.gain.setTargetAtTime(eq.lowGain ?? 0, now, 0.02);
    this.masterMid.gain.setTargetAtTime(eq.midGain ?? 0, now, 0.02);
    this.masterHigh.gain.setTargetAtTime(eq.highGain ?? 0, now, 0.02);
  }
  // 应用 Tremolo 参数（开关/深度/速率变化时调用）
  applyTremolo() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.tremLfo.frequency.setTargetAtTime(tremoloRate, now, 0.03);
    this.tremDepthL.gain.setTargetAtTime(tremoloOn ? tremoloDepth : 0, now, 0.03);
    this.tremDepthR.gain.setTargetAtTime(tremoloOn ? -tremoloDepth : 0, now, 0.03);
  }
  setVolume(v) {
    this.volume = v;
    if (this.ctx) {
      this.masterGain.gain.cancelScheduledValues(this.ctx.currentTime);
      this.masterGain.gain.linearRampToValueAtTime(v, this.ctx.currentTime + 0.04);
    }
  }
  // 交叉淡化：旧音按 release 淡出，新音按 attack 淡入，重叠消除咔哒声
  playNotes(freqs) {
    if (!this.ctx || freqs.length === 0) return;
    const key = freqs.map((f) => f.toFixed(1)).join(",");
    if (key === this.currentKey) return;
    const now = this.ctx.currentTime;
    const fadeOut = Math.min(0.5, this.release), fadeIn = Math.min(0.5, this.attack);

    // 旧音淡出
    this.voices.forEach(({ oscs, gain }) => {
      try {
        gain.gain.cancelScheduledValues(now);
        gain.gain.setValueAtTime(gain.gain.value, now);
        gain.gain.linearRampToValueAtTime(0, now + fadeOut);
        oscs.forEach((o) => o.stop(now + fadeOut + 0.02));
      } catch {}
    });
    this.fadingOut.push(...this.voices);
    this.fadingOut = this.fadingOut.filter((v) => {
      try { return v.oscs[0].context.currentTime <= now + fadeOut; } catch { return false; }
    });

    // 新音淡入（支持自定义谐波 / 双振荡器 detune）
    const tone = currentToneDef;
    this.voices = freqs.map((freq) => {
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(1, now + fadeIn);
      gain.connect(this.filter);
      const oscs = [];
      const mkOsc = (detune) => {
        const osc = this.ctx.createOscillator();
        if (tone.type === "custom") {
          // custom 波形只能通过 setPeriodicWave() 设置（会自动把 type 置为 custom），直接赋 "custom" 会抛 InvalidStateError
          osc.setPeriodicWave(this.waves[tone.base] || this.waves.triangle);
        } else {
          osc.type = tone.type;
        }
        osc.frequency.value = freq;
        if (detune) osc.detune.value = detune;
        osc.connect(gain);
        osc.start(now);
        oscs.push(osc);
      };
      if (tone.dual) { mkOsc(-tone.detune); mkOsc(tone.detune); }
      else mkOsc(0);
      return { oscs, gain };
    });
    this.currentKey = key;
  }
  stop() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const rel = Math.min(0.5, this.release);
    this.voices.forEach(({ oscs, gain }) => {
      try {
        gain.gain.cancelScheduledValues(now);
        gain.gain.setValueAtTime(gain.gain.value, now);
        gain.gain.linearRampToValueAtTime(0, now + rel);
        oscs.forEach((o) => o.stop(now + rel + 0.02));
      } catch {}
    });
    this.voices = []; this.currentKey = null;
  }
}
const synth = new SynthEngine();

// 生成混响脉冲（指数衰减随机噪声）
function makeImpulse(ctx, seconds, decay) {
  const rate = ctx.sampleRate, len = Math.floor(rate * seconds);
  const buf = ctx.createBuffer(2, len, rate);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
  }
  return buf;
}

// ============================================================
//  防抖：160ms 内稳定才认
// ============================================================
const HOLD_MS = 220;
let stableState = null, candidateState = null, candidateSince = 0, lastValid = 0;
function sameState(a, b) {
  if (!a || !b) return a === b;
  return a.rootName === b.rootName && a.quality === b.quality && a.ring === b.ring;
}
function stabilize(raw, now) {
  if (raw !== null) lastValid = now;
  let eff = raw;
  if (raw === null && now - lastValid < 100) eff = candidateState;
  if (!sameState(eff, candidateState)) { candidateState = eff; candidateSince = now; }
  if (now - candidateSince >= HOLD_MS) stableState = candidateState;
  return stableState;
}

// ============================================================
//  钢琴键盘可视化
// ============================================================
const PIANO_LOW = 48, PIANO_HIGH = 83;
const WHITE_SET = new Set([0, 2, 4, 5, 7, 9, 11]);
const keyEls = new Map();
function buildPiano() {
  pianoEl.innerHTML = ""; keyEls.clear();
  const whiteMidis = [];
  for (let m = PIANO_LOW; m <= PIANO_HIGH; m++) if (WHITE_SET.has(m % 12)) whiteMidis.push(m);
  const n = whiteMidis.length, w = 100 / n;
  let wc = 0;
  for (let m = PIANO_LOW; m <= PIANO_HIGH; m++) {
    const pc = m % 12;
    const el = document.createElement("div");
    if (WHITE_SET.has(pc)) { el.className = "key-white"; pianoEl.appendChild(el); keyEls.set(m, el); wc++; }
    else {
      el.className = "key-black";
      el.style.left = wc * w - w * 0.3 + "%";
      el.style.width = w * 0.6 + "%";
      pianoEl.appendChild(el); keyEls.set(m, el);
    }
    el.addEventListener("click", () => playPianoKey(m));   // 钢琴可点击试音
  }
}
// 点击钢琴键：播放该键单音并高亮（音色编辑时试单音最方便）
function playPianoKey(midi) {
    if (auditionOn) stopAudition();
    if (!synth.ctx) {
        synth.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (synth.ctx.state === "suspended") {
        synth.ctx.resume();
    }
    synth.currentKey = null;
    const f = midiToFreq(midi);
    synth.playNotes([f]);
    lightPiano([f]);
}
const freqToMidi = (f) => Math.round(69 + 12 * Math.log2(f / 440));
function lightPiano(freqs) {
  keyEls.forEach((el) => el.classList.remove("active"));
  for (const f of freqs) { const el = keyEls.get(freqToMidi(f)); if (el) el.classList.add("active"); }
}

// ============================================================
//  高亮过渡动画（参考级数按钮 CSS transition，150ms 淡入）
//  记录显示值变化时间，绘制时用进度插值 → 切换平滑不闪
// ============================================================
const displayAnim = {
  accidental: { value: null, changedAt: 0 },
  quality: { value: null, changedAt: 0 },
  ring: { value: null, changedAt: 0 },
};
const ANIM_MS = 150;
function animProgress(key, target, now) {
  const a = displayAnim[key];
  if (target !== a.value) {
    a.value = target;
    a.changedAt = now;
  }
  return Math.min(1, (now - a.changedAt) / ANIM_MS);
}
// ease-out 曲线
const easeOut = (t) => 1 - Math.pow(1 - t, 3);

// ============================================================
//  圆角矩形辅助
// ============================================================
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// ============================================================
//  画左侧升降区域（三个竖排符号块：# / ♮ / b，带淡入动画）
//  纵向铺开对应手势检测区：上移=升(#)、中=还原(♮)、下移=降(b)
//  （左上控制面板默认折叠成齿轮，不再遮挡）
// ============================================================
function drawAccidentalZones(accidental, now) {
  const cw = canvasEl.width, ch = canvasEl.height;
  const zoneW = Math.min(64, cw * 0.09);
  const zoneH = Math.min(54, ch * 0.085);
  const x0 = 14;
  const zones = [
    { label: "#", acc: 1,  color: "252,211,77",  yCenter: ch * 0.16 },
    { label: "♮", acc: 0,  color: "94,234,212",  yCenter: ch * 0.50 },
    { label: "b", acc: -1, color: "125,211,252", yCenter: ch * 0.78 },
  ];
  const t = easeOut(animProgress("accidental", accidental, now));
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  zones.forEach((z) => {
    const y = z.yCenter - zoneH / 2;
    const on = accidental === z.acc && accidental !== null;
    // 深色底块（提升符号对比）
    ctx.fillStyle = "rgba(20,28,42,0.60)";
    roundRect(ctx, x0, y, zoneW, zoneH, 16);
    ctx.fill();
    // 彩色高亮填充：on 从 0.18 渐到 0.85
    const fillA = on ? 0.18 + (0.85 - 0.18) * t : 0.18;
    ctx.fillStyle = `rgba(${z.color},${fillA})`;
    roundRect(ctx, x0, y, zoneW, zoneH, 16);
    ctx.fill();
    ctx.strokeStyle = `rgba(${z.color},${on ? 0.55 + 0.45 * t : 0.45})`;
    ctx.lineWidth = on ? 1.5 + 2.5 * t : 1.5;
    ctx.stroke();
    // 符号：黑描边 + 亮填充
    ctx.font = "bold 28px sans-serif";
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.lineWidth = 4;
    ctx.strokeText(z.label, x0 + zoneW / 2, z.yCenter);
    ctx.fillStyle = on ? "#ffffff" : `rgba(${z.color},0.95)`;
    ctx.fillText(z.label, x0 + zoneW / 2, z.yCenter);
  });
  // 标题（黑描边）
  ctx.font = "bold 11px 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif";
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.lineWidth = 3;
  ctx.strokeText("左手 · 上下移", x0 + zoneW / 2, ch * 0.16 - zoneH / 2 - 12);
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillText("左手 · 上下移", x0 + zoneW / 2, ch * 0.16 - zoneH / 2 - 12);
  ctx.restore();
}

// ============================================================
//  画右侧属性圆盘（像素空间、直径×1.2、淡入动画、深色清晰风格）
//  角度=家族（6 扇区） 半径=形态（3 环：①近圆心 ②中圈 ③外圈）
// ============================================================
function drawQualityWheel(selected, fingerTip, now) {
  const cw = canvasEl.width, ch = canvasEl.height;
  const center = normToCanvas(WHEEL_CX, WHEEL_CY);
  const cx = center.x, cy = center.y;
  const R = Math.min(cw, ch) * WHEEL_R_RATIO;
  const RT = R * WHEEL_TRIGGER_EXT;

  const selQuality = selected ? selected.quality : null;
  const selRing = selected ? selected.ring : null;

  const tq = easeOut(animProgress("quality", selQuality, now));
  const tr = easeOut(animProgress("ring", selRing, now));

  const CN_FONT = "'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif";
  // 文字描边辅助：黑描边 + 亮填充，任何背景都可读
  function label(text, x, y, font, fill, strokeW) {
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.lineWidth = strokeW;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = fill;
    ctx.fillText(text, x, y);
  }

  ctx.save();

  // —— 苹果毛玻璃底盘：多层径向渐变 + 外发光 + 玻璃高光 ——
  ctx.shadowColor = "rgba(94,234,212,0.30)";
  ctx.shadowBlur = 36;
  const baseGrad = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R);
  baseGrad.addColorStop(0, "rgba(96,112,138,0.90)");
  baseGrad.addColorStop(0.55, "rgba(50,62,82,0.84)");
  baseGrad.addColorStop(1, "rgba(30,40,58,0.88)");
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.fillStyle = baseGrad;
  ctx.fill();
  ctx.shadowBlur = 0;

  // 玻璃边缘：外圈亮边 + 内圈细高光
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, R - 2.5, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // 外延触发范围提示（虚线圈）
  ctx.beginPath(); ctx.arc(cx, cy, RT, 0, Math.PI * 2);
  ctx.setLineDash([5, 8]);
  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);

  // —— 6 扇区（径向渐变上色，选中高亮 + 外发光）——
  for (let i = 0; i < 6; i++) {
    const startA = (i * 60 - 90) * Math.PI / 180;
    const endA   = ((i + 1) * 60 - 90) * Math.PI / 180;
    const key = QUALITY_ORDER[i];
    const q = CHORD_QUALITIES[key];
    const selectedQ = selQuality === key;
    const alpha = selectedQ ? 0.55 + (1.0 - 0.55) * tq : 0.46;

    const grad = ctx.createRadialGradient(cx, cy, R * 0.12, cx, cy, R);
    grad.addColorStop(0, `rgba(${q.color},${Math.min(1, alpha + 0.18)})`);
    grad.addColorStop(1, `rgba(${q.color},${alpha * 0.55})`);

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, startA, endA);
    ctx.closePath();
    if (selectedQ) {
      ctx.shadowColor = `rgba(${q.color},0.8)`;
      ctx.shadowBlur = 28;
    }
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.shadowBlur = 0;

    // 扇区分割线（细白线，越靠近中心越淡）
    ctx.strokeStyle = selectedQ ? "rgba(255,255,255,0.42)" : "rgba(255,255,255,0.20)";
    ctx.lineWidth = selectedQ ? 2 : 1.2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(startA) * R, cy + Math.sin(startA) * R);
    ctx.stroke();

    // 属性标签（外缘，黑描边+白字）
    const midA = ((i + 0.5) * 60 - 90) * Math.PI / 180;
    const lx = cx + Math.cos(midA) * R * 0.82;
    const ly = cy + Math.sin(midA) * R * 0.82;
    if (selectedQ) {
      label(q.label, lx, ly, `bold ${Math.round(14 + 2 * tq)}px ${CN_FONT}`, "#ffffff", 3.5);
    } else {
      label(q.label, lx, ly, `bold 13px ${CN_FONT}`, "rgba(255,255,255,0.88)", 3);
    }
  }

  // —— 形态环边界线（环数 = 所有家族形态数最大值，几何统一）——
  const ringOwner = CHORD_QUALITIES[selQuality ?? "maj"];
  const nRings = maxRings();
  for (let i = 1; i < nRings; i++) {
    ctx.beginPath(); ctx.arc(cx, cy, R * i / nRings, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.30)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // —— 空形态环带（当前家族没有的环 → 浅色"未使用"区域）——
  const emptyFrom = ringOwner.rings.length;
  for (let i = emptyFrom; i < nRings; i++) {
    ctx.beginPath();
    ctx.arc(cx, cy, R * (i + 1) / nRings, 0, Math.PI * 2);
    ctx.arc(cx, cy, R * i / nRings, 0, Math.PI * 2, true);
    ctx.fillStyle = "rgba(255,255,255,0.055)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.lineWidth = 1;
    ctx.stroke();
    label("—", cx, cy - R * (0.18 + 0.14 * i), `600 11px ${CN_FONT}`, "rgba(255,255,255,0.40)", 2.5);
  }

  // —— 选中形态环高亮（淡入；空环不带高亮）——
  if (selRing !== null && selRing < nRings && selRing < ringOwner.rings.length) {
    const rr = R * (selRing + 1) / nRings;
    ctx.save();
    ctx.shadowColor = "rgba(94,234,212,0.8)";
    ctx.shadowBlur = 18;
    ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(94,234,212,${0.65 + 0.35 * tr})`;
    ctx.lineWidth = 2 + 2.5 * tr;
    ctx.stroke();
    ctx.restore();
  }

  // —— 玻璃高光：上半部柔白反光，下半部暗部反光 ——
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, R - 1, 0, Math.PI * 2); ctx.clip();
  const hiGrad = ctx.createLinearGradient(cx, cy - R, cx, cy + R * 0.35);
  hiGrad.addColorStop(0, "rgba(255,255,255,0.30)");
  hiGrad.addColorStop(0.45, "rgba(255,255,255,0.04)");
  hiGrad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = hiGrad;
  ctx.fillRect(cx - R, cy - R, R * 2, R * 1.35);
  ctx.restore();

  // —— 形态标签（沿圆心向上方排列，跟随当前家族形态；未选中默认 Maj 家族）——
  ringOwner.rings.forEach((r, i) => {
    const on = selRing === i && selQuality !== null;
    let voicingTxt = "";
    if (r.voicing && r.voicing !== "close") {
      if (Array.isArray(r.voicing)) voicingTxt = "自定义";
      else voicingTxt = r.voicing === "drop24" ? "d2&4" : r.voicing.replace("drop", "d");
    }
    const text = voicingTxt ? `${r.label}·${voicingTxt}` : r.label;
    label(text, cx, cy - R * (0.18 + 0.14 * i),
          on ? `bold 12px ${CN_FONT}` : `600 11px ${CN_FONT}`,
          on ? "#ffffff" : "rgba(255,255,255,0.70)", 3);
  });

  // 中心点（玻璃高光小圆）
  ctx.beginPath();
  ctx.arc(cx, cy, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.35)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // 标题
  label("右手 · 食指指圆盘", cx, cy - R - 18, `bold 12px ${CN_FONT}`, "rgba(255,255,255,0.9)", 3);
  label("角度=属性 · 半径=形态", cx, cy - R - 4, `600 10px ${CN_FONT}`, "rgba(255,255,255,0.6)", 2.5);

  // 食指光标（白点 + 十字）
  if (fingerTip) {
    const fp = normToCanvas(1 - fingerTip.x, fingerTip.y);
    const px = fp.x, py = fp.y;
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.8)"; ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px - 15, py); ctx.lineTo(px + 15, py);
    ctx.moveTo(px, py - 15); ctx.lineTo(px, py + 15);
    ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = 2; ctx.stroke();
  }
  ctx.restore();
}

// ============================================================
//  顶部级数按钮高亮 + 调性变更刷新音名
// ============================================================
function updateDegreeBar(activeDegree, accidental) {
  degreeBtns.forEach((btn) => {
    const d = Number(btn.dataset.degree);
    const on = d === activeDegree && activeDegree > 0;
    btn.classList.toggle("active", on);
    btn.classList.remove("sharp", "flat");
    if (on && accidental > 0) btn.classList.add("sharp");
    else if (on && accidental < 0) btn.classList.add("flat");
  });
}
function updateDegreeNotes() {
  degreeBtns.forEach((btn) => {
    const d = Number(btn.dataset.degree);
    const semitone = DEGREE_SEMITONES[d];
    const pc = ((KEY_PC[currentKeyName] + semitone) % 12 + 12) % 12;
    const noteEl = btn.querySelector(".note");
    if (noteEl) noteEl.textContent = CHROMATIC[pc];
  });
}
function updateAccidentalHint(accidental) {
  if (!accidentalHintEl) return;
  accidentalHintEl.classList.remove("sharp", "flat", "natural");
  if (accidental === null || accidental === undefined) return;
  if (accidental > 0) accidentalHintEl.classList.add("sharp");
  else if (accidental < 0) accidentalHintEl.classList.add("flat");
  else accidentalHintEl.classList.add("natural");
}

// ============================================================
//  摄像头 / MediaPipe
// ============================================================
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  videoEl.srcObject = stream;
  return new Promise((res) => { videoEl.onloadedmetadata = () => { videoEl.play(); res(); }; });
}
async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO", numHands: 2,
  });
}

// 头部检测（用于"头部距离控制音量"），失败不阻塞主流程
let faceDetector = null;
async function setupFaceDetector() {
  try {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
    });
    return true;
  } catch (e) {
    console.warn("FaceDetector 初始化失败，头部音量不可用：", e);
    return false;
  }
}

// 脸宽 → 音量（脸越小越远 → 音量越小，低于阈值静音）
function headVolumeFromFaceW(fw) {
  if (fw === null || fw < 0.12) return 0;      // 太远/没检测到 → 静音
  if (fw >= 0.30) return 1;                    // 很近 → 满
  return (fw - 0.12) / (0.30 - 0.12);          // 中间线性
}

function computeCoverRect(sw, sh, dw, dh) {
  const sr = sw / sh, dr = dw / dh;
  if (sr > dr) { const h = sh, w = sh * dr; return { sx: (sw - w) / 2, sy: 0, sWidth: w, sHeight: h }; }
  const w = sw, h = sw / dr; return { sx: 0, sy: (sh - h) / 2, sWidth: w, sHeight: h };
}

// 右上角波形可视化（振荡器探针 → 时域波形）
// 修复：自动电平归一化（RMS/峰值），并按估算基频截取约 2 个周期，从上升过零点开始画，
// 这样不会再“振幅顶满/周期挤成一截”，能看清每个周期的形状变化。
const scopeEl = document.getElementById("scope");
const scopeCtx = scopeEl ? scopeEl.getContext("2d") : null;
const scopeData = new Float32Array(2048);
let scopeF0 = 0;   // 平滑后的估算基频
function estimateScopeF0(data, sampleRate) {
  let crossings = 0;
  for (let i = 1; i < data.length; i++) if (data[i - 1] < 0 && data[i] >= 0) crossings++;
  return crossings * sampleRate / Math.max(1, data.length - 1);
}
function drawScope() {
  if (!scopeEl || !scopeCtx) return;
  if (scopeEl.width !== 380) { scopeEl.width = 380; scopeEl.height = 128; }   // 2x 分辨率防模糊
  const w = scopeEl.width, h = scopeEl.height;
  const c = scopeCtx;
  c.clearRect(0, 0, w, h);
  c.fillStyle = "rgba(12,17,26,0.72)";
  c.fillRect(0, 0, w, h);

  // 网格 + 中线
  c.strokeStyle = "rgba(255,255,255,0.07)";
  c.lineWidth = 1;
  for (let gx = 1; gx < 4; gx++) {
    c.beginPath(); c.moveTo((w * gx) / 4, 0); c.lineTo((w * gx) / 4, h); c.stroke();
  }
  c.strokeStyle = "rgba(255,255,255,0.12)";
  c.beginPath(); c.moveTo(0, h / 2); c.lineTo(w, h / 2); c.stroke();

  if (synth.ctx && synth.analyser) {
    synth.analyser.getFloatTimeDomainData(scopeData);
    const sr = synth.ctx.sampleRate || 48000;
    const instF0 = estimateScopeF0(scopeData, sr);
    scopeF0 = scopeF0 > 0 ? scopeF0 * 0.85 + instF0 * 0.15 : instF0;

    // 选择窗口：约 2 个基频周期；基频不可靠时退回半窗，避免太密
    let winLen = Math.floor(scopeData.length / 2);
    if (scopeF0 > 55 && scopeF0 < 1800) {
      winLen = Math.max(96, Math.min(scopeData.length, Math.round((sr * 2.0) / scopeF0)));
    }
    // 从第一个上升过零点开始截取，波形稳定不左右跳
    let start = -1;
    for (let i = 1; i <= scopeData.length - winLen; i++) {
      if (scopeData[i - 1] < 0 && scopeData[i] >= 0) { start = i; break; }
    }
    if (start < 0) start = 0;
    if (start + winLen > scopeData.length) start = scopeData.length - winLen;

    // 窗口内电平归一化：以峰值与 RMS 共同决定缩放，避免大振幅顶满
    let peak = 0, sumSq = 0;
    for (let i = start; i < start + winLen; i++) {
      const v = scopeData[i];
      const av = Math.abs(v);
      if (av > peak) peak = av;
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / Math.max(1, winLen));
    const ref = Math.max(peak, rms * 2.2, 0.0005);
    const scale = (h * 0.34) / ref;

    c.beginPath();
    for (let i = 0; i < winLen; i++) {
      const idx = start + i;
      const x = (i / Math.max(1, winLen - 1)) * w;
      const y = Math.max(2, Math.min(h - 2, h / 2 - scopeData[idx] * scale));
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.strokeStyle = "#5eead4";
    c.lineWidth = 1.8;
    c.shadowColor = "rgba(94,234,212,0.65)";
    c.shadowBlur = 6;
    c.stroke();
    c.shadowBlur = 0;

    // 左上角显示估算基频（静音显示 --）
    c.fillStyle = "rgba(94,234,212,0.75)";
    c.font = "10px ui-monospace, Menlo, monospace";
    c.textAlign = "left";
    c.textBaseline = "top";
    c.fillText(scopeF0 > 40 && scopeF0 < 1800 ? `${Math.round(scopeF0)} Hz · 2 cycles` : "--", 6, 5);
  } else {
    c.beginPath(); c.moveTo(0, h / 2); c.lineTo(w, h / 2);
    c.strokeStyle = "rgba(255,255,255,0.14)";
    c.lineWidth = 1.2;
    c.stroke();
  }
}
function drawFrame() {
  const cw = canvasEl.width, ch = canvasEl.height;
  const sw = videoEl.videoWidth, sh = videoEl.videoHeight;
  if (!sw || !sh) return;
  const { sx, sy, sWidth, sHeight } = computeCoverRect(sw, sh, cw, ch);
  ctx.save();
  ctx.clearRect(0, 0, cw, ch);
  ctx.translate(cw, 0); ctx.scale(-1, 1);
  ctx.drawImage(videoEl, sx, sy, sWidth, sHeight, 0, 0, cw, ch);
  ctx.restore();
}
function drawHandLandmarks(landmarks) {
  const cw = canvasEl.width, ch = canvasEl.height;
  const sw = videoEl.videoWidth, sh = videoEl.videoHeight;
  if (!sw || !sh || !landmarks || landmarks.length === 0) return;
  const { sx, sy, sWidth, sHeight } = computeCoverRect(sw, sh, cw, ch);
  ctx.save();
  ctx.translate(cw, 0); ctx.scale(-1, 1);
  const conns = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17],
  ];
  for (const lm of landmarks) {
    ctx.strokeStyle = "rgba(45,212,191,0.55)";
    ctx.lineWidth = 2.5;
    for (const [a, b] of conns) {
      const pa = lm[a], pb = lm[b];
      const x1 = ((pa.x * sw - sx) / sWidth) * cw;
      const y1 = ((pa.y * sh - sy) / sHeight) * ch;
      const x2 = ((pb.x * sw - sx) / sWidth) * cw;
      const y2 = ((pb.y * sh - sy) / sHeight) * ch;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    for (const p of lm) {
      const px = ((p.x * sw - sx) / sWidth) * cw;
      const py = ((p.y * sh - sy) / sHeight) * ch;
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#2dd4bf";
      ctx.fill();
    }
  }
  ctx.restore();
}
function resizeCanvas() { canvasEl.width = window.innerWidth; canvasEl.height = window.innerHeight; }

// 归一化坐标(基于视频) → canvas 像素（含 cover 变换），用于绘制与实际触发范围对齐
function normToCanvas(nx, ny) {
  const cw = canvasEl.width, ch = canvasEl.height;
  const sw = videoEl.videoWidth, sh = videoEl.videoHeight;
  if (!sw || !sh) return { x: nx * cw, y: ny * ch };
  const { sx, sy, sWidth, sHeight } = computeCoverRect(sw, sh, cw, ch);
  return {
    x: ((nx * sw - sx) / sWidth) * cw,
    y: ((ny * sh - sy) / sHeight) * ch,
    // 归一化 1.0 对应的 canvas 像素长度（x/y 不同，因为 cover 缩放比不同）
    scaleX: cw / sWidth * sw,
    scaleY: ch / sHeight * sh,
  };
}

// ============================================================
//  主循环
// ============================================================
async function main() {
  // 🔥 iOS 提前激活音频
  synth.ensureContext();
  if (synth.ctx && synth.ctx.state === "suspended") synth.ctx.resume();
  
  await setupCamera();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  buildPiano();
  updateDegreeNotes();

  const handLandmarker = await setupHandLandmarker();
  setupFaceDetector();   // 异步加载，不影响主流程
  let lastVideoTime = -1;
  let cachedLeft = null, cachedRight = null, cachedLandmarks = [];

  function loop() {
    const now = performance.now();

    // —— 检测：只在新视频帧时跑（30fps）——
    if (videoEl.currentTime !== lastVideoTime) {
      lastVideoTime = videoEl.currentTime;
      const results = handLandmarker.detectForVideo(videoEl, now);
      cachedLandmarks = results.landmarks;

      // 按镜像后 X 分到左半屏 / 右半屏
      let bestLeftDist = Infinity, bestRightDist = Infinity;
      let newLeft = null, newRight = null;
      results.landmarks.forEach((lm) => {
        const mx = 1 - lm[0].x;
        if (mx < 0.5) {
          const d = Math.abs(mx - 0.25);
          if (d < bestLeftDist) { bestLeftDist = d; newLeft = lm; }
        } else {
          const d = Math.abs(mx - 0.75);
          if (d < bestRightDist) { bestRightDist = d; newRight = lm; }
        }
      });
      cachedLeft = newLeft;
      cachedRight = newRight;

      // —— 头部检测（头部音量开关开启时）——
      if (headVolumeEnabled && faceDetector) {
        try {
          const fr = faceDetector.detectForVideo(videoEl, now);
          if (fr.detections && fr.detections.length > 0) {
            const bb = fr.detections[0].boundingBox;
            const fw = bb.width / videoEl.videoWidth;
            faceWidthSmooth = faceWidthSmooth === null ? fw : faceWidthSmooth * 0.7 + fw * 0.3;
          } else {
            // 没检测到脸 → 渐弱到静音
            faceWidthSmooth = faceWidthSmooth === null ? null : faceWidthSmooth * 0.88;
            if (faceWidthSmooth < 0.01) faceWidthSmooth = null;
          }
        } catch {}
      }
    }

    // —— 绘制：每个 rAF 帧都完整重绘（60fps），避免半透明叠加闪烁 ——
    drawFrame();
    drawHandLandmarks(cachedLandmarks);

    const left = cachedLeft, right = cachedRight;

    // 右手食指指尖 EMA 平滑（加重平滑 0.7/0.3 → 解决闪烁）
    if (right && isFingerExtended(right, "index")) {
      const raw = { x: right[8].x, y: right[8].y };
      if (!smoothRightTip) smoothRightTip = { x: raw.x, y: raw.y };
      else {
        smoothRightTip.x = smoothRightTip.x * 0.7 + raw.x * 0.3;
        smoothRightTip.y = smoothRightTip.y * 0.7 + raw.y * 0.3;
      }
    } else {
      smoothRightTip = null;
      // 不清 locked，保持高亮稳定；出圆盘后 active 自然为 false → 不发声
    }

    const leftInfo     = left ? getLeftHandInfo(left) : null;
    const rightResult  = qualityFromTip(smoothRightTip);   // {quality, ring, active} | null
    const rightTip     = smoothRightTip;

    // 发声条件：左手有根音 + 右手在圆盘内(active) + 该环在当前家族有形态（空环不发声）
    let raw = null;
    if (leftInfo && rightResult && rightResult.active) {
      const fam = CHORD_QUALITIES[rightResult.quality];
      if (rightResult.ring < fam.rings.length) {
        raw = {
          rootName: leftInfo.rootName,
          quality: rightResult.quality,
          ring: rightResult.ring,
          rootFreq: leftInfo.rootFreq,
          degree: leftInfo.degree,
          accidental: leftInfo.accidental,
        };
      }
    }
    const st = stabilize(raw, now);

    // 绘制用稳定值：左手升降用 lockedAccidental（滞后后），右手高亮用 rightResult 的 locked 值
    const curAccidental = leftInfo ? leftInfo.accidental : lockedAccidental;
    const curDegree     = leftInfo ? leftInfo.degree : -1;

    // —— 右手拇指休止检测：拇指伸出=休止(静音)，收起=出声 ——
    const rightThumbOut = right ? isThumbExtended(right, "right") : false;

    if (st) {
      if (auditionOn) stopAudition();   // 用户开始演奏 → 试听让位
      chordPreviewOn = false;            // 和弦编辑器试听也让位
      // 兜底：形态被删除后 st.ring 可能越界，回退到第 0 形态
      const fam = CHORD_QUALITIES[st.quality];
      const ringIdx = Math.min(st.ring, fam.rings.length - 1);
      const name = chordName(st.rootName, st.quality, ringIdx);
      chordDisplayEl.textContent = name;
      const ring = fam.rings[ringIdx];
      // 衔接模式下实际音高不是固定 voicing 排列，不显示 voicing 后缀
      const voicingTxt = (!voiceLeadEnabled && ring.voicing && ring.voicing !== "close")
        ? ` · ${Array.isArray(ring.voicing) ? "自定义" : ring.voicing === "drop24" ? "drop2&4" : ring.voicing}` : "";
      const thumbTip = rightThumbOut ? " · 🔇 休止" : "";
      qualityDisplayEl.textContent = `${ring.label} · 根音 ${st.rootName}${voicingTxt}${thumbTip}`;
      filterDisplayEl.textContent = `属性: ${ring.label}${voicingTxt}`;
      // 就近衔接模式：以上一次实际演奏的音为锚点，各声部就近移动；优先使用该形态选定的 voicing
      let freqs;
      if (voiceLeadEnabled && lastChordFreqs) {
        freqs = voiceLeadFreqs(st.rootFreq, ring.intervals, lastChordFreqs, ring.voicing);
      } else {
        freqs = buildChordFreqs(st.rootFreq, st.quality, ringIdx);
      }
      lastChordFreqs = freqs.slice().sort((a, b) => a - b);
      synth.playNotes(freqs);
      lightPiano(freqs);
      updateDegreeBar(st.degree, st.accidental);
      updateAccidentalHint(st.accidental);
    } else {
      chordDisplayEl.textContent = "--";
      qualityDisplayEl.textContent = leftInfo
        ? "右手伸出食指指圆盘选属性"
        : (left ? "左手做手势选根音" : "左手做手势选根音 · 右手食指指圆盘选属性");
      filterDisplayEl.textContent = "属性: --";
      if (!auditionOn && !chordPreviewOn) {   // 试听/和弦预览中不打断发声
        synth.stop();
        lastChordFreqs = null;
      }
      if (!auditionOn && !chordPreviewOn) lightPiano([]);
      updateDegreeBar(curDegree, curAccidental ?? 0);
      updateAccidentalHint(curAccidental ?? 0);
    }

    // —— 音量应用：头部控制 or 滑块，再叠加拇指休止 ——
    const sliderVol = Number(volumeSlider.value) / 100;
    let headFactor = null;
    let targetVolume = sliderVol;
    if (headVolumeEnabled) {
      headFactor = headVolumeFromFaceW(faceWidthSmooth);
      targetVolume = sliderVol * headFactor;   // 滑块做上限，头部距离做动态系数
      if (headVolumeMeterEl) {
        const pct = Math.round(headFactor * 100);
        headVolumeMeterEl.textContent = `头部音量 ${pct}%`;
      }
    } else if (headVolumeMeterEl) {
      headVolumeMeterEl.textContent = "";
    }
    // 拇指休止：直接静音（制音/休止效果，保持和弦状态）
    if (rightThumbOut) targetVolume = 0;
    if (Math.abs(targetVolume - lastAppliedVolume) > 0.01) {
      synth.setVolume(targetVolume);
      lastAppliedVolume = targetVolume;
    }

    // 更新右上角音量进度条：头部模式直接反映头部距离(0-100%)，休止归零
    if (volumeBarFillEl) {
      let displayVol;
      if (rightThumbOut) displayVol = 0;
      else if (headVolumeEnabled) displayVol = headFactor ?? 0;
      else displayVol = sliderVol;
      const pct = Math.round(displayVol * 100);
      volumeBarFillEl.style.width = pct + "%";
      volumeBarFillEl.classList.toggle("muted", rightThumbOut);
    }

    drawAccidentalZones(curAccidental, now);
    drawQualityWheel(rightResult, rightTip, now);
    drawScope();
    requestAnimationFrame(loop);
  }
  loop();
}

// ============================================================
//  UI 事件
// ============================================================
startOverlayEl.addEventListener("click", () => {
    // 🔥 iPadOS 18 终极裸奔方案：绝对同步，无任何异步包裹
    if (!synth.ctx) {
        synth.ctx = new (window.AudioContext || window.webkitAudioContext)();
        console.log("✅ AudioContext 已创建");
    }
    if (synth.ctx.state === "suspended") {
        synth.ctx.resume();
        console.log("✅ AudioContext 已恢复");
    }
    // 二次强推（某些情况下需要两次 resume）
    if (synth.ctx.state === "suspended") {
        synth.ctx.resume();
    }
    synth.setVolume(Number(volumeSlider.value) / 100);
    startOverlayEl.style.display = "none";
    canvasEl.classList.remove("dimmed");
});


volumeSlider.addEventListener("input", () => synth.setVolume(Number(volumeSlider.value) / 100));
if (headVolumeToggleEl) {
  headVolumeToggleEl.addEventListener("change", () => {
    headVolumeEnabled = headVolumeToggleEl.checked;
    volumeSlider.disabled = headVolumeEnabled;
    if (headVolumeEnabled) {
      faceWidthSmooth = null;   // 重置，重新检测
      if (!faceDetector) setupFaceDetector();
    } else {
      synth.setVolume(Number(volumeSlider.value) / 100);
    }
  });
}
const voiceLeadToggleEl = document.getElementById("voiceLeadToggle");
if (voiceLeadToggleEl) {
  voiceLeadToggleEl.addEventListener("change", () => {
    voiceLeadEnabled = voiceLeadToggleEl.checked;
    lastChordFreqs = null;   // 切换模式时重置锚点，重新起手用标准 voicing
  });
}
// 控制面板折叠开关（默认折叠，给左侧升降区让位）
const controlsEl = document.getElementById("controls");
const controlsToggleEl = document.getElementById("controlsToggle");
if (controlsEl && controlsToggleEl) {
  controlsToggleEl.addEventListener("click", () => {
    controlsEl.classList.toggle("collapsed");
    if (!controlsEl.classList.contains("collapsed")) controlsEl.classList.remove("collapsed");
  });
}
helpButton.addEventListener("click", () => helpModal.classList.remove("hidden"));
closeHelp.addEventListener("click", (e) => { e.stopPropagation(); helpModal.classList.add("hidden"); });
helpModal.addEventListener("click", (e) => { if (e.target === helpModal) helpModal.classList.add("hidden"); });
keySelectEl.addEventListener("change", () => {
  currentTonicFreq = Number(keySelectEl.value);
  currentKeyName = keySelectEl.selectedOptions[0].dataset.note;
  updateDegreeNotes();
});
toneSelectEl.addEventListener("change", () => {
  currentTone = toneSelectEl.value;
  currentToneDef = getToneDef(currentTone);
  synth.currentKey = null;
  synth.applyToneParams(currentToneDef.fx);
  loadToneToEditor(currentTone);   // 若编辑器已打开则同步
});

// Tremolo 控制（音量颤音 + 立体声 L/R 反相）
const tremoloToggleEl = document.getElementById("tremoloToggle");
const tremoloDepthEl = document.getElementById("tremoloDepth");
const tremoloRateEl = document.getElementById("tremoloRate");
if (tremoloToggleEl) {
  tremoloToggleEl.addEventListener("change", () => {
    tremoloOn = tremoloToggleEl.checked;
    synth.applyTremolo();
  });
}
if (tremoloDepthEl) {
  tremoloDepthEl.addEventListener("input", () => {
    tremoloDepth = Number(tremoloDepthEl.value) / 100;   // 0~100 → 0~1
    synth.applyTremolo();
  });
}
if (tremoloRateEl) {
  tremoloRateEl.addEventListener("input", () => {
    tremoloRate = Number(tremoloRateEl.value);
    synth.applyTremolo();
  });
}

// 主控三段 EQ（设置面板滑杆，Roland 风格 low/mid/high）
function bindMasterEQ(id, key) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("input", () => {
    masterEQ[key] = Number(el.value);
    const valEl = el.closest(".eq-ctrl")?.querySelector(".eq-val");
    if (valEl) valEl.textContent = (masterEQ[key] > 0 ? "+" : "") + masterEQ[key].toFixed(1) + " dB";
    synth.applyMasterEQ(masterEQ);
  });
}
bindMasterEQ("masterLowGain", "lowGain");
bindMasterEQ("masterMidGain", "midGain");
bindMasterEQ("masterHighGain", "highGain");

// 手势说明按钮（展开/收起说明面板）
const guideToggleBtn = document.getElementById("guideToggle");
if (guideToggleBtn && gestureGuideEl) {
  guideToggleBtn.addEventListener("click", () => {
    gestureGuideEl.classList.toggle("hidden");
  });
}

// ============================================================
//  音色编辑器（新建 / 覆盖 / 删除，localStorage 持久化）
// ============================================================
const toneEditorBtn = document.getElementById("toneEditorBtn");
const toneModal = document.getElementById("toneModal");
const closeToneEditor = document.getElementById("closeToneEditor");
const toneLibSelect = document.getElementById("toneLibSelect");
const toneNameInput = document.getElementById("toneNameInput");
const toneBaseSelect = document.getElementById("toneBaseSelect");
const toneNewBtn = document.getElementById("toneNewBtn");
const toneDeleteBtn = document.getElementById("toneDeleteBtn");
const toneSaveBtn = document.getElementById("toneSaveBtn");
const toneResetBtn = document.getElementById("toneResetBtn");
const toneAuditionBtn = document.getElementById("toneAuditionBtn");

// 效果器参数旋钮元数据
const FX_META = {
  lowGain:   { label: "低音 Low",  min: -12,  max: 12,   step: 0.5 },
  midGain:   { label: "中音 Mid",  min: -12,  max: 12,   step: 0.5 },
  highGain:  { label: "高音 High", min: -12,  max: 12,   step: 0.5 },
  cutoff:    { label: "低通 Cutoff", min: 200,  max: 8000, step: 10 },
  q:         { label: "共振 Res",  min: 0.1,  max: 10,   step: 0.1 },
  attack:    { label: "起音 Attack", min: 0,    max: 0.5,  step: 0.01 },
  release:   { label: "释音 Release", min: 0,    max: 0.5,  step: 0.01 },
  delayTime: { label: "延迟 Time", min: 0,    max: 0.8,  step: 0.01 },
  delayFb:   { label: "反馈 Fb",   min: 0,    max: 0.7,  step: 0.01 },
  reverb:    { label: "混响 Rev",  min: 0,    max: 0.7,  step: 0.01 },
};
let toneFxValues = { ...FX_DEFAULTS };   // 当前编辑中的参数
const toneKnobs = {};                    // key → { set(v) } 旋钮实例
let editingToneId = null;   // "triangle" | "user:<id>"

function fxText(key, v) {
  if (key === "cutoff") return Math.round(v) + " Hz";
  if (key === "lowGain" || key === "midGain" || key === "highGain") return (v > 0 ? "+" : "") + Number(v).toFixed(1) + " dB";
  if (key === "attack" || key === "release" || key === "delayTime") return Number(v).toFixed(2) + "s";
  return Number(v).toFixed(2);
}
// 生成一个旋钮控件（SVG：底圆 + 270° 弧 + 指针；垂直拖动调节，双击回中）
function makeKnob(key, meta) {
  const NS = "http://www.w3.org/2000/svg";
  const wrap = document.createElement("div");
  wrap.className = "knob";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 64 64");
  svg.classList.add("knob-svg");
  const mkEl = (tag, cls) => { const e = document.createElementNS(NS, tag); e.setAttribute("class", cls); return e; };
  const body = mkEl("circle", "knob-body"); body.setAttribute("cx", 32); body.setAttribute("cy", 32); body.setAttribute("r", 26);

  // 用显式 path 画 270° 弧（从 -135° 到 +135°，顺时针），避免 circle 元素 stroke-dasharray
  // 方向不确定导致的“进度弧与指针不同位”bug。pathLength=100 后，dasharray 直接按百分比画。
  const R = 26, CX = 32, CY = 32;
  const a0 = -135 * Math.PI / 180, a1 = 135 * Math.PI / 180;
  const x0 = CX + R * Math.cos(a0), y0 = CY + R * Math.sin(a0);
  const x1 = CX + R * Math.cos(a1), y1 = CY + R * Math.sin(a1);
  const arcPath = `M ${x0.toFixed(3)} ${y0.toFixed(3)} A ${R} ${R} 0 1 1 ${x1.toFixed(3)} ${y1.toFixed(3)}`;

  const track = mkEl("path", "knob-track"); track.setAttribute("d", arcPath); track.setAttribute("pathLength", 100);
  track.style.strokeDasharray = "75 100";
  const arc = mkEl("path", "knob-arc"); arc.setAttribute("d", arcPath); arc.setAttribute("pathLength", 100);
  arc.style.strokeDasharray = "0 100";
  const ptr = mkEl("line", "knob-ptr"); ptr.setAttribute("x1", 32); ptr.setAttribute("y1", 32); ptr.setAttribute("x2", 17.86); ptr.setAttribute("y2", 17.86);
  svg.append(body, track, arc, ptr);
  const labelEl = document.createElement("div"); labelEl.className = "knob-label"; labelEl.textContent = meta.label;
  const valEl = document.createElement("div"); valEl.className = "knob-val";
  wrap.append(svg, labelEl, valEl);

  let val = meta.min;
  const paint = (v) => {
    const frac = Math.min(1, Math.max(0, (v - meta.min) / (meta.max - meta.min)));
    arc.style.strokeDasharray = `${frac * 75} 100`;
    // 指针直接按同一角度公式计算端点，不再用 SVG rotate（避免 transform 中心/方向不一致）
    const ang = (-135 + frac * 270) * Math.PI / 180;
    const px = 32 + 20 * Math.cos(ang);
    const py = 32 + 20 * Math.sin(ang);
    ptr.setAttribute("x2", px.toFixed(3));
    ptr.setAttribute("y2", py.toFixed(3));
    valEl.textContent = fxText(key, v);
  };
  const set = (v, commit) => {
    v = Number(v);
    if (Number.isFinite(meta.step)) v = Math.round(v / meta.step) * meta.step;
    v = Math.min(meta.max, Math.max(meta.min, v));
    v = Math.round(v * 1000) / 1000;
    val = v; toneFxValues[key] = v;
    paint(v);
    if (commit) synth.applyToneParams({ ...toneFxValues });
  };
  // 拖动
  let dragging = false, startY = 0, startVal = 0;
  svg.addEventListener("pointerdown", (e) => {
    dragging = true; startY = e.clientY; startVal = val;
    svg.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  svg.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const range = meta.max - meta.min;
    const dv = -(e.clientY - startY) / 120 * range;
    set(startVal + dv, true);   // 拖动中实时试听
  });
  const endDrag = () => { dragging = false; };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("dblclick", () => set(FX_DEFAULTS[key], true));
  document.getElementById("toneKnobs").appendChild(wrap);
  toneKnobs[key] = { set };
  set(toneFxValues[key], false);
}
Object.keys(FX_META).forEach((key) => makeKnob(key, FX_META[key]));

function refreshToneSelect(keep) {
  toneSelectEl.innerHTML = "";
  for (const key of Object.keys(BASE_TONES)) {
    const opt = document.createElement("option");
    opt.value = key; opt.textContent = BASE_TONES[key].name;
    toneSelectEl.appendChild(opt);
  }
  userTones.forEach((u) => {
    const opt = document.createElement("option");
    opt.value = "user:" + u.id; opt.textContent = u.name;
    toneSelectEl.appendChild(opt);
  });
  toneSelectEl.value = keep || currentTone;
}
function refreshToneLibSelect() {
  toneLibSelect.innerHTML = "";
  for (const key of Object.keys(BASE_TONES)) {
    const opt = document.createElement("option");
    opt.value = key; opt.textContent = BASE_TONES[key].name + "（内置）";
    toneLibSelect.appendChild(opt);
  }
  userTones.forEach((u) => {
    const opt = document.createElement("option");
    opt.value = "user:" + u.id; opt.textContent = u.name;
    toneLibSelect.appendChild(opt);
  });
  if (editingToneId) toneLibSelect.value = editingToneId;
}
function readFxFromForm() { return { ...toneFxValues }; }
function writeFxToForm(fx) {
  toneFxValues = { ...FX_DEFAULTS, ...fx };
  for (const key of Object.keys(FX_META)) toneKnobs[key].set(toneFxValues[key], false);
}
function loadToneToEditor(id) {
  editingToneId = id;
  const def = getToneDef(id);
  toneNameInput.value = def.name;
  toneBaseSelect.value = def.base;
  writeFxToForm(def.fx);
  if (toneDeleteBtn) toneDeleteBtn.disabled = !id.startsWith("user:");
  refreshToneLibSelect();
  updateToneHint();
}
function commitTone(asNew) {
  const fx = readFxFromForm();
  const name = toneNameInput.value.trim();
  if (!asNew && editingToneId && editingToneId.startsWith("user:")) {
    const u = userTones.find((t) => t.id === editingToneId.slice(5));
    if (u) { u.name = name || u.name; u.base = toneBaseSelect.value; u.fx = fx; }
    currentTone = editingToneId;
  } else {
    const id = "t" + Date.now().toString(36);
    userTones.push({ id, name: name || "新音色", base: toneBaseSelect.value, fx });
    currentTone = "user:" + id;
  }
  saveUserTones();
  refreshToneSelect();
  currentToneDef = getToneDef(currentTone);
  synth.currentKey = null;
  synth.applyToneParams(currentToneDef.fx);
  loadToneToEditor(currentTone);
}
toneEditorBtn.addEventListener("click", () => { loadToneToEditor(currentTone); toneModal.classList.remove("hidden"); });
closeToneEditor.addEventListener("click", () => toneModal.classList.add("hidden"));
toneModal.addEventListener("click", (e) => { if (e.target === toneModal) toneModal.classList.add("hidden"); });
toneLibSelect.addEventListener("change", () => {
  currentTone = toneLibSelect.value;
  currentToneDef = getToneDef(currentTone);
  toneSelectEl.value = currentTone;
  synth.currentKey = null;
  synth.applyToneParams(currentToneDef.fx);
  loadToneToEditor(currentTone);
});
toneNewBtn.addEventListener("click", () => commitTone(true));
toneSaveBtn.addEventListener("click", () => commitTone(false));
toneDeleteBtn.addEventListener("click", () => {
  if (!editingToneId.startsWith("user:")) return;
  userTones = userTones.filter((t) => t.id !== editingToneId.slice(5));
  saveUserTones();
  currentTone = "triangle";
  currentToneDef = getToneDef(currentTone);
  refreshToneSelect();
  toneSelectEl.value = currentTone;
  synth.currentKey = null;
  synth.applyToneParams(currentToneDef.fx);
  loadToneToEditor(currentTone);
});
toneResetBtn.addEventListener("click", () => {
  writeFxToForm({ ...FX_DEFAULTS });
  synth.applyToneParams(readFxFromForm());
});

// 随机试听（toggle）：点击持续响（每 1.6s 循环重触发，让起音/释音参数变化可听），再点停止
let auditionOn = false;
let auditionTimer = null;
let auditionFreqs = [];
// 和弦编辑器试听：点击琴键/试听按钮后持续发声，直到下一次演奏或关闭编辑器
let chordPreviewOn = false;
function stopChordPreview() {
  chordPreviewOn = false;
  synth.stop();
  lightPiano([]);
}
function stopAudition() {
  auditionOn = false;
  clearTimeout(auditionTimer);
  synth.stop();
  lightPiano([]);
  toneAuditionBtn.textContent = "▶ 随机试听";
}
toneAuditionBtn.addEventListener("click", () => {
  if (auditionOn) { stopAudition(); return; }
  chordPreviewOn = false;
  synth.ensureContext();
  if (synth.ctx && synth.ctx.state === "suspended") synth.ctx.resume();
  synth.applyToneParams(readFxFromForm());
  const roots = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67];
  const ivs = [[0, 4, 7], [0, 3, 7], [0, 4, 7, 11], [0, 4, 7, 10], [0, 3, 7, 10], [0, 4, 7, 11, 14], [0, 5, 7], [0, 4, 10, 21]];
  const r = roots[Math.floor(Math.random() * roots.length)];
  const iv = ivs[Math.floor(Math.random() * ivs.length)];
  auditionFreqs = iv.map((i) => 440 * Math.pow(2, (r + i - 69) / 12));
  synth.currentKey = null;   // 强制重播
  synth.playNotes(auditionFreqs);
  lightPiano(auditionFreqs);
  auditionOn = true;
  toneAuditionBtn.textContent = "■ 停止试听";
  const retrig = () => {
    if (!auditionOn) return;
    synth.currentKey = null;             // 重触发：前音按 release 淡出，新音按 attack 淡入
    synth.playNotes(auditionFreqs);
    lightPiano(auditionFreqs);
    auditionTimer = setTimeout(retrig, 1600);
  };
  clearTimeout(auditionTimer);
  auditionTimer = setTimeout(retrig, 1600);
});

// 波形提示：谐波少的波形对低通/起音不敏感，提醒切换试听波形
const toneHintEl = document.getElementById("toneHint");
const RICH_HARMONICS = new Set(["sawtooth", "square", "bell", "organ", "pad"]);
function updateToneHint() {
  if (!toneHintEl) return;
  if (editingToneId && !RICH_HARMONICS.has(getToneDef(editingToneId).base)) {
    toneHintEl.textContent = "提示：纯净/暖色波形谐波很少，低通与起音变化不明显——可切到 明亮/铃音/风琴/铺底 感受明显差异";
  } else {
    toneHintEl.textContent = "";
  }
}

// ============================================================
//  和弦编辑器（家族形态增删改，voicing 可选，localStorage 持久化）
// ============================================================
const chordEditorBtn = document.getElementById("chordEditorBtn");
const chordModal = document.getElementById("chordModal");
const closeChordEditor = document.getElementById("closeChordEditor");
const chordFamilySelect = document.getElementById("chordFamilySelect");
const chordRingList = document.getElementById("chordRingList");
const chordPianoEl = document.getElementById("chordPiano");
const chordAddRing = document.getElementById("chordAddRing");
const chordSaveBtn = document.getElementById("chordSave");
const chordResetBtn = document.getElementById("chordReset");
let editingFamily = "maj";
let editingRingIndex = 0;

const CHORD_VOICING_OPTIONS = [
  ["close", "原位 close"],
  ["drop2", "drop2"],
  ["drop3", "drop3"],
  ["drop24", "drop2&4"],
  ["drop23", "drop2&3"],
  ["custom", "自定义排列"],
];
const CHORD_TONE_CHIPS = [
  ["R", 0], ["b3", 3], ["3", 4], ["4", 5], ["b5", 6], ["5", 7], ["#5", 8], ["6", 9],
  ["b7", 10], ["7", 11], ["b9", 13], ["9", 14], ["#9", 15], ["11", 17], ["#11", 18], ["b13", 20], ["13", 21],
];
const DROP_KEYS = new Set(["drop2", "drop3", "drop24", "drop23"]);

function getEditingRing() {
  const fam = CHORD_QUALITIES[editingFamily];
  if (!fam) return null;
  if (editingRingIndex >= fam.rings.length) editingRingIndex = Math.max(0, fam.rings.length - 1);
  return fam.rings[editingRingIndex] || null;
}
// 有序解析 voicing 自定义排列（保留顺序，用于转位/开放排列）
function parseOrderedVoicing(str) {
  const out = [];
  for (const raw of str.trim().split(/[\s,，]+/)) {
    const tok = raw.trim().toLowerCase();
    if (!tok) continue;
    let semi = CHORD_TONE_TO_SEMI[tok];
    if (semi === undefined) {
      const n = Number(tok);
      if (Number.isFinite(n) && n >= 0 && n <= 36) semi = Math.round(n);
    }
    if (semi !== undefined && !out.includes(semi)) out.push(semi);
  }
  return out;
}
function voicingToSelectValue(voicing) {
  if (Array.isArray(voicing)) return "custom";
  return DROP_KEYS.has(voicing) || voicing === "close" ? voicing : "custom";
}
function playCurrentRing() {
  if (auditionOn) stopAudition();
  const ring = getEditingRing();
  if (!ring || ring.intervals.length === 0) return;
  const offsets = applyVoicing(ring.intervals, ring.voicing);
  if (!offsets.length) return;
  synth.ensureContext();
  if (synth.ctx && synth.ctx.state === "suspended") synth.ctx.resume();
  synth.currentKey = null;   // 强制重播，确保同一个和弦可反复试听
  const freqs = offsets.map((o) => midiToFreq(60 + o));
  synth.playNotes(freqs);
  lightPiano(freqs);
  chordPreviewOn = true;      // 保持发声，避免主循环每帧 stop 把试听打断
}
function updateRingPreview(prevEl, ring) {
  if (!prevEl) return;
  const offsets = applyVoicing(ring.intervals, ring.voicing);
  prevEl.textContent = offsets.length
    ? "预览 " + offsets.map((n) => midiNoteName(60 + n)).join(" · ")
    : "预览 —（先在下方键盘选音）";
}
function voicingMatchesRing(ring, sh) {
  if (sh.key === "close") return ring.voicing === "close";
  if (DROP_KEYS.has(sh.key)) return ring.voicing === sh.key;
  return Array.isArray(ring.voicing) && ring.voicing.join(",") === sh.offsets.join(",");
}
function applyVoicingSuggestion(ring, sh) {
  if (sh.key === "close") ring.voicing = "close";
  else if (DROP_KEYS.has(sh.key)) ring.voicing = sh.key;
  else ring.voicing = sh.offsets.slice();   // 转位 / rootless / spread 等存为自定义排列
}
function renderToneChips(row, ring) {
  const box = document.createElement("div");
  box.className = "ring-tones";
  const title = document.createElement("span");
  title.className = "suggest-title";
  title.textContent = "和弦内音（点击切换）";
  box.appendChild(title);
  CHORD_TONE_CHIPS.forEach(([label, semi]) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "ring-tone-chip" + (ring.intervals.includes(semi) ? " on" : "");
    chip.textContent = label;
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      const r = getEditingRing();
      if (!r) return;
      if (r.intervals.includes(semi)) r.intervals = r.intervals.filter((x) => x !== semi);
      else r.intervals.push(semi);
      r.intervals.sort((a, b) => a - b);
      renderRingList();
      playCurrentRing();
    });
    box.appendChild(chip);
  });
  row.appendChild(box);
}
function renderVoicingSuggestions(row, ring, customInput) {
  const box = document.createElement("div");
  box.className = "ring-voicing-suggest";
  const title = document.createElement("span");
  title.className = "suggest-title";
  title.textContent = "voicing 候选（点击即用；也可在下方键盘选音后点 ✎ 自定义）";
  box.appendChild(title);
  suggestVoicings(ring.intervals).forEach((sh) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "ring-suggest-chip" + (voicingMatchesRing(ring, sh) ? " on" : "");
    chip.textContent = sh.label;
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      const r = getEditingRing();
      if (!r) return;
      applyVoicingSuggestion(r, sh);
      renderRingList();
      playCurrentRing();
    });
    box.appendChild(chip);
  });
  // 自定义排列 chip：点击后显示/聚焦输入框
  const customChip = document.createElement("button");
  customChip.type = "button";
  customChip.className = "ring-suggest-chip" + (Array.isArray(ring.voicing) ? " on" : "");
  customChip.textContent = "✎ 自定义排列";
  customChip.addEventListener("click", (e) => {
    e.stopPropagation();
    const r = getEditingRing();
    if (!r) return;
    if (!Array.isArray(r.voicing)) {
      const arr = parseOrderedVoicing(customInput ? customInput.value : "");
      r.voicing = arr.length ? arr : r.intervals.slice();
    }
    renderRingList();
    const input = chordRingList.querySelector(".ring-edit.selected .ring-custom-voicing");
    if (input) input.focus();
  });
  box.appendChild(customChip);
  row.appendChild(box);
}
function renderRingList() {
  chordRingList.innerHTML = "";
  const fam = CHORD_QUALITIES[editingFamily];
  if (!fam) return;
  if (editingRingIndex >= fam.rings.length) editingRingIndex = Math.max(0, fam.rings.length - 1);

  fam.rings.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "ring-edit" + (i === editingRingIndex ? " selected" : "");
    row.addEventListener("click", (e) => {
      if (e.target.closest("input, select, button")) return;
      editingRingIndex = i;
      renderRingList();
    });

    // 第 1 行：命名 + 试听/删除
    const head = document.createElement("div");
    head.className = "ring-head";
    const index = document.createElement("span");
    index.className = "ring-index";
    index.textContent = "形态 " + (i + 1);
    const lab = document.createElement("input");
    lab.className = "ring-label";
    lab.value = r.label;
    lab.placeholder = "命名，如 Maj9";
    lab.title = "形态名称（会显示在圆盘上）";
    lab.addEventListener("input", (e) => {
      e.stopPropagation();
      r.label = lab.value.trim() || "形态";
    });
    const audition = document.createElement("button");
    audition.type = "button";
    audition.className = "ring-audition";
    audition.textContent = "▶ 试听";
    audition.title = "用当前音色试听这个 voicing";
    audition.addEventListener("click", (e) => {
      e.stopPropagation();
      editingRingIndex = i;
      playCurrentRing();
    });
    const del = document.createElement("button");
    del.type = "button";
    del.className = "ring-del";
    del.textContent = "✕";
    del.title = "删除形态";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      CHORD_QUALITIES[editingFamily].rings.splice(i, 1);
      editingRingIndex = Math.min(editingRingIndex, CHORD_QUALITIES[editingFamily].rings.length - 1);
      renderRingList();
    });
    head.append(index, lab, audition, del);

    // 第 2 行：和弦内音输入（自动转成 b3/b7/9/11/13 记法）
    const iv = document.createElement("input");
    iv.className = "ring-intervals";
    iv.value = formatIntervals(r.intervals);
    iv.placeholder = "R 3 5 b7 9 11 13";
    iv.title = "和弦内音记法：R b3 3 4 b5 5 #5 6 b7 7 b9 9 #9 11 #11 b13 13";
    iv.addEventListener("input", (e) => {
      e.stopPropagation();
      const parsed = parseIntervals(iv.value);
      r.intervals = parsed;
      updateRingPreview(row.querySelector(".ring-preview"), r);
    });
    iv.addEventListener("change", (e) => {
      e.stopPropagation();
      r.intervals = parseIntervals(iv.value);
      iv.value = formatIntervals(r.intervals);
      renderRingList();
      playCurrentRing();
    });

    // 第 3 行：voicing 由下方候选 chips 选择；自定义排列时显示输入框
    const customInput = document.createElement("input");
    customInput.className = "ring-custom-voicing";
    customInput.placeholder = "自定义排列，如 0 7 4 10";
    customInput.value = Array.isArray(r.voicing) ? r.voicing.join(" ") : "";
    customInput.style.display = Array.isArray(r.voicing) ? "" : "none";
    customInput.addEventListener("change", (e) => {
      e.stopPropagation();
      const arr = parseOrderedVoicing(customInput.value);
      r.voicing = arr.length ? arr : "close";
      renderRingList();
      playCurrentRing();
    });

    const prev = document.createElement("div");
    prev.className = "ring-preview";

    row.append(head, iv, customInput, prev);
    chordRingList.appendChild(row);
    renderToneChips(row, r);
    renderVoicingSuggestions(row, r, customInput);
    updateRingPreview(prev, r);
  });
  syncChordPiano();
}
// 底部键盘：C4~A5，点击切换当前形态的和弦内音（实时试听）
function buildChordPiano() {
  chordPianoEl.innerHTML = "";
  for (let m = 60; m <= 81; m++) {
    const key = document.createElement("div");
    key.className = "cp-key " + (WHITE_SET.has(m % 12) ? "white" : "black");
    key.textContent = midiNoteName(m);
    key.dataset.midi = String(m);
    key.addEventListener("click", () => {
      const ring = getEditingRing();
      if (!ring) return;
      const semi = m - 60;
      if (ring.intervals.includes(semi)) ring.intervals = ring.intervals.filter((x) => x !== semi);
      else ring.intervals.push(semi);
      ring.intervals.sort((a, b) => a - b);
      renderRingList();
      playCurrentRing();
    });
    chordPianoEl.appendChild(key);
  }
  syncChordPiano();
}
function syncChordPiano() {
  if (!chordPianoEl) return;
  const ring = getEditingRing();
  const onSet = new Set(ring ? ring.intervals : []);
  chordPianoEl.querySelectorAll(".cp-key").forEach((el) => {
    const m = Number(el.dataset.midi);
    el.classList.toggle("on", onSet.has(m - 60));
  });
}
chordEditorBtn.addEventListener("click", () => {
  editingFamily = "maj";
  editingRingIndex = 0;
  chordFamilySelect.innerHTML = "";
  QUALITY_ORDER.forEach((k) => {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = CHORD_QUALITIES[k].label;
    chordFamilySelect.appendChild(opt);
  });
  chordFamilySelect.value = editingFamily;
  buildChordPiano();
  renderRingList();
  chordModal.classList.remove("hidden");
});
closeChordEditor.addEventListener("click", () => {
  if (chordPreviewOn) stopChordPreview();
  chordModal.classList.add("hidden");
});
chordModal.addEventListener("click", (e) => {
  if (e.target === chordModal) {
    if (chordPreviewOn) stopChordPreview();
    chordModal.classList.add("hidden");
  }
});
chordFamilySelect.addEventListener("change", () => {
  editingFamily = chordFamilySelect.value;
  editingRingIndex = 0;
  renderRingList();
});
chordAddRing.addEventListener("click", () => {
  const fam = CHORD_QUALITIES[editingFamily];
  fam.rings.push({ label: "新形态", intervals: [0, 4, 7], voicing: "close" });
  editingRingIndex = fam.rings.length - 1;
  renderRingList();
});
chordSaveBtn.addEventListener("click", () => {
  saveChordLib();
  chordSaveBtn.textContent = "已保存 ✓";
  setTimeout(() => { chordSaveBtn.textContent = "保存"; }, 1200);
});
chordResetBtn.addEventListener("click", () => {
  CHORD_QUALITIES[editingFamily].rings = JSON.parse(JSON.stringify(DEFAULT_CHORD_LIB[editingFamily].rings));
  editingRingIndex = 0;
  renderRingList();
});

refreshToneSelect();
main().catch((err) => { console.error(err); alert("初始化失败：请在 localhost/https 下访问并允许摄像头。\n" + err); });
