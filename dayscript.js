/* ========= CONFIG ========= */
const TP_POINTS = 30;
const SL_POINTS = 20;
const WINDOW_BACK = 100;   // bars shown before decision point
const LOOKAHEAD_MAX = 600;  // max bars to search for outcome after entry
const TICK_SIZE = 0.25;    // NQ tick size
const SESSION_TZ_OFFSET_MIN = -240;
/* ========================= */

let lwcChart, candleSeries, entryLine, tpLine, slLine, ema9Series, ema21Series;
let volumeChart, volumeSeries;
let aiChart, aiSeries;
let bars = [];
let idx = 0;
let timer = null;


init();

function init(){
  const priceContainer  = document.getElementById('priceChart');
  const volumeContainer = document.getElementById('volumeChart');
  const aiContainer = document.getElementById('aiChart');

  lwcChart = LightweightCharts.createChart(priceContainer, {
    layout: { background: { color: '#141821' }, textColor: '#dbe2ea' },
    grid: { vertLines: { color: '#1c2333' }, horzLines: { color: '#1c2333' } },
    rightPriceScale: { borderColor: '#222b' },
    timeScale: {
      borderColor: '#222b',
      rightOffset: 30,
      rightBarStaysOnScroll: true
    },
    crosshair: { mode: 1 }
  });
  lwcChart.timeScale().applyOptions({ rightOffset: 30 });

  candleSeries = lwcChart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    borderVisible: false
  });

  // --- new volume chart below ---
  volumeChart = LightweightCharts.createChart(volumeContainer, {
    layout: { background: { color: '#141821' }, textColor: '#dbe2ea' },
    grid: { vertLines: { color: '#1c2333' }, horzLines: { color: '#1c2333' } },
    rightPriceScale: { borderColor: '#222b' },
    timeScale: {
      borderColor: '#222b',
      rightBarStaysOnScroll: true
    }
  });

  volumeSeries = volumeChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    color: '#4682B4'
  });

  aiChart = LightweightCharts.createChart(aiContainer, {
    layout: { background: { color: '#141821' }, textColor: '#dbe2ea' },
    grid: { vertLines: { color: '#1c2333' }, horzLines: { color: '#1c2333' } },
    rightPriceScale: { borderColor: '#222b' },
    timeScale: {
      borderColor: '#222b',
      rightBarStaysOnScroll: true
    }
  });

  aiSeries = aiChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    color: '#4682B4'
  });

  // sync x-axis (time) between price & volume
  const priceTimeScale = lwcChart.timeScale();
  const volTimeScale   = volumeChart.timeScale();
  const aiTimeScale   = aiChart.timeScale();

  priceTimeScale.subscribeVisibleLogicalRangeChange(logicalRange => {
    volTimeScale.setVisibleLogicalRange(logicalRange);
    aiTimeScale.setVisibleLogicalRange(logicalRange);
  });
  ema9Series = lwcChart.addLineSeries({
    color: '#00d7ff', // gold color for visibility
    lineWidth: 1,
    priceLineVisible: false,
  });
  ema21Series = lwcChart.addLineSeries({
    color: '#ffd700', // gold color for visibility
    lineWidth: 1,
    priceLineVisible: false,
  });

  // ---- on-chart hover time label ----
  const chartEl = document.getElementById('priceChart');
  const timeHint = document.createElement('div');
  timeHint.id = 'timeHint';
  Object.assign(timeHint.style, {
    position: 'absolute',
    bottom: '8px',
    left: '10px',
    padding: '6px 10px',
    borderRadius: '8px',
    background: 'rgba(20,24,33,0.85)',
    color: '#cfd6e3',
    fontWeight: '600',
    fontSize: '12px',
    pointerEvents: 'none',
    zIndex: 2,
  });
  timeHint.textContent = 'Time: —';
  chartEl.style.position = 'relative'; // ensure positioning context
  chartEl.appendChild(timeHint);

  const pad2 = (n)=> String(n).padStart(2,'0');
  function fmtTS(sec){
    const d = new Date(sec * 1000);
    let minutesLocal = d.getHours() * 60 + d.getMinutes();
    let minutesSession = (minutesLocal + SESSION_TZ_OFFSET_MIN + 24*60) % (24*60);

    const sessHour = Math.floor(minutesSession / 60);
    const sessMin  = minutesSession % 60;

    const Y = d.getFullYear(), M = pad2(d.getMonth()+1), D = pad2(d.getDate());
    const h = pad2(sessHour),  m = pad2(sessMin);
    return `${Y}-${M}-${D} ${h}:${m}`;
  }


  lwcChart.subscribeCrosshairMove(param => {
    if (!param || !param.time) { timeHint.textContent = 'Time: —'; return; }

    // lightweight-charts can give UTCTimestamp (number) or BusinessDay object
    let tsSec = null;
    if (typeof param.time === 'number') {
      tsSec = param.time;                  // we feed unix seconds in setData
    } else if (param.time.year) {
      // business day -> approximate midnight; not expected here, but just in case
      tsSec = Math.floor(Date.UTC(param.time.year, param.time.month - 1, param.time.day) / 1000);
    }
    timeHint.textContent = tsSec ? `Time: ${fmtTS(tsSec)}` : 'Time: —';
  });



  // Play
  document.getElementById('btnPlay').onclick = ()=>{
    if(timer) return;
    timer = setInterval(()=>{
      if(idx >= bars.length-1){
        clearInterval(timer); timer=null;
        setMsg("End of day.");
        return;
      }
      idx++;
      renderUpTo(idx);
      getAIScore();
    }, 250);
  };

  // Pause
  document.getElementById('btnPause').onclick = ()=>{
    if(timer){ clearInterval(timer); timer=null; }
  };

  // Skip +10
  document.getElementById('btnSkip10').onclick = ()=>{
    idx = Math.min(idx+10, bars.length-1);
    renderUpTo(idx);
    getAIScore();
  };

  // ------------------
  // Take Trade (algo)
  // ------------------
  document.getElementById('btnTrade').onclick = ()=>{
    if(idx < 100){ setMsg("Need 100 bars before entry."); return; }

    const windowBars = bars.slice(idx-100, idx+600);
    const entryIndex = 100;

    const res = algo(windowBars, entryIndex);
    const long = res.pre.long;
    const short = res.pre.short;

    setMsg(
      `Long: ${long.outcome} (${long.score.toFixed(1)} pts) — ` +
      `Short: ${short.outcome} (${short.score.toFixed(1)} pts)`
    );
  };

  async function handleFileInput(e) {
      const file = e.target.files?.[0]; if (!file) return;
      const text = await file.text();
      handleFile(text);
  }

  document.getElementById('fileInput').addEventListener('change', handleFileInput);

}

function getBarTime(bar) {
  const tsSec = bar.time
  const d = new Date(tsSec * 1000);
  let minutesLocal = d.getHours() * 60 + d.getMinutes();
  let minutesSession = (minutesLocal + SESSION_TZ_OFFSET_MIN + 24*60) % (24*60);
  return minutesSession;
}

function loadRandomDay(){
  let randomIndex = 0;

  while (getBarTime(bars[randomIndex]) != 9*60 + 30) {
    randomIndex = (randomIndex + 1) % bars.length
  }
  idx = randomIndex;

  document.getElementById("pillDay").textContent = `Day: ${randomIndex}`;
  renderUpTo(idx);
  setMsg(`Loaded day`);
}

// -----------------------------
// Replay controls
// -----------------------------
function renderUpTo(i){
  const slice = bars.slice(0, i+1);
  candleSeries.setData(slice);
  volumeSeries.setData(slice.map(b=>({time:b.time, value:b.volume||0})));
  updatePills();
}

function updatePills(){
  document.getElementById('pillIndex').textContent = `Bar: ${idx}`;
  if(bars[idx])
    document.getElementById('pillTime').textContent =
      `Time: ${new Date(bars[idx].time*1000).toLocaleTimeString()}`;
}

function computeEMA(bars, period = 21) {
  if (bars.length < period) return [];
  const ema = [];
  let k = 2 / (period + 1);
  let prev = bars[0].close;
  for (let i = 0; i < bars.length; i++) {
    const c = bars[i].close;
    prev = i === 0 ? c : (c - prev) * k + prev;
    ema.push({ time: bars[i].time, value: prev });
  }
  return ema;
}

function setMsg(s){ document.getElementById('message').textContent = s; }


async function handleFile(text){
  bars = [];

  try {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 1) throw new Error("File has no rows.");

    const split = (s) => s.split(/,|;|\t/).map(x => x.trim());
    const firstCols = split(lines[0]).map(h => h.toLowerCase());

    // Heuristics: does first row look like a header?
    const hasHeaderNames = ["time","date","timestamp","datetime","open","high","low","close","o","h","l","c"]
      .some(k => firstCols.includes(k));
    // Or does col 0 look like yyyymmdd HHmm[ss] (headerless data)?
    const looksLikeDataRow = /^\d{8}\s+\d{4,6}$/.test(firstCols[0]) || /^\d{10,13}$/.test(firstCols[0]);

    let startRow = 1;
    let idxTime, idxOpen, idxHigh, idxLow, idxClose, idxVol;

    if (hasHeaderNames && !looksLikeDataRow) {
      // Map by header names
      const header = firstCols;
      const findIdx = (names) => {
        for (const n of names) { const i = header.indexOf(n); if (i !== -1) return i; }
        return -1;
      };
      idxTime  = findIdx(["time","date","timestamp","datetime"]);
      idxOpen  = findIdx(["open","o"]);
      idxHigh  = findIdx(["high","h"]);
      idxLow   = findIdx(["low","l"]);
      idxClose = findIdx(["close","c"]);
      idxVol   = findIdx(["volume","vol","v"]);
      if ([idxTime, idxOpen, idxHigh, idxLow, idxClose].some(i => i === -1)) {
        throw new Error("Missing one of: time/date, open, high, low, close columns in header: " + firstCols.join(", "));
      }
    } else {
      // Headerless. Assume fixed order:
      // time, open, high, low, close [, volume]
      startRow = 0;
      idxTime = 0; idxOpen = 1; idxHigh = 2; idxLow = 3; idxClose = 4; idxVol = 5;
    }
    
    for (let i = startRow; i < lines.length; i++) {
      const cols = split(lines[i]);
      const row = {
        time:  cols[idxTime],
        open:  parseFloat((cols[idxOpen]  ?? "").replaceAll(",", "")),
        high:  parseFloat((cols[idxHigh]  ?? "").replaceAll(",", "")),
        low:   parseFloat((cols[idxLow]   ?? "").replaceAll(",", "")),
        close: parseFloat((cols[idxClose] ?? "").replaceAll(",", "")),
        volume: idxVol != null && idxVol !== -1 ? parseFloat((cols[idxVol] ?? "0").replaceAll(",", "")) : 0,
      };
      const nb = normalizeBar(row);
      if (nb) bars.push(nb);
    }

    if (!bars.length) throw new Error("Parsed 0 bars. Check delimiter and header names.");

    // Sort by time, dedupe
    bars.sort((a,b)=>a.time - b.time);
    const dedup = [];
    let lastT = null;
    for (const b of bars) {
      if (b.time !== lastT) { dedup.push(b); lastT = b.time; }
    }
    loadRandomDay();
  } catch (err) {
    console.error(err);
    setMsg("❗ Load error: " + err.message);
  }
}

function normalizeBar(b){
  const t = parseAnyTime(b.time);
  if (!t || !isFinite(t)) return null;

  const O = +b.open, H = +b.high, L = +b.low, C = +b.close;
  if (![O,H,L,C].every(Number.isFinite)) return null;

  // 🔒 Force volume to be a finite number
  const Vraw = +b.volume;
  const V = Number.isFinite(Vraw) ? Vraw : 0;

  return {
    time: Math.floor(t / 1000),
    open:  O,
    high:  H,
    low:   L,
    close: C,
    volume: V
  };
}

function parseAnyTime(val){
  if (val == null) return null;
  if (typeof val === "number") return val > 1e12 ? val : val * 1000;
  const s = String(val).trim();

  // epoch sec/ms
  if (/^\d{10}$/.test(s)) return parseInt(s, 10) * 1000;
  if (/^\d{13}$/.test(s)) return parseInt(s, 10);

  // yyyy-MM-dd HH:mm[:ss]
  let m = s.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?$/);
  if (m) {
    const [_, Y, M, D, h, min, sec="00"] = m;
    return new Date(`${Y}-${M}-${D}T${h}:${min}:${sec}`).getTime();
  }

  // MM/dd/yyyy HH:mm[:ss]
  m = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})[ T](\d{2}):(\d{2})(?::(\d{2}))?$/);
  if (m) {
    const [_, mo, d, Y, h, min, sec="00"] = m;
    const mm = mo.padStart(2,"0"), dd = d.padStart(2,"0");
    return new Date(`${Y}-${mm}-${dd}T${h}:${min}:${sec}`).getTime();
  }

  // yyyymmdd HHmm[ss]
  m = s.match(/^(\d{8})\s+(\d{4,6})$/);
  if (m) {
    const Y = s.slice(0,4), M = s.slice(4,6), D = s.slice(6,8);
    const tt = m[2].padEnd(6, '0'); // HHmm -> HHmm00
    const h = tt.slice(0,2), min = tt.slice(2,4), sec = tt.slice(4,6);
    return new Date(`${Y}-${M}-${D}T${h}:${min}:${sec}`).getTime();
  }

  // yyyymmddHHmm[ss] (no space)
  m = s.match(/^(\d{8})(\d{4,6})$/);
  if (m) {
    const Y = s.slice(0,4), M = s.slice(4,6), D = s.slice(6,8);
    const tt = m[2].padEnd(6, '0');
    const h = tt.slice(0,2), min = tt.slice(2,4), sec = tt.slice(4,6);
    return new Date(`${Y}-${M}-${D}T${h}:${min}:${sec}`).getTime();
  }

  // Fallback
  const t = new Date(s).getTime();
  return Number.isFinite(t) ? t : null;
}

async function loadDefault() {
  try {
    const url = 'https://raw.githubusercontent.com/Drybones3363/TradeGamba/refs/heads/main/Replays/NQ-09-25.Last.csv';
    fetch(url).then(r=>r.text()).then((text)=>{
      handleFile(text);
    });
  } catch (err) {
    console.error(err);
  }
}

loadDefault();


const API = "http://localhost:5000";

async function sendBarsForTraining(bars, entry, rewards) {
  const r = await fetch(`${API}/train`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ bars, entry, rewards })
  });
  return r.json();
}

async function getScore(bars) {
  const r = await fetch(`${API}/score`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ bars })
  });
  return r.json(); // { score }
}
