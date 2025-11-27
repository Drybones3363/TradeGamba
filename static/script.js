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
let animTimer = null;
let rounds = [];      // array of { bars: [{time,open,high,low,close}], entryIndex }
let roundIdx = 0;
let score = 0, wins = 0, losses = 0;
let canDecide = true;
let entryMarker = null;

let bars = [];


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


  // default: build 50 random-walk rounds so you can play instantly
  //rounds = buildRandomRounds(50);
  //showRound();

  document.getElementById('btnLong').onclick = () => choose('long');
  document.getElementById('btnShort').onclick = () => choose('short');
  document.getElementById('btnSkip').onclick = () => handleSkip();
  document.getElementById('btnNext').onclick = () => nextRound();

  function handleSkip(){
    setMsg('⏭️ Skipped. Press Next to continue.');
    finalizeRoundUI();
  }


  document.addEventListener('keydown', (e)=>{
  if (e.key === 'ArrowUp'   && canDecide) choose('long');
  if (e.key === 'ArrowDown' && canDecide) choose('short');
  if (e.key === 'ArrowRight'){
    if (canDecide) {
      // treat ArrowRight as Skip before a decision
      handleSkip();
    } else {
      // after result, ArrowRight advances to next round
      nextRound();
    }
  }
});

async function handleFileInput(e) {
    const file = e.target.files?.[0]; if (!file) return;
    const text = await file.text();
    handleFile(text);
}

document.getElementById('fileInput').addEventListener('change', handleFileInput);

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


function setDecisionUI(active){
  // active=true: enable Long/Short/Skip, hide Next
  // active=false: disable Long/Short/Skip, show Next
  canDecide = active;

  document.getElementById('btnLong').disabled  = !active;
  document.getElementById('btnShort').disabled = !active;
  document.getElementById('btnSkip').disabled  = !active;

  const nextBtn = document.getElementById('btnNext');
  if (active) {
    nextBtn.style.display = 'none';
    nextBtn.disabled = true;
  } else {
    nextBtn.style.display = 'inline-block';
    nextBtn.disabled = false;
  }
}

function finalizeRoundUI(){
  // Called right after showing a result (win/loss/timeout) OR after Skip
  setDecisionUI(false);
}


function showRound(){
  const r = rounds[roundIdx];
  if (!r) { setMsg("Done! No more rounds."); return; }

  // show only history up to entryIndex (decision point)
  const start = Math.max(0, r.entryIndex - WINDOW_BACK);
  const viewBars = r.bars.slice(start, r.entryIndex + 1);

  candleSeries.setData(viewBars);
  lwcChart.timeScale().fitContent();
  volumeSeries.setData(viewBars.map(b => ({
    time: b.time,
    value: Number.isFinite(b.volume) ? b.volume : 0
  })));

  const emaData9 = computeEMA(viewBars, 9);
  ema9Series.setData(emaData9);
  const emaData21 = computeEMA(viewBars, 21);
  ema21Series.setData(emaData21);
  lwcChart.timeScale().fitContent();

  clearOutcomeLines();
  updatePills();
  setMsg("Pick Long / Short / Skip.");
  setDecisionUI(true);
  const { score } = getScore(viewBars).then((obj)=>{
    document.getElementById('aiPill').textContent = `AI: ${obj.score.toFixed(3)}`;
    document.getElementById('newAIPill').textContent = ` `;
  });

  function toList(dict) {
    let lst = new Array();
    let i = 0;
    while (dict[i]) {
      lst.push(dict[i]);
      i++;
    }
    return lst;
  }
  let aiInfo = {};
  for (let i = start;i < WINDOW_BACK + 1;i++) {
    const mainStart = r.mainEntryIndex;
    const b = bars[mainStart - 100 + i];
    const scoreBars = bars.slice(mainStart - 100 + i,mainStart + i);
    getScore(scoreBars).then((obj)=>{
      const score = obj.score;
      aiInfo[i] = {
        time: b.time,
        value: score
      };
      const aiList = toList(aiInfo);
      if (aiList.length >= 0) { //r.entryIndex + 1 - start) {
        aiSeries.setData(aiList);
        aiChart.timeScale().fitContent();
      }
    }); // Math.round(Math.random() * 200 - 100);
  }

}

function updatePills(){
  document.getElementById('roundPill').textContent = `Round: ${roundIdx + 1}`;
  document.getElementById('scorePill').textContent = `Score: ${score}`;
  document.getElementById('statsPill').textContent = `W/L: ${wins}/${losses}`;
}

function setMsg(s){ document.getElementById('message').textContent = s; }

function choose(side){
  const r = rounds[roundIdx];
  const entryBar = r.bars[r.entryIndex];
  const entry = entryBar.close;

  // draw static guides first
  drawEntryTP_SL(entry, side);

  // --- pull predetermined outcome so scoring remains deterministic ---
  const pre = r.pre ? (side === 'long' ? r.pre.long : r.pre.short) : null;
  let outcome = pre?.outcome ?? 'timeout';
  let resolveIdx = Number.isFinite(pre?.index) ? pre.index : (r.entryIndex + LOOKAHEAD_MAX);

  const rewards = {
    long: r.pre.long.score,
    short: r.pre.short.score,
    none: 0.0
  };
  let scoreDelta = Number.isFinite(pre?.score) ? pre.score : 0;

  // UI: lock decision buttons during playback, hide Next until done
  setDecisionUI(false);
  document.getElementById('btnNext').style.display = 'none';
  document.getElementById('btnNext').disabled = true;

  // playback window
  const startIdx = Math.max(0, r.entryIndex - WINDOW_BACK);
  let shownEnd = r.entryIndex + 1; // start anim just after entry

  // trailing state for the visual SL line
  const trigger = side === 'long' ? entry + 10 : entry - 10;
  const beTrail  = side === 'long' ? snapToTick(entry + 2) : snapToTick(entry - 2);
  let slActive   = side === 'long' ? entry - SL_POINTS : entry + SL_POINTS;
  let trailed    = false;

  // ensure clean slate
  if (animTimer) { clearInterval(animTimer); animTimer = null; }

  // start animation: every 0.1s reveal one more bar until resolveIdx
  animTimer = setInterval(()=>{
    // stop if we've shown everything needed
    if (shownEnd > Math.min(r.bars.length, resolveIdx + 1)) {
      clearInterval(animTimer); animTimer = null;

      // finalize W/L, score, message, and show Next
      if (outcome === 'win') wins++;
      else if (outcome === 'loss') losses++;
      score += scoreDelta;
      updatePills();

      let msg;
      if (outcome === 'win')            msg = `✅ ${side.toUpperCase()} win`;
      else if (outcome === 'loss')      msg = `❌ ${side.toUpperCase()} loss`;
      else if (outcome === 'breakeven') msg = `🟦 ${side.toUpperCase()} stopped at BE±2`;
      else                              msg = `⏱️ Timeout (no exit within ${LOOKAHEAD_MAX} bars)`;
      if (Number.isFinite(scoreDelta) && scoreDelta !== 0) {
        msg += ` · Score ${scoreDelta >= 0 ? '+' : ''}${scoreDelta}`;
      }
      setMsg(msg);

      // now allow Next
      const nextBtn = document.getElementById('btnNext');
      nextBtn.style.display = 'inline-block';
      nextBtn.disabled = false;
      return;
    }

    // extend the visible bars by one
    const slice = r.bars.slice(startIdx, shownEnd);
    candleSeries.setData(slice);
    volumeSeries.setData(slice.map(b => ({
      time: b.time,
      value: Number.isFinite(b.volume) ? b.volume : 0
    })));

    lwcChart.timeScale().fitContent();
    volumeChart.timeScale().fitContent();

    // evaluate trailing trigger & exit checks at the *new* bar
    const i = shownEnd - 1;
    if (i >= r.entryIndex + 1) {
      const b = r.bars[i];

      // arm trail if not yet
      if (!trailed) {
        const hitTrig = side === 'long' ? (b.close >= trigger) : (b.close <= trigger);
        if (hitTrig) {
          trailed = true;
          slActive = beTrail;
          // move the SL line visually
          if (slLine) candleSeries.removePriceLine(slLine);
          slLine = candleSeries.createPriceLine({
            price: slActive, color:'#ee6666', lineWidth:.2, lineStyle:0,
            title: `SL ${trailed ? 'BE' : SL_POINTS}`
          });
        }
      }

      // if we just reached the resolve bar, pin the final frame next tick
      if (i >= resolveIdx) {
        shownEnd++; // let the interval terminate on next loop
      }
    }

    shownEnd++;
  }, 100);
  sendBarsForTraining(r.bars.slice(0,r.entryIndex+1), entry, rewards).then((obj)=>{
    document.getElementById('newAIPill').textContent = `New AI: ${obj.score.toFixed(3)}`;
  });

}


function nextRound(){
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  roundIdx++;
  if (roundIdx >= rounds.length){
    setMsg("🎉 Finished all rounds. Reload or load a new file to continue.");
    return;
  }
  showRound();
}


function clearOutcomeLines(){
  [entryLine, tpLine, slLine].forEach(l => { if (l) { candleSeries.removePriceLine(l); }});
  entryLine = tpLine = slLine = null;
  candleSeries.setMarkers([]); 
  entryMarker = null;
}

function drawEntryTP_SL(entry, side){
  clearOutcomeLines();
  entryLine = candleSeries.createPriceLine({
    price: entry, color:'#6aa0ff', lineWidth:.2, lineStyle:0, title:`Entry ${entry.toFixed(2)}`
  });
  tpLine = candleSeries.createPriceLine({
    price: side === 'long' ? entry + TP_POINTS : entry - TP_POINTS,
    color:'#3ddc91', lineWidth:.2, lineStyle:2, title:`TP ${TP_POINTS}`
  });
  slLine = candleSeries.createPriceLine({
    price: side === 'long' ? entry - SL_POINTS : entry + SL_POINTS,
    color:'#ee6666', lineWidth:.2, lineStyle:2, title:`SL ${SL_POINTS}`
  });
  // entryLine = candleSeries.createPriceLine({
  //   price: entry,
  //   color:'#6666ee', lineWidth:2, lineStyle:0, title:`ENTRY ${entry}`
  // });
  // --- bright entry dot marker ---
const bar = rounds[roundIdx].bars[rounds[roundIdx].entryIndex];
if (bar && candleSeries) {
  entryMarker = [{
    time: bar.time,
    position: side === 'long' ? 'belowBar' : 'aboveBar',
    color: side === 'long' ? '#00ffcc' : '#ffcc00',
    shape: 'circle',
    size: 2,
    text: 'ENTRY'
  }];
  candleSeries.setMarkers(entryMarker);
}

}

function snapToTick(price){
  const ticks = Math.round(price / TICK_SIZE);
  return ticks * TICK_SIZE;
}

function isSessionTime(tsSec){
  // tsSec is unix seconds already (normalizeBar does this)
  const d = new Date(tsSec * 1000); // interpreted as local time
  
  // Local minutes-from-midnight
  let minutesLocal = d.getHours() * 60 + d.getMinutes();
  
  // Shift into "session" timezone (e.g., ET) using the offset.
  // Add 24h before modulo so negative offsets don't go negative.
  let minutesSession = (minutesLocal + SESSION_TZ_OFFSET_MIN + 24*60) % (24*60);

  // Now compare against *session* time (9:30–15:30)
  return minutesSession >= (9*60 + 30) && minutesSession <= (15*60 + 30);
}


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

    rounds = [];

    const minBarsNeeded = WINDOW_BACK + 1 + LOOKAHEAD_MAX;
    if (dedup.length < minBarsNeeded) {
      setMsg(`Not enough bars: have ${dedup.length}, need at least ${minBarsNeeded}. Export a larger range or lower WINDOW_BACK/LOOKAHEAD_MAX.`);
      return;
    }
    const range = (dedup.length - LOOKAHEAD_MAX) - minBarsNeeded;

    const TARGET_ROUNDS = 100;
    let attempts = 0;

    while (rounds.length < TARGET_ROUNDS && attempts < TARGET_ROUNDS * 20) {
      attempts++;

      const start = Math.round(Math.random() * range) + minBarsNeeded - (WINDOW_BACK + 1);
      const entryIndex = WINDOW_BACK;
      const mainEntryIndex = start + entryIndex;
      const entryBar = dedup[mainEntryIndex];

      if (!entryBar || !isSessionTime(entryBar.time)) continue; // ⬅ only pick 9:30–3:30 entries

      const end   = start + (WINDOW_BACK + 1) + LOOKAHEAD_MAX;
      const slice = dedup.slice(start, end);

      const info = algo(slice, entryIndex);
      info.mainEntryIndex = mainEntryIndex;
      rounds.push(info);  // uses your precomputed outcomes
    }

    if (!rounds.length) {
      setMsg("❗ No eligible rounds found between 9:30–3:30 in this file.");
      return;
    }

    roundIdx = 0;
    score = wins = losses = 0;
    showRound();
    setMsg(`Loaded ${rounds.length} session-only rounds from file (${dedup.length} bars total).`);
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
