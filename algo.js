function algo(bars, entryIndex) {
  // Pre-trim bars so that either choice (long/short) can be resolved
  // using a BE±2 trailing stop that arms at ±5, plus your TP/SL.
  // Uses globals: TP_POINTS, SL_POINTS, TICK_SIZE, LOOKAHEAD_MAX (from script.js).

  const snapToTick = (p) => {
    const ts = (typeof TICK_SIZE !== 'undefined' && TICK_SIZE) ? TICK_SIZE : 0.25;
    return Math.round(p / ts) * ts;
  };

  const maxLook = (typeof LOOKAHEAD_MAX !== 'undefined' && LOOKAHEAD_MAX) ? LOOKAHEAD_MAX : 600;
  const TP = (typeof TP_POINTS !== 'undefined' && TP_POINTS) ? TP_POINTS : 30;
  const SL = (typeof SL_POINTS !== 'undefined' && SL_POINTS) ? SL_POINTS : 20;

  const entry = bars[entryIndex].close;

  function simulate(side){
    const tp = side === 'long' ? entry + TP : entry - TP;
    let sl = side === 'long' ? entry - SL : entry + SL;

    const trigger = side === 'long' ? entry + 10 : entry - 10;
    let trailed = false;

    for (let i = entryIndex + 1; i < Math.min(bars.length, entryIndex + 1 + maxLook); i++){
      const b = bars[i];
      const currentSL = sl;

      // arm trailing stop once price reaches ±5
      if (!trailed){
        const hitTrig = side === 'long' ? (b.close >= trigger) : (b.close <= trigger);
        if (hitTrig){
          sl = side === 'long' ? snapToTick(entry + 2) : snapToTick(entry - 2);
          trailed = true;
        }
      }

      // exit checks
      if (side === 'long'){
        if (b.low <= currentSL) return { outcome: trailed ? 'breakeven' : 'loss', index: i, score: currentSL - entry };
        if (b.high >= tp) return { outcome: 'win', index: i, score: tp - entry };
      } else {
        if (b.high >= currentSL) return { outcome: trailed ? 'breakeven' : 'loss', index: i, score: -currentSL + entry };
        if (b.low  <= tp) return { outcome: 'win', index: i, score: -tp + entry };
      }
    }
    // no decision within window
    return { outcome: 'timeout', index: Math.min(bars.length - 1, entryIndex + maxLook) };
  }

  const resLong  = simulate('long');
  const resShort = simulate('short');

  // Ensure enough bars to resolve either choice
  const needUntil = Math.max(resLong.index, resShort.index);
  const endIdx = Math.min(needUntil, bars.length - 1);

  return {
    bars: bars.slice(0, endIdx + 1),
    entryIndex,
    // Optional: use these to skip recomputing in evaluate()
    pre: { long: resLong, short: resShort }
  };
}
