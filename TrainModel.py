DEBUG_MODE = False
MODEL_FILE = "BruteforceA"
CSV_FILE = "Replays/NQ-09-25.Last.csv"  # Change to your file
NUM_GENERATIONS = 100000
NUM_TRAINS = 1000
NEW_AI = False

# Strategy config (match your JavaScript)
TP_POINTS = 20.0
SL_POINTS = 10.0
TRAIL_TRIGGER = 7.0
TRAIL_OFFSET = 2.0

WINDOW_BACK = 100
LOOKAHEAD_MAX = 600

if DEBUG_MODE:
    NUM_GENERATIONS = 1
    NUM_TRAINS = 10



import numpy as np
from datetime import datetime
from learningAI import SetupEnv, StrategyConfig, ScoringModel
from AITranscribe import *



def extractFeatures(bars):
    return extract_features_raw_ohlcv(bars)

cfg = StrategyConfig()
learner = SetupEnv(cfg)
model = ScoringModel()  # add this near learner initialization
if not NEW_AI:
    model.agent.load(MODEL_FILE+".pkl")

totalProfit = 0
trialNum = 0
genInfo = {'short': 0,'none': 0,'long': 0}

def newGeneration():
    global totalProfit
    totalProfit = 0
    global genInfo
    genInfo = {'short': 0,'none': 0,'long': 0}

def trainModel(trialInfo):
    bars = trialInfo["bars"]
    rewardInfo = trialInfo["rewardInfo"]

    # features from the full window you sent
    feats = extractFeatures(bars)

    # compute rewards for each possible action at this setup

    #oldActionOdds = model.get_action_odds(feats)
    action_index = model.choose_action(feats)
    action_table = ['short','none','long']
    actionTaken = action_table[action_index+1]

    reward = rewardInfo[actionTaken]

    global trialNum
    trialNum = trialNum + 1

    global totalProfit
    totalProfit = totalProfit + reward

    global genInfo
    genInfo[actionTaken] = genInfo[actionTaken] + 1

    if actionTaken == 'none':
        reward = -1

    model.train_on_batch(
        [feats],              # list of one features dict
        [action_index],       # index (0/1/2)
        [.05*reward],             # scalar reward
        iters=1
    )
    action_index = model.choose_action(feats)
    #newActionOdds = model.get_action_odds(feats)
    #if actionTaken != 'none' or DEBUG_MODE:
        #print(trialNum,totalProfit,reward,'\t\t',actionTaken,rewardInfo)
        #print(oldActionOdds,newActionOdds)

def is_session_time(timestamp_str):
    """Check if timestamp is during trading session (9:30-15:30 EST)"""
    try:
        # Parse: 20250701 144500
        dt = datetime.strptime(timestamp_str, "%Y%m%d %H%M%S")
        minutes = dt.hour * 60 + dt.minute
        return 600 <= minutes <= 930  # 10:00 to 15:30
    except:
        return True  # If can't parse, include it

def parse_csv(filename):
    """Parse CSV and return list of bars"""
    bars = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ';' not in line:
                continue
            
            parts = line.split(';')
            if len(parts) < 5:
                continue
            
            try:
                # Parse: timestamp;open;high;low;close;volume
                dt = datetime.strptime(parts[0], "%Y%m%d %H%M%S")
                timestamp = int(dt.timestamp())
                
                bar = {
                    'time': timestamp,
                    'open': float(parts[1]),
                    'high': float(parts[2]),
                    'low': float(parts[3]),
                    'close': float(parts[4]),
                    'volume': int(parts[5]) if len(parts) > 5 else 1000
                }
                bars.append(bar)
            except Exception as e:
                print(f"Skipping line: {line[:50]}... Error: {e}")
                continue
    return bars


def simulate_long(bars, entry):
    """Simulate a long trade"""
    stop = entry - SL_POINTS
    tp = entry + TP_POINTS
    trailed = False
    
    for i in range(min(LOOKAHEAD_MAX, len(bars))):
        hi, lo = bars[i]['high'], bars[i]['low']
        
        # Check SL first
        if lo <= stop:
            return stop - entry  # Loss
        
        # Check TP
        if hi >= tp:
            return tp - entry  # Win
        
        # Trail stop
        if not trailed and hi >= entry + TRAIL_TRIGGER:
            stop = entry + TRAIL_OFFSET
            trailed = True
    
    # Timeout
    last_price = (bars[min(LOOKAHEAD_MAX-1, len(bars)-1)]['high'] + 
                  bars[min(LOOKAHEAD_MAX-1, len(bars)-1)]['low']) / 2.0
    return last_price - entry

def simulate_short(bars, entry):
    """Simulate a short trade"""
    stop = entry + SL_POINTS
    tp = entry - TP_POINTS
    trailed = False
    
    for i in range(min(LOOKAHEAD_MAX, len(bars))):
        hi, lo = bars[i]['high'], bars[i]['low']
        
        # Check SL first
        if hi >= stop:
            return entry - stop  # Loss
        
        # Check TP
        if lo <= tp:
            return entry - tp  # Win
        
        # Trail stop
        if not trailed and lo <= entry - TRAIL_TRIGGER:
            stop = entry - TRAIL_OFFSET
            trailed = True
    
    # Timeout
    last_price = (bars[min(LOOKAHEAD_MAX-1, len(bars)-1)]['high'] + 
                  bars[min(LOOKAHEAD_MAX-1, len(bars)-1)]['low']) / 2.0
    return entry - last_price

def filterTrials(bars):
    minBarsNeeded = WINDOW_BACK + 1 + LOOKAHEAD_MAX
    barRange = (len(bars) - LOOKAHEAD_MAX) - minBarsNeeded
    ret = []
    while len(ret) < NUM_TRAINS:
        trialInfo = {}
        start = int(np.round(np.random.rand() * barRange)) + minBarsNeeded - (WINDOW_BACK + 1)
        entryIndex = start + WINDOW_BACK - 1
        entry = 0
        priceMove = 0
        while priceMove < 25 and entryIndex < len(bars):
            entryIndex = entryIndex + 1
            entryBar = bars[entryIndex]
            entry = entryBar['close']
            if not is_session_time(entryBar['time']):
                continue
            priceMove = max(
                abs(entry - bars[entryIndex+1]['close']),
                abs(entry - bars[entryIndex+2]['close']),
                abs(entry - bars[entryIndex+3]['close']),
                abs(entry - bars[entryIndex+4]['close']),
                abs(entry - bars[entryIndex+5]['close'])
            )
        if entryIndex >= len(bars) or priceMove < 25:
            continue
        barsSliced = bars[slice(entryIndex - 100,entryIndex + 10)]
        barsSimulated = bars[slice(entryIndex,entryIndex + 10)]
        trialInfo['entryIndex'] = entryIndex
        trialInfo['entry'] = entry
        trialInfo['bars'] = barsSliced
        trialInfo['rewardInfo'] = {
            'short' : simulate_short(barsSimulated,entry),
            'none' : 0,
            'long' : simulate_long(barsSimulated,entry)
        }

        ret.append(trialInfo)
    return ret

def main():
    bars = parse_csv(CSV_FILE)
    print("STARTED")
    genResults = []
    for genNum in range(NUM_GENERATIONS):
        newGeneration()
        trialInfo = filterTrials(bars)
        for i in range(len(trialInfo)):
            trainModel(trialInfo[i])
        global totalProfit
        global genInfo
        genResults.append(totalProfit)
        model.agent.save(MODEL_FILE+".pkl")
        if (genNum+1) % 1000 == 0:
            model.agent.save(MODEL_FILE+str(genNum+1)+".pkl")
        print("Generation",genNum,":\t",totalProfit,genInfo)

    print("RESULTS! ==================")
    for genNum in range(NUM_GENERATIONS):
        print("Generation",genNum,genResults[genNum])

    return


if __name__ == "__main__":
    main()

