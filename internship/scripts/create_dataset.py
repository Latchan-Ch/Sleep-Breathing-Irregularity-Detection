import os, glob, argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def find_file(folder, keyword, skip=None):
    for f in os.listdir(folder):
        if keyword.lower() in f.lower():
            if skip and skip.lower() in f.lower():
                continue
            return os.path.join(folder, f)

def load_signal(path):
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i,l in enumerate(lines) if l.strip()=='Data:') + 1
    rows = []
    for line in lines[start:]:
        parts = line.strip().split(';')
        if len(parts) < 2: continue
        try:
            t = pd.to_datetime(parts[0].strip().replace(',','.'), format='%d.%m.%Y %H:%M:%S.%f')
            v = float(parts[1].strip())
            rows.append((t,v))
        except: continue
    return pd.DataFrame(rows, columns=['time','value']).set_index('time')

def load_events(path):
    events = []
    with open(path) as f:
        for line in f:
            if ';' not in line or line.startswith(('Signal','Start','Unit')): continue
            p = line.strip().split(';')
            if len(p) < 3: continue
            try:
                date = p[0][:10]
                times = p[0][11:].split('-')
                s = pd.to_datetime(date+' '+times[0].replace(',','.'), format='%d.%m.%Y %H:%M:%S.%f')
                e = pd.to_datetime(date+' '+times[1].replace(',','.'), format='%d.%m.%Y %H:%M:%S.%f')
                if e < s: e += pd.Timedelta(days=1)
                events.append((s, e, p[2].strip()))
            except: continue
    return events

def bandpass(sig, fs=4, lo=0.17, hi=0.4):
    nyq = fs/2
    b,a = butter(2, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def get_label(ws, we, events):
    dur = (we - ws).total_seconds()
    for s,e,typ in events:
        ov = (min(we,e) - max(ws,s)).total_seconds()
        if ov/dur > 0.5:
            return typ
    return 'Normal'
    
parser = argparse.ArgumentParser()
parser.add_argument('-in_dir',  required=True)
parser.add_argument('-out_dir', required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
all_rows = []

for pid in sorted(os.listdir(args.in_dir)):
    folder = os.path.join(args.in_dir, pid)
    if not os.path.isdir(folder): continue
    print(f'processing {pid}...')

    flow   = load_signal(find_file(folder, 'flow',   skip='event'))
    thorac = load_signal(find_file(folder, 'thorac'))
    spo2   = load_signal(find_file(folder, 'spo2'))
    events = load_events(find_file(folder, 'event'))

    df = pd.DataFrame({
        'flow':   flow.resample('250ms').mean().value,
        'thorac': thorac.resample('250ms').mean().value,
        'spo2':   spo2.resample('250ms').mean().value
    }).dropna()

    df['flow']   = bandpass(df.flow.values)
    df['thorac'] = bandpass(df.thorac.values)
    df['spo2']   = bandpass(df.spo2.values)

    ts = df.index
    vals = df.values
    W, step = 120, 60  

    for i in range(0, len(df)-W+1, step):
        win = vals[i:i+W]
        label = get_label(ts[i], ts[i+W-1], events)
        all_rows.append([pid, str(ts[i]), label] + win.T.flatten().tolist())

cols = ['participant','window_start','label']
cols += [f'flow_{i}'   for i in range(120)]
cols += [f'thorac_{i}' for i in range(120)]
cols += [f'spo2_{i}'   for i in range(120)]

out = pd.DataFrame(all_rows, columns=cols)
out.to_csv(os.path.join(args.out_dir, 'breathing_dataset.csv'), index=False)
print(f'done. {len(out)} windows')
print(out.label.value_counts())
