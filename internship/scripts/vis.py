import os, glob, argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
            rows.append((t, v))
        except: continue
    df = pd.DataFrame(rows, columns=['time','value']).set_index('time')
    return df
def load_events(path):
    events = []
    with open(path) as f:
        for line in f:
            if ';' not in line or line.startswith('Signal') or line.startswith('Start') or line.startswith('Unit'):
                continue
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
parser = argparse.ArgumentParser()
parser.add_argument('-name', required=True)
args = parser.parse_args()
folder = args.name
pid = os.path.basename(folder)

flow   = load_signal(find_file(folder, 'flow', skip='event'))
thorac = load_signal(find_file(folder, 'thorac'))
spo2   = load_signal(find_file(folder, 'spo2'))
events = load_events(find_file(folder, 'event'))

flow_plot   = flow[::8]
thorac_plot = thorac[::8]
chunk = pd.Timedelta(minutes=5)
t = flow_plot.index[0]
end = flow_plot.index[-1]
colors = {'Hypopnea': 'orange', 'Obstructive Apnea': 'red'}

with PdfPages(f'Visualizations/{pid}_visualization.pdf') as pdf:
    while t < end:
        t2 = t + chunk
        fig, axes = plt.subplots(3,1, figsize=(16,6), sharex=True)
        axes[0].plot(flow_plot[t:t2].index,   flow_plot[t:t2].value,   lw=0.4, color='steelblue')
        axes[1].plot(thorac_plot[t:t2].index, thorac_plot[t:t2].value, lw=0.4, color='orange')
        axes[2].plot(spo2[t:t2].index,        spo2[t:t2].value,        lw=0.6, color='gray')
        axes[0].set_ylabel('Nasal Flow')
        axes[1].set_ylabel('Thoracic')
        axes[2].set_ylabel('SpO2 (%)')
        axes[0].set_title(f'{pid}  {t.strftime("%H:%M")} - {min(t2,end).strftime("%H:%M")}')
        for s,e,typ in events:
            if e < t or s > t2: continue
            c = colors.get(typ, 'yellow')
            for ax in axes:
                ax.axvspan(s, e, alpha=0.3, color=c, label=typ)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        t = t2
print(f'saved {pid}_visualization.pdf')
