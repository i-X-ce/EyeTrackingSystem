import os
import re
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RangeSlider

# ==========================================
# 初期設定
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

COLORS = ['#2ca02c', '#ff7f0e', '#1f77b4']
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ==========================================
# データ処理関数
# ==========================================
def extract_time_and_type(directory, filename):
    match = re.search(r'\d{14}', filename)
    time_val = datetime.strptime(match.group(), "%Y%m%d%H%M%S") if match else None
    
    ui_type = "Unknown"
    if 'log' in filename.lower() and time_val:
        try:
            df = pd.read_csv(os.path.join(directory, filename))
            start_row = df[df['tag'] == 'start']
            if not start_row.empty:
                msg = str(start_row['message'].values[0])
                if 'type=view' in msg: ui_type = "View"
                elif 'type=classic' in msg: ui_type = "Classic"
                elif 'type=none' in msg: ui_type = "None"
        except: pass
    return time_val, ui_type

def auto_pair_files(directory):
    files = os.listdir(directory)
    gaze_files = [f for f in files if 'gaze' in f.lower() and f.endswith('.csv')]
    log_files = [f for f in files if 'log' in f.lower() and f.endswith('.csv')]
    
    pairs, unpaired = [], []
    log_info = {log_f: extract_time_and_type(directory, log_f) for log_f in log_files}

    for log_f, (log_time, ui_type) in log_info.items():
        if not log_time:
            unpaired.append(('log', log_f, "時間抽出不可", "Unknown"))
            continue
            
        best_match, min_diff = None, float('inf')
        for gaze_f in gaze_files:
            gaze_time, _ = extract_time_and_type(directory, gaze_f)
            if gaze_time:
                diff = abs((log_time - gaze_time).total_seconds())
                if diff <= 60 and diff < min_diff:
                    min_diff, best_match = diff, gaze_f
                    
        if best_match and min_diff <= 60:
            pairs.append((log_f, best_match, min_diff, ui_type))
            gaze_files.remove(best_match)
        else:
            unpaired.append(('log', log_f, "ペアなし(誤差大)", ui_type))
            
    for gaze_f in gaze_files: unpaired.append(('gaze', gaze_f, "ペアなし", "Unknown"))
    return sorted(pairs, key=lambda x: x[0]), unpaired

def extract_all_pupil_trajectories(df_log, df_gaze, tag='add'):
    events = df_log[df_log['tag'] == tag]['timestamp'].values
    bins = np.arange(-500, 2501, 250)
    bin_centers = bins[:-1] + 125
    trajectories = []
    for ts in events:
        window = df_gaze[(df_gaze['Time'] >= ts - 500) & (df_gaze['Time'] <= ts + 2500)].copy()
        if not window.empty:
            window['rel_time'] = window['Time'] - ts
            window['bin'] = pd.cut(window['rel_time'], bins=bins, labels=bin_centers)
            binned = window.groupby('bin', observed=False)['LeftPupil'].mean().reset_index()
            binned['bin'] = binned['bin'].astype(float)
            trajectories.append(binned.set_index('bin')['LeftPupil'])
    return trajectories

def calculate_gaze_stability(df_log, df_gaze, tag='add'):
    events = df_log[df_log['tag'] == tag]['timestamp'].values
    stds = []
    for ts in events:
        window = df_gaze[(df_gaze['Time'] >= ts - 1500) & (df_gaze['Time'] < ts)]
        if not window.empty and not window['LeftGazeX'].dropna().empty:
            stds.append(window['LeftGazeX'].std())
    return np.nanmean(stds) if stds else np.nan

def align_gaze_to_screen(df_gaze, screen_width=1920, screen_height=1080):
    gaze_x = df_gaze['LeftGazeX']
    gaze_y = df_gaze['LeftGazeY']
    valid_x, valid_y = gaze_x.dropna(), gaze_y.dropna()
    if len(valid_x) < 2 or len(valid_y) < 2: return gaze_x, gaze_y
    gx_min, gx_max = valid_x.min(), valid_x.max()
    gy_min, gy_max = valid_y.min(), valid_y.max()
    
    if valid_x.quantile(0.95) <= 5.0 and valid_y.quantile(0.95) <= 5.0:
        gaze_x_aligned, gaze_y_aligned = gaze_x * screen_width, gaze_y * screen_height
    else:
        gaze_x_aligned = (gaze_x - gx_min) / (gx_max - gx_min) * screen_width if gx_max > gx_min else gaze_x * 0
        gaze_y_aligned = (gaze_y - gy_min) / (gy_max - gy_min) * screen_height if gy_max > gy_min else gaze_y * 0
    return gaze_x_aligned, gaze_y_aligned

def calculate_mouse_ratio(df_subset):
    """迂回比率（実移動距離 / 直線距離）を計算"""
    df_moves = df_subset[df_subset['tag'] == 'mouse_move'].dropna(subset=['x_num', 'y_num'])
    if len(df_moves) < 2: return np.nan, 0
    dx, dy = df_moves['x_num'].diff(), df_moves['y_num'].diff()
    actual_dist = np.nansum(np.sqrt(dx**2 + dy**2))
    start_pt = (df_moves['x_num'].iloc[0], df_moves['y_num'].iloc[0])
    end_pt = (df_moves['x_num'].iloc[-1], df_moves['y_num'].iloc[-1])
    straight_dist = math.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
    if straight_dist == 0: return (1.0 if actual_dist == 0 else np.nan), actual_dist
    return actual_dist / straight_dist, actual_dist

def analyze_mouse_hesitation(df_log):
    """全体およびサブタスク（Open→Add）のマウス迂回比率を分析"""
    df_log['x_num'] = pd.to_numeric(df_log['x'], errors='coerce')
    df_log['y_num'] = pd.to_numeric(df_log['y'], errors='coerce')
    
    start_row = df_log[df_log['tag'] == 'start']
    order_row = df_log[df_log['tag'] == 'order']
    if start_row.empty: return 0, 1.0, 1.0
    
    start_ts = start_row['timestamp'].values[0]
    order_ts = order_row['timestamp'].values[0] if not order_row.empty else df_log['timestamp'].max()
    
    total_mask = (df_log['timestamp'] >= start_ts) & (df_log['timestamp'] <= order_ts)
    total_ratio, total_dist = calculate_mouse_ratio(df_log[total_mask])
    
    subtask_ratios = []
    open_ts = None
    df_filtered = df_log[total_mask].sort_values('timestamp')
    for _, row in df_filtered.iterrows():
        if row['tag'] == 'open_modal': open_ts = row['timestamp']
        elif row['tag'] == 'add' and open_ts is not None:
            sub_mask = (df_log['timestamp'] >= open_ts) & (df_log['timestamp'] <= row['timestamp'])
            ratio, _ = calculate_mouse_ratio(df_log[sub_mask])
            subtask_ratios.append(ratio)
            open_ts = None
            
    avg_subtask_ratio = np.nanmean(subtask_ratios) if subtask_ratios else 1.0
    return total_dist, total_ratio, avg_subtask_ratio

def calculate_task_durations(df_log):
    start_row = df_log[df_log['tag'] == 'start']
    order_row = df_log[df_log['tag'] == 'order']
    if start_row.empty: return 0, []
    start_ts = start_row['timestamp'].values[0]
    order_ts = order_row['timestamp'].values[0] if not order_row.empty else df_log['timestamp'].max()
    total_time = (order_ts - start_ts) / 1000.0
    subtask_durations = []
    open_ts = None
    df_filtered = df_log[(df_log['timestamp'] >= start_ts) & (df_log['timestamp'] <= order_ts)].sort_values('timestamp')
    for _, row in df_filtered.iterrows():
        if row['tag'] == 'open_modal': open_ts = row['timestamp']
        elif row['tag'] == 'add' and open_ts is not None:
            subtask_durations.append((row['timestamp'] - open_ts) / 1000.0)
            open_ts = None
    return total_time, subtask_durations

def calculate_phase_gaze_stability(df_log, df_gaze, phase_tag, duration_ms=1000):
    events = df_log[df_log['tag'] == phase_tag]['timestamp'].values
    gaze_x_px, gaze_y_px = align_gaze_to_screen(df_gaze, SCREEN_WIDTH, SCREEN_HEIGHT)
    df_gaze['px_x'], df_gaze['px_y'] = gaze_x_px, gaze_y_px
    
    stds_x, stds_y = [], []
    for ts in events:
        window = df_gaze[(df_gaze['Time'] >= ts) & (df_gaze['Time'] <= ts + duration_ms)]
        vx, vy = window['px_x'].dropna(), window['px_y'].dropna()
        if len(vx) > 5:
            stds_x.append(vx.std())
            stds_y.append(vy.std())
    return np.nanmean(stds_x) if stds_x else 0, np.nanmean(stds_y) if stds_y else 0

# ==========================================
# Tkinter GUI
# ==========================================
class EyeTrackingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("視線・操作ログ 統合アナライザー")
        self.root.geometry("1450x850")
        
        self.current_dir = "./data/" if os.path.exists("./data/") else os.getcwd()
        self.paired_data = []
        self.loaded_dfs = []
        self.sliders = []
        
        self.setup_ui()
        if os.path.exists(self.current_dir): self.scan_directory(self.current_dir)

    def setup_ui(self):
        left_frame = tk.Frame(self.root, width=380, bg="#f5f5f5", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)
        
        tk.Button(left_frame, text="📂 フォルダを選択", command=self.select_directory, font=("", 12, "bold")).pack(fill=tk.X, pady=(0, 10))
        self.lbl_dir = tk.Label(left_frame, text=f"現在: {self.current_dir}", bg="#f5f5f5", anchor="w", justify="left", wraplength=350)
        self.lbl_dir.pack(fill=tk.X, pady=(0, 10))
        tk.Label(left_frame, text="自動ペアリング結果 (60秒以内):", bg="#f5f5f5", font=("", 10, "bold")).pack(anchor="w")
        
        columns = ("type", "ui", "file", "info")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", selectmode="extended", height=18)
        self.tree.heading("type", text="状態")
        self.tree.heading("ui", text="UI")
        self.tree.heading("file", text="ファイル名")
        self.tree.heading("info", text="時間差")
        self.tree.column("type", width=40, anchor="center")
        self.tree.column("ui", width=60, anchor="center")
        self.tree.column("file", width=180)
        self.tree.column("info", width=70, anchor="center")
        self.tree.tag_configure('error', background='#ffcccc')
        self.tree.tag_configure('ok', background='#e6ffe6')
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        tk.Button(left_frame, text="📊 選択したデータを比較・描画", command=self.analyze_selected, font=("", 12, "bold"), bg="#4CAF50", fg="white").pack(fill=tk.X, pady=10)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.tab_control = ttk.Notebook(self.right_frame)
        self.tabs = [ttk.Frame(self.tab_control) for _ in range(8)]
        tab_names = ["①瞳孔(平均)", "②瞳孔(分布)", "③タイムライン", "④2D全体軌跡", "⑤動的リプレイ", "⑥所要時間", "⑦局面別視線", "⑧マウス迷い"]
        for i, name in enumerate(tab_names): self.tab_control.add(self.tabs[i], text=f" {name} ")
        self.tab_control.pack(expand=1, fill="both")

    def select_directory(self):
        d = filedialog.askdirectory(initialdir=self.current_dir)
        if d:
            self.current_dir = d
            self.lbl_dir.config(text=f"現在: {self.current_dir}")
            self.scan_directory(d)

    def scan_directory(self, directory):
        for item in self.tree.get_children(): self.tree.delete(item)
        pairs, unpaired = auto_pair_files(directory)
        self.paired_data = pairs
        for idx, (log_f, gaze_f, diff, ui_type) in enumerate(pairs):
            self.tree.insert("", tk.END, iid=f"pair_{idx}", values=("OK", ui_type, log_f, f"{diff:.0f}秒"), tags=('ok',))
        for type_f, f_name, reason, ui_type in unpaired:
            self.tree.insert("", tk.END, values=("ERR", ui_type, f_name, reason), tags=('error',))

    def analyze_selected(self):
        valid_selections = [iid for iid in self.tree.selection() if iid.startswith("pair_")]
        if not valid_selections: return messagebox.showwarning("選択エラー", "緑色のペアを選択してください。")
            
        for tab in self.tabs:
            for widget in tab.winfo_children(): widget.destroy()
        
        self.loaded_dfs, self.sliders = [], []
        for i, iid in enumerate(valid_selections):
            idx = int(iid.split("_")[1])
            log_f, gaze_f, _, ui_type = self.paired_data[idx]
            try:
                df_log = pd.read_csv(os.path.join(self.current_dir, log_f))
                df_gaze = pd.read_csv(os.path.join(self.current_dir, gaze_f))
                self.loaded_dfs.append((df_log, df_gaze, f"Data {i+1} ({ui_type})", COLORS[i % len(COLORS)], ui_type))
            except Exception as e: messagebox.showerror("エラー", f"ロード失敗: {log_f}\n{e}")
                
        if self.loaded_dfs:
            methods = [self.draw_tab1, self.draw_tab2, self.draw_tab3, self.draw_tab4, self.draw_tab5, self.draw_tab6, self.draw_tab7, self.draw_tab8]
            for m in methods: m()

    # --- タブ描画関数 ---
    def draw_tab1(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        for df_log, df_gaze, label, color, _ in self.loaded_dfs:
            trajectories = extract_all_pupil_trajectories(df_log, df_gaze, 'add')
            if trajectories: ax.plot(pd.concat(trajectories, axis=1).mean(axis=1).index, pd.concat(trajectories, axis=1).mean(axis=1).values, label=f"{label}", marker='o', linewidth=2.5, color=color)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.6, label='add(0ms)')
        ax.set_title("【add】瞳孔径の推移 (平均値)", fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        FigureCanvasTkAgg(fig, master=self.tabs[0]).get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab2(self):
        fig, axes = plt.subplots(1, len(self.loaded_dfs), figsize=(6 * len(self.loaded_dfs), 5), sharey=True)
        if len(self.loaded_dfs) == 1: axes = [axes]
        for i, (df_log, df_gaze, label, color, _) in enumerate(self.loaded_dfs):
            trajectories = extract_all_pupil_trajectories(df_log, df_gaze, 'add')
            if trajectories:
                for traj in trajectories: axes[i].plot(traj.index, traj.values, color=color, alpha=0.15)
                axes[i].plot(pd.concat(trajectories, axis=1).mean(axis=1).index, pd.concat(trajectories, axis=1).mean(axis=1).values, color='black', marker='o', linewidth=3)
            axes[i].set_title(label, fontweight='bold'); axes[i].grid(True, alpha=0.3)
        plt.tight_layout(); FigureCanvasTkAgg(fig, master=self.tabs[1]).get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab3(self):
        fig, axes = plt.subplots(len(self.loaded_dfs), 1, figsize=(10, 2.5 * len(self.loaded_dfs)), sharex=True)
        if len(self.loaded_dfs) == 1: axes = [axes]
        for i, (df_log, df_gaze, label, color, _) in enumerate(self.loaded_dfs):
            start = df_log[df_log['tag'] == 'start']
            if start.empty: continue
            sts = start['timestamp'].values[0]
            ots = df_log[df_log['tag'] == 'order']['timestamp'].values[0] if not df_log[df_log['tag'] == 'order'].empty else df_log['timestamp'].max()
            d_gaze = df_gaze[(df_gaze['Time'] >= sts) & (df_gaze['Time'] <= ots)]
            is_nan = d_gaze['LeftPupil'].isna()
            axes[i].plot((d_gaze['Time'] - sts) / 1000.0, d_gaze['LeftPupil'], color=color, alpha=0.6)
            axes[i].set_title(f"{label} | 瞬き: {(is_nan & ~is_nan.shift(1, fill_value=False)).sum()}回", fontweight='bold')
            for _, r in df_log[(df_log['timestamp'] >= sts) & (df_log['timestamp'] <= ots)].iterrows():
                if r['tag'] in ['add', 'open_modal']: axes[i].axvline((r['timestamp'] - sts)/1000, color='blue' if r['tag']=='add' else 'purple', alpha=0.5)
            axes[i].grid(True, alpha=0.2)
        plt.tight_layout(); FigureCanvasTkAgg(fig, master=self.tabs[2]).get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab4(self):
        fig, axes = plt.subplots(1, len(self.loaded_dfs), figsize=(6 * len(self.loaded_dfs), 6))
        if len(self.loaded_dfs) == 1: axes = [axes]
        for i, (df_log, df_gaze, label, color, _) in enumerate(self.loaded_dfs):
            df_log['x_num'], df_log['y_num'] = pd.to_numeric(df_log['x'], errors='coerce'), pd.to_numeric(df_log['y'], errors='coerce')
            m = df_log[df_log['tag'] == 'mouse_move'].dropna(subset=['x_num', 'y_num'])
            axes[i].plot(m['x_num'], m['y_num'], color=color, alpha=0.3, linewidth=1.5)
            gx, gy = align_gaze_to_screen(df_gaze, SCREEN_WIDTH, SCREEN_HEIGHT)
            axes[i].plot(gx, gy, color=color, linestyle=':', alpha=0.15)
            axes[i].set_xlim(0, SCREEN_WIDTH); axes[i].set_ylim(SCREEN_HEIGHT, 0)
            axes[i].set_title(label, fontweight='bold'); axes[i].grid(True, alpha=0.2)
        plt.tight_layout(); FigureCanvasTkAgg(fig, master=self.tabs[3]).get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab5(self):
        num_plots = len(self.loaded_dfs)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        fig.subplots_adjust(bottom=0.25)
        if num_plots == 1: axes = [axes]
        
        self.plot_elements = [] 
        max_duration = 0 
        self.processed_data = []
        
        for i, (df_log, df_gaze, label, color, _) in enumerate(self.loaded_dfs):
            start_row = df_log[df_log['tag'] == 'start']
            if start_row.empty: continue
            
            start_ts = start_row['timestamp'].values[0]
            order_ts = df_log[df_log['tag'] == 'order']['timestamp'].values[0] if not df_log[df_log['tag'] == 'order'].empty else df_log['timestamp'].max()
            duration = (order_ts - start_ts) / 1000.0
            max_duration = max(max_duration, duration)
            
            log_work = df_log[(df_log['timestamp'] >= start_ts) & (df_log['timestamp'] <= order_ts)].copy()
            log_work['time_sec'] = (log_work['timestamp'] - start_ts) / 1000.0
            log_work['x_num'] = pd.to_numeric(log_work['x'], errors='coerce')
            log_work['y_num'] = pd.to_numeric(log_work['y'], errors='coerce')
            
            gaze_work = df_gaze[(df_gaze['Time'] >= start_ts) & (df_gaze['Time'] <= order_ts)].copy()
            gaze_work['time_sec'] = (gaze_work['Time'] - start_ts) / 1000.0
            
            gx_sync, gy_sync = align_gaze_to_screen(gaze_work, SCREEN_WIDTH, SCREEN_HEIGHT)
            gaze_work['gaze_x_px'] = gx_sync
            gaze_work['gaze_y_px'] = gy_sync
            
            axes[i].set_xlim(0, SCREEN_WIDTH)
            axes[i].set_ylim(SCREEN_HEIGHT, 0)
            axes[i].set_title(f"{label} (動的リプレイ)", fontsize=11, fontweight='bold')
            axes[i].grid(True, alpha=0.2)
            
            self.processed_data.append({
                'ax': axes[i], 'log': log_work, 'gaze': gaze_work, 'color': color
            })
            self.plot_elements.append({'lines': [], 'scatters': []})

        if max_duration == 0: return

        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        range_slider = RangeSlider(slider_ax, "表示時間(秒)", 0, max_duration, valinit=(0, min(10.0, max_duration)), color='#1f77b4')

    def draw_tab5(self):
        num_plots = len(self.loaded_dfs)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        fig.subplots_adjust(bottom=0.25)
        if num_plots == 1: axes = [axes]
        
        self.plot_elements = [] 
        max_duration = 0 
        self.processed_data = []
        
        for i, (df_log, df_gaze, label, color, _) in enumerate(self.loaded_dfs):
            start_row = df_log[df_log['tag'] == 'start']
            if start_row.empty: continue
            
            start_ts = start_row['timestamp'].values[0]
            order_ts = df_log[df_log['tag'] == 'order']['timestamp'].values[0] if not df_log[df_log['tag'] == 'order'].empty else df_log['timestamp'].max()
            duration = (order_ts - start_ts) / 1000.0
            max_duration = max(max_duration, duration)
            
            log_work = df_log[(df_log['timestamp'] >= start_ts) & (df_log['timestamp'] <= order_ts)].copy()
            log_work['time_sec'] = (log_work['timestamp'] - start_ts) / 1000.0
            log_work['x_num'] = pd.to_numeric(log_work['x'], errors='coerce')
            log_work['y_num'] = pd.to_numeric(log_work['y'], errors='coerce')
            
            gaze_work = df_gaze[(df_gaze['Time'] >= start_ts) & (df_gaze['Time'] <= order_ts)].copy()
            gaze_work['time_sec'] = (gaze_work['Time'] - start_ts) / 1000.0
            
            gx_sync, gy_sync = align_gaze_to_screen(gaze_work, SCREEN_WIDTH, SCREEN_HEIGHT)
            gaze_work['gaze_x_px'] = gx_sync
            gaze_work['gaze_y_px'] = gy_sync
            
            axes[i].set_xlim(0, SCREEN_WIDTH)
            axes[i].set_ylim(SCREEN_HEIGHT, 0)
            axes[i].set_title(f"{label} (動的リプレイ)", fontsize=11, fontweight='bold')
            axes[i].grid(True, alpha=0.2)
            
            self.processed_data.append({
                'ax': axes[i], 'log': log_work, 'gaze': gaze_work, 'color': color
            })
            self.plot_elements.append({'lines': [], 'scatters': []})

        if max_duration == 0: return

        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        range_slider = RangeSlider(slider_ax, "表示時間(秒)", 0, max_duration, valinit=(0, min(10.0, max_duration)), color='#1f77b4')

        def update_plot(val):
            t_min, t_max = val
            for i, data in enumerate(self.processed_data):
                ax = data['ax']
                log_df = data['log']
                gaze_df = data['gaze']
                c = data['color']
                elements = self.plot_elements[i]
                
                # 描画要素のクリア
                for l in elements['lines']: l.remove()
                for s in elements['scatters']: s.remove()
                elements['lines'].clear()
                elements['scatters'].clear()
                
                # マウス軌跡の描画
                mask_log = (log_df['time_sec'] >= t_min) & (log_df['time_sec'] <= t_max)
                cur_log = log_df[mask_log]
                cur_mouse = cur_log[cur_log['tag'] == 'mouse_move'].dropna(subset=['x_num', 'y_num'])
                
                if not cur_mouse.empty:
                    l_mouse, = ax.plot(cur_mouse['x_num'], cur_mouse['y_num'], color=c, alpha=0.8, linewidth=2, label='Mouse')
                    elements['lines'].append(l_mouse)

                # 視線軌跡の描画
                mask_gaze = (gaze_df['time_sec'] >= t_min) & (gaze_df['time_sec'] <= t_max)
                cur_gaze = gaze_df[mask_gaze].dropna(subset=['gaze_x_px', 'gaze_y_px'])
                
                if not cur_gaze.empty:
                    l_gaze, = ax.plot(cur_gaze['gaze_x_px'], cur_gaze['gaze_y_px'], color=c, linestyle=':', alpha=0.4, linewidth=1.5, label='Gaze')
                    elements['lines'].append(l_gaze)
                    
                # 【修正箇所】イベントマーカー（add, open_modal）の再描画
                # スライダーの範囲内（t_min ~ t_max）に発生したイベントだけをプロットする
                cur_add = cur_log[cur_log['tag'] == 'add'].dropna(subset=['x_num', 'y_num'])
                if not cur_add.empty:
                    sc = ax.scatter(cur_add['x_num'], cur_add['y_num'], color='red', marker='*', s=200, zorder=5)
                    elements['scatters'].append(sc)
                    
                cur_open = cur_log[cur_log['tag'] == 'open_modal'].dropna(subset=['x_num', 'y_num'])
                if not cur_open.empty:
                    sc = ax.scatter(cur_open['x_num'], cur_open['y_num'], color='purple', marker='o', s=80, zorder=4)
                    elements['scatters'].append(sc)
                
            fig.canvas.draw_idle()

        range_slider.on_changed(update_plot)
        self.sliders.append(range_slider)
        update_plot((0, min(10.0, max_duration)))

        canvas = FigureCanvasTkAgg(fig, master=self.tabs[4])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
    def draw_tab6(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for df_log, df_gaze, label, color, _ in self.loaded_dfs:
            tot, subs = calculate_task_durations(df_log)
            bar = ax1.bar(label, tot, color=color, alpha=0.8, width=0.4)
            ax1.annotate(f'{tot:.2f}s', xy=(bar[0].get_x() + bar[0].get_width()/2, tot), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
            ax2.plot(np.arange(1, len(subs)+1), subs, marker='o', lw=2.5, color=color, label=label)
        ax1.set_title("全体タスク総所要時間", fontweight='bold'); ax2.set_title("個別(Open→Add)所要時間", fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y'); ax2.grid(True, alpha=0.3); ax2.legend()
        plt.tight_layout(); FigureCanvasTkAgg(fig, master=self.tabs[5]).get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab7(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for df_log, df_gaze, label, color, _ in self.loaded_dfs:
            ox, _ = calculate_phase_gaze_stability(df_log, df_gaze, 'open_modal', 1000)
            cx, _ = calculate_phase_gaze_stability(df_log, df_gaze, 'close_modal', 1000)
            b1 = ax1.bar(label, ox, color=color, alpha=0.8, width=0.4)
            b2 = ax2.bar(label, cx, color=color, alpha=0.8, width=0.4)
            ax1.annotate(f'{ox:.1f}px', xy=(b1[0].get_x()+b1[0].get_width()/2, ox), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
            ax2.annotate(f'{cx:.1f}px', xy=(b2[0].get_x()+b2[0].get_width()/2, cx), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        ax1.set_title("Open直後1秒間の視線ブレ(X軸)", fontweight='bold'); ax2.set_title("Close直後1秒間の視線ブレ(X軸)", fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y'); ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); FigureCanvasTkAgg(fig, master=self.tabs[6]).get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab8(self):
        """新タブ⑧: マウス迂回比率（迷い）分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for df_log, df_gaze, label, color, _ in self.loaded_dfs:
            _, tot_ratio, sub_ratio = analyze_mouse_hesitation(df_log)
            
            # 全体比率
            b1 = ax1.bar(label, tot_ratio, color=color, alpha=0.8, width=0.4)
            ax1.annotate(f'{tot_ratio:.1f}倍', xy=(b1[0].get_x()+b1[0].get_width()/2, tot_ratio), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
            
            # 個別比率
            b2 = ax2.bar(label, sub_ratio, color=color, alpha=0.8, width=0.4)
            ax2.annotate(f'{sub_ratio:.1f}倍', xy=(b2[0].get_x()+b2[0].get_width()/2, sub_ratio), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

        ax1.set_title("全体の迂回比率\n※高いほどタスク間の次への準備（徘徊）が多い", fontweight='bold')
        ax2.set_title("個別操作(Open→Add)の平均迂回比率\n※高いほどターゲットを探して迷っている", fontweight='bold')
        ax1.set_ylabel("実移動距離 / 直線距離")
        ax1.grid(True, alpha=0.3, axis='y'); ax2.grid(True, alpha=0.3, axis='y')
        
        # コメント
        fig.text(0.5, 0.02, "【考察】viewは個別操作での迷いが最も少ない(1.9倍)ですが、全体では一番高く(15.5倍)なります。これは次への予測・待機行動がスムーズに行えている証拠です。", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        canvas = FigureCanvasTkAgg(fig, master=self.tabs[7])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackingDashboard(root)
    root.mainloop()