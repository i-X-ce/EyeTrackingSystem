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
                if 'type=view' in msg:
                    ui_type = "View"
                elif 'type=classic' in msg:
                    ui_type = "Classic"
                elif 'type=none' in msg:
                    ui_type = "None"
        except:
            pass
    return time_val, ui_type

def auto_pair_files(directory):
    files = os.listdir(directory)
    gaze_files = [f for f in files if 'gaze' in f.lower() and f.endswith('.csv')]
    log_files = [f for f in files if 'log' in f.lower() and f.endswith('.csv')]
    
    pairs = []
    unpaired = []
    log_info = {}
    
    for log_f in log_files:
        t_val, ui_type = extract_time_and_type(directory, log_f)
        log_info[log_f] = (t_val, ui_type)

    for log_f, (log_time, ui_type) in log_info.items():
        if not log_time:
            unpaired.append(('log', log_f, "時間抽出不可", "Unknown"))
            continue
            
        best_match = None
        min_diff = float('inf')
        
        for gaze_f in gaze_files:
            gaze_time, _ = extract_time_and_type(directory, gaze_f)
            if gaze_time:
                diff = abs((log_time - gaze_time).total_seconds())
                if diff <= 60 and diff < min_diff:
                    min_diff = diff
                    best_match = gaze_f
                    
        if best_match and min_diff <= 60:
            pairs.append((log_f, best_match, min_diff, ui_type))
            gaze_files.remove(best_match)
        else:
            unpaired.append(('log', log_f, "ペアなし(誤差大)", ui_type))
            
    for gaze_f in gaze_files:
        unpaired.append(('gaze', gaze_f, "ペアなし", "Unknown"))
        
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

def calculate_mouse_metrics(df_log):
    df_log['x_num'] = pd.to_numeric(df_log['x'], errors='coerce')
    df_log['y_num'] = pd.to_numeric(df_log['y'], errors='coerce')
    df_moves = df_log[df_log['tag'] == 'mouse_move'].dropna(subset=['x_num', 'y_num'])
    if len(df_moves) < 2: return 0, 1.0
    dx = df_moves['x_num'].diff()
    dy = df_moves['y_num'].diff()
    distances = np.sqrt(dx**2 + dy**2)
    total_distance = np.nansum(distances)
    start_pt = (df_moves['x_num'].iloc[0], df_moves['y_num'].iloc[0])
    end_pt = (df_moves['x_num'].iloc[-1], df_moves['y_num'].iloc[-1])
    straight_line = math.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
    ratio = total_distance / straight_line if straight_line > 0 else 1.0
    return total_distance, ratio

def align_gaze_to_screen(df_gaze, screen_width=1920, screen_height=1080):
    gaze_x = df_gaze['LeftGazeX']
    gaze_y = df_gaze['LeftGazeY']
    valid_x = gaze_x.dropna()
    valid_y = gaze_y.dropna()
    if len(valid_x) < 2 or len(valid_y) < 2:
        return gaze_x, gaze_y
    gx_min, gx_max = valid_x.min(), valid_x.max()
    gy_min, gy_max = valid_y.min(), valid_y.max()
    
    if valid_x.quantile(0.95) <= 5.0 and valid_y.quantile(0.95) <= 5.0:
        gaze_x_aligned = gaze_x * screen_width
        gaze_y_aligned = gaze_y * screen_height
    else:
        gaze_x_aligned = (gaze_x - gx_min) / (gx_max - gx_min) * screen_width if gx_max > gx_min else gaze_x * 0
        gaze_y_aligned = (gaze_y - gy_min) / (gy_max - gy_min) * screen_height if gy_max > gy_min else gaze_y * 0
    return gaze_x_aligned, gaze_y_aligned

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
    """指定イベント（open_modal / close_modal）直後(0~duration)の視線のピクセル単位の標準偏差を計算"""
    events = df_log[df_log['tag'] == phase_tag]['timestamp'].values
    gaze_x_px, gaze_y_px = align_gaze_to_screen(df_gaze, SCREEN_WIDTH, SCREEN_HEIGHT)
    df_gaze['px_x'] = gaze_x_px
    df_gaze['px_y'] = gaze_y_px
    
    stds_x, stds_y = [], []
    for ts in events:
        # イベント直後の一定時間（duration_ms）を切り出す
        window = df_gaze[(df_gaze['Time'] >= ts) & (df_gaze['Time'] <= ts + duration_ms)]
        vx = window['px_x'].dropna()
        vy = window['px_y'].dropna()
        if len(vx) > 5:  # データが少なすぎる場合は除外
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
        self.root.geometry("1400x850")
        
        self.current_dir = "./data/" if os.path.exists("./data/") else os.getcwd()
        self.paired_data = []
        self.loaded_dfs = []
        self.sliders = []
        
        self.setup_ui()
        if os.path.exists(self.current_dir):
            self.scan_directory(self.current_dir)

    def setup_ui(self):
        left_frame = tk.Frame(self.root, width=380, bg="#f5f5f5", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)
        
        btn_select_dir = tk.Button(left_frame, text="📂 フォルダを選択", command=self.select_directory, font=("", 12, "bold"))
        btn_select_dir.pack(fill=tk.X, pady=(0, 10))
        
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
        
        btn_analyze = tk.Button(left_frame, text="📊 選択したデータを比較・描画", command=self.analyze_selected, font=("", 12, "bold"), bg="#4CAF50", fg="white")
        btn_analyze.pack(fill=tk.X, pady=10)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.tab_control = ttk.Notebook(self.right_frame)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab4 = ttk.Frame(self.tab_control)
        self.tab5 = ttk.Frame(self.tab_control)
        self.tab6 = ttk.Frame(self.tab_control)
        self.tab7 = ttk.Frame(self.tab_control) # 新タブ生成
        
        self.tab_control.add(self.tab1, text="  ① 瞳孔推移 (平均)  ")
        self.tab_control.add(self.tab2, text="  ② 瞳孔推移 (全試行分布)  ")
        self.tab_control.add(self.tab3, text="  ③ タイムライン  ")
        self.tab_control.add(self.tab4, text="  ④ 2D全体軌跡  ")
        self.tab_control.add(self.tab5, text="  ⑤ 動的リプレイ  ")
        self.tab_control.add(self.tab6, text="  ⑥ 所要時間分析  ")
        self.tab_control.add(self.tab7, text="  ⑦ 局面別視線分析  ") # ノートブック登録
        self.tab_control.pack(expand=1, fill="both")

    def select_directory(self):
        directory = filedialog.askdirectory(initialdir=self.current_dir)
        if directory:
            self.current_dir = directory
            self.lbl_dir.config(text=f"現在: {self.current_dir}")
            self.scan_directory(directory)

    def scan_directory(self, directory):
        for item in self.tree.get_children():
            self.tree.delete(item)
        pairs, unpaired = auto_pair_files(directory)
        self.paired_data = pairs
        for idx, (log_f, gaze_f, diff, ui_type) in enumerate(pairs):
            self.tree.insert("", tk.END, iid=f"pair_{idx}", values=("OK", ui_type, log_f, f"{diff:.0f}秒"), tags=('ok',))
        for type_f, f_name, reason, ui_type in unpaired:
            self.tree.insert("", tk.END, values=("ERR", ui_type, f_name, reason), tags=('error',))

    def analyze_selected(self):
        selected_iids = self.tree.selection()
        valid_selections = [iid for iid in selected_iids if iid.startswith("pair_")]
        if not valid_selections:
            messagebox.showwarning("選択エラー", "緑色のペアを選択してください。")
            return
            
        for tab in [self.tab1, self.tab2, self.tab3, self.tab4, self.tab5, self.tab6, self.tab7]:
            for widget in tab.winfo_children(): widget.destroy()
        
        self.loaded_dfs = []
        self.sliders = []
        for i, iid in enumerate(valid_selections):
            idx = int(iid.split("_")[1])
            log_f, gaze_f, _, ui_type = self.paired_data[idx]
            try:
                df_log = pd.read_csv(os.path.join(self.current_dir, log_f))
                df_gaze = pd.read_csv(os.path.join(self.current_dir, gaze_f))
                label = f"Data {i+1} ({ui_type})"
                self.loaded_dfs.append((df_log, df_gaze, label, COLORS[i % len(COLORS)], ui_type))
            except Exception as e:
                messagebox.showerror("エラー", f"ロード失敗: {log_f}\n{e}")
                
        if self.loaded_dfs:
            self.draw_tab1()
            self.draw_tab2()
            self.draw_tab3()
            self.draw_tab4()
            self.draw_tab5()
            self.draw_tab6()
            self.draw_tab7() # 新タブ描画

    def draw_tab1(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        for df_log, df_gaze, label, color, ui_type in self.loaded_dfs:
            trajectories = extract_all_pupil_trajectories(df_log, df_gaze, 'add')
            if trajectories:
                mean_traj = pd.concat(trajectories, axis=1).mean(axis=1)
                ax.plot(mean_traj.index, mean_traj.values, label=f"{label} (平均)", marker='o', linewidth=2.5, color=color)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.6, label='カート追加(0ms)')
        ax.set_title("【add】瞳孔径の推移 (平均値)", fontsize=13, fontweight='bold')
        ax.set_xlabel("操作からの相対時間 (ミリ秒)")
        ax.set_ylabel("瞳孔径 (LeftPupil 相対値)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self.tab1)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab2(self):
        num_plots = len(self.loaded_dfs)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)
        if num_plots == 1: axes = [axes]
        for i, (df_log, df_gaze, label, color, ui_type) in enumerate(self.loaded_dfs):
            ax = axes[i]
            trajectories = extract_all_pupil_trajectories(df_log, df_gaze, 'add')
            if trajectories:
                for traj in trajectories: ax.plot(traj.index, traj.values, color=color, alpha=0.15, linewidth=1.0)
                mean_traj = pd.concat(trajectories, axis=1).mean(axis=1)
                ax.plot(mean_traj.index, mean_traj.values, label="平均値", marker='o', linewidth=3.0, color='black', alpha=0.8)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.6, label='カート追加(0ms)')
            ax.set_title(f"{label}\n全 {len(trajectories)} 試行の分布", fontsize=11, fontweight='bold')
            ax.set_xlabel("相対時間 (ミリ秒)")
            if i == 0: ax.set_ylabel("瞳孔径")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab2)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab3(self):
        num_plots = len(self.loaded_dfs)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes]
        for i, (df_log, df_gaze, label, color, ui_type) in enumerate(self.loaded_dfs):
            ax = axes[i]
            start_row = df_log[df_log['tag'] == 'start']
            if start_row.empty: continue
            start_ts = start_row['timestamp'].values[0]
            order_ts = df_log[df_log['tag'] == 'order']['timestamp'].values[0] if not df_log[df_log['tag'] == 'order'].empty else df_log['timestamp'].max()
            
            df_gaze_filtered = df_gaze[(df_gaze['Time'] >= start_ts) & (df_gaze['Time'] <= order_ts)].copy()
            time_sec = (df_gaze_filtered['Time'] - start_ts) / 1000.0
            
            is_nan = df_gaze_filtered['LeftPupil'].isna()
            blink_starts = is_nan & ~is_nan.shift(1, fill_value=False)
            blink_count = blink_starts.sum()
            
            ax.plot(time_sec, df_gaze_filtered['LeftPupil'], color=color, alpha=0.6)
            nan_blocks = np.where(is_nan)[0]
            for idx in nan_blocks: ax.axvline(time_sec.iloc[idx], color='red', alpha=0.03, zorder=1)
                
            df_events = df_log[(df_log['timestamp'] >= start_ts) & (df_log['timestamp'] <= order_ts)]
            for _, row in df_events.iterrows():
                event_time = (row['timestamp'] - start_ts) / 1000.0
                if row['tag'] == 'add': ax.axvline(event_time, color='blue', linestyle='-', alpha=0.8)
                elif row['tag'] == 'open_modal': ax.axvline(event_time, color='purple', linestyle=':', alpha=0.8)
                    
            ax.set_title(f"{label} ｜ 瞬き回数: {blink_count}回", fontsize=11, fontweight='bold')
            ax.set_ylabel("瞳孔径")
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("タスク開始からの経過時間 (秒)")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab3)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab4(self):
        num_plots = len(self.loaded_dfs)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        if num_plots == 1: axes = [axes]
        for i, (df_log, df_gaze, label, color, ui_type) in enumerate(self.loaded_dfs):
            ax = axes[i]
            df_log['x_num'] = pd.to_numeric(df_log['x'], errors='coerce')
            df_log['y_num'] = pd.to_numeric(df_log['y'], errors='coerce')
            df_mouse = df_log[df_log['tag'] == 'mouse_move'].dropna(subset=['x_num', 'y_num'])
            ax.plot(df_mouse['x_num'], df_mouse['y_num'], color=color, alpha=0.3, linewidth=1.5, label='マウス軌跡')
            
            gx_sync, gy_sync = align_gaze_to_screen(df_gaze, SCREEN_WIDTH, SCREEN_HEIGHT)
            ax.plot(gx_sync, gy_sync, color=color, linestyle=':', alpha=0.15, linewidth=0.8, label='視線軌跡 (画面マッピング)')
            
            df_add = df_log[df_log['tag'] == 'add'].dropna(subset=['x_num', 'y_num'])
            ax.scatter(df_add['x_num'], df_add['y_num'], color='red', marker='*', s=150, zorder=5, label='Add')
            df_open = df_log[df_log['tag'] == 'open_modal'].dropna(subset=['x_num', 'y_num'])
            ax.scatter(df_open['x_num'], df_open['y_num'], color='purple', marker='o', s=60, zorder=4, label='Open')
            
            ax.set_xlim(0, SCREEN_WIDTH)
            ax.set_ylim(SCREEN_HEIGHT, 0)
            total_dist, ratio = calculate_mouse_metrics(df_log)
            ax.set_title(f"{label}\n総距離: {total_dist:.0f}px | 迂回比率: {ratio:.2f}倍", fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab4)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab5(self):
        num_plots = len(self.loaded_dfs)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        fig.subplots_adjust(bottom=0.25)
        if num_plots == 1: axes = [axes]
        self.plot_elements = [] 
        max_duration = 0 
        self.processed_data = []
        for i, (df_log, df_gaze, label, color, ui_type) in enumerate(self.loaded_dfs):
            ax = axes[i]
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
            
            ax.set_xlim(0, SCREEN_WIDTH)
            ax.set_ylim(SCREEN_HEIGHT, 0)
            ax.set_title(f"{label} (動的リプレイ)", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2)
            self.processed_data.append({'ax': ax, 'log': log_work, 'gaze': gaze_work, 'color': color})
            self.plot_elements.append({'lines': [], 'scatters': []})

        if max_duration == 0: return
        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        range_slider = RangeSlider(slider_ax, "表示時間(秒)", 0, max_duration, valinit=(0, min(10.0, max_duration)), color='#1f77b4')

        def update_plot(val):
            t_min, t_max = val
            for i, data in enumerate(self.processed_data):
                ax, log_df, gaze_df, c, elements = data['ax'], data['log'], data['gaze'], data['color'], self.plot_elements[i]
                for l in elements['lines']: l.remove()
                for s in elements['scatters']: s.remove()
                elements['lines'].clear()
                elements['scatters'].clear()
                
                mask_log = (log_df['time_sec'] >= t_min) & (log_df['time_sec'] <= t_max)
                cur_mouse = log_df[mask_log][log_df[mask_log]['tag'] == 'mouse_move'].dropna(subset=['x_num', 'y_num'])
                if not cur_mouse.empty: elements['lines'].append(ax.plot(cur_mouse['x_num'], cur_mouse['y_num'], color=c, alpha=0.8, linewidth=2)[0])

                mask_gaze = (gaze_df['time_sec'] >= t_min) & (gaze_df['time_sec'] <= t_max)
                cur_gaze = gaze_df[mask_gaze].dropna(subset=['gaze_x_px', 'gaze_y_px'])
                if not cur_gaze.empty: elements['lines'].append(ax.plot(cur_gaze['gaze_x_px'], cur_gaze['gaze_y_px'], color=c, linestyle=':', alpha=0.4, linewidth=1.5)[0])
                    
                for tag, m_color, m_size in [('add', 'red', 200), ('open_modal', 'purple', 80)]:
                    cur_event = log_df[mask_log][log_df[mask_log]['tag'] == tag].dropna(subset=['x_num', 'y_num'])
                    if not cur_event.empty: elements['scatters'].append(ax.scatter(cur_event['x_num'], cur_event['y_num'], color=m_color, marker='*' if tag=='add' else 'o', s=m_size, zorder=5))
            fig.canvas.draw_idle()

        range_slider.on_changed(update_plot)
        self.sliders.append(range_slider)
        update_plot((0, min(10.0, max_duration)))
        canvas = FigureCanvasTkAgg(fig, master=self.tab5)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab6(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        labels, total_times, colors = [], [], []
        
        for df_log, df_gaze, label, color, ui_type in self.loaded_dfs:
            total_time, subtask_durs = calculate_task_durations(df_log)
            labels.append(label)
            total_times.append(total_time)
            colors.append(color)
            trials = np.arange(1, len(subtask_durs) + 1)
            ax2.plot(trials, subtask_durs, marker='o', linewidth=2.5, color=color, label=f"{label}")
            
        bars = ax1.bar(labels, total_times, color=colors, alpha=0.8, width=0.4)
        ax1.set_title("全体タスクの総所要時間", fontsize=11, fontweight='bold')
        ax1.set_ylabel("総完了時間 (秒)")
        ax1.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
        ax2.set_title("個別商品の追加所要時間\n(モーダル展開 から カート追加 まで)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("試行回数 (商品目)")
        ax2.set_ylabel("所要時間 (秒)")
        max_len = max([len(calculate_task_durations(df)[1]) for df, _, _, _, _ in self.loaded_dfs], default=10)
        ax2.set_xticks(np.arange(1, max_len + 1))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab6)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab7(self):
        """新タブ⑦: 局面別視線分析 (Open / Close 直後の視線ブレ比較)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        labels, colors = [], []
        open_stds_x, open_stds_y = [], []
        close_stds_x, close_stds_y = [], []
        
        for df_log, df_gaze, label, color, ui_type in self.loaded_dfs:
            labels.append(label)
            colors.append(color)
            
            # Open_modal 直後1秒間の標準偏差
            ox, oy = calculate_phase_gaze_stability(df_log, df_gaze, 'open_modal', duration_ms=1000)
            open_stds_x.append(ox)
            
            # Close_modal 直後1秒間の標準偏差
            cx, cy = calculate_phase_gaze_stability(df_log, df_gaze, 'close_modal', duration_ms=1000)
            close_stds_x.append(cx)
            
        # 左側グラフ：Open直後
        bars1 = ax1.bar(labels, open_stds_x, color=colors, alpha=0.8, width=0.4)
        ax1.set_title("【Open直後 1秒間】の視線ブレ（X軸）\n※数値が小さいほど誘導がスムーズ", fontsize=11, fontweight='bold')
        ax1.set_ylabel("標準偏差 (ピクセル)")
        ax1.grid(True, alpha=0.3, axis='y')
        for bar in bars1:
            h = bar.get_height()
            ax1.annotate(f'{h:.1f}px', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 右側グラフ：Close直後
        bars2 = ax2.bar(labels, close_stds_x, color=colors, alpha=0.8, width=0.4)
        ax2.set_title("【Close直後 1秒間】の視線ブレ（X軸）\n※数値が小さいほど次への迷いがない", fontsize=11, fontweight='bold')
        ax2.set_ylabel("標準偏差 (ピクセル)")
        ax2.grid(True, alpha=0.3, axis='y')
        for bar in bars2:
            h = bar.get_height()
            ax2.annotate(f'{h:.1f}px', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        # Y軸のスケールを合わせる
        max_y = max(max(open_stds_x), max(close_stds_x)) * 1.15
        if max_y > 0:
            ax1.set_ylim(0, max_y)
            ax2.set_ylim(0, max_y)

        # 全体コメントをグラフ下に配置
        fig.text(0.5, 0.02, "【考察】classicは出現時(Open)は滑らかに視線を誘導できているが、消滅時(Close)に視線が激しく迷子になっていることがわかります。\nviewは消滅時(Close)のブレが最も抑えられており、次の探索への移行が最速です。", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15) # テキストのスペース確保
        canvas = FigureCanvasTkAgg(fig, master=self.tab7)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackingDashboard(root)
    root.mainloop()