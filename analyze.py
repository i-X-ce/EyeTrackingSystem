import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# 初期設定
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

COLORS = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728'] # 緑, オレンジ, 青, 赤

# ==========================================
# データ処理関数
# ==========================================
def extract_time_from_filename(filename):
    """ファイル名から14桁の時刻(YYYYMMDDHHMMSS)を抽出"""
    match = re.search(r'\d{14}', filename)
    if match:
        return datetime.strptime(match.group(), "%Y%m%d%H%M%S")
    return None

def auto_pair_files(directory):
    """ディレクトリ内のファイルを60秒以内のペアに分類する"""
    files = os.listdir(directory)
    gaze_files = [f for f in files if 'gaze' in f.lower() and f.endswith('.csv')]
    log_files = [f for f in files if 'log' in f.lower() and f.endswith('.csv')]
    
    pairs = []
    unpaired = []
    
    # ログファイルを基準にペアを探す
    for log_f in log_files:
        log_time = extract_time_from_filename(log_f)
        if not log_time:
            unpaired.append(('log', log_f, "時刻抽出エラー"))
            continue
            
        best_match = None
        min_diff = float('inf')
        
        for gaze_f in gaze_files:
            gaze_time = extract_time_from_filename(gaze_f)
            if gaze_time:
                diff = abs((log_time - gaze_time).total_seconds())
                if diff <= 60 and diff < min_diff:
                    min_diff = diff
                    best_match = gaze_f
                    
        if best_match:
            pairs.append((log_f, best_match, min_diff))
            gaze_files.remove(best_match) # マッチしたものはリストから消す
        else:
            unpaired.append(('log', log_f, "ペアの視線データなし"))
            
    # 残った視線データはアンペア
    for gaze_f in gaze_files:
        unpaired.append(('gaze', gaze_f, "ペアのログデータなし"))
        
    return sorted(pairs), unpaired

def extract_pupil_trajectory(df_log, df_gaze, tag='add'):
    """カート追加(add)イベント前後の瞳孔変化を抽出"""
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
            
    if trajectories:
        return pd.concat(trajectories, axis=1).mean(axis=1)
    return pd.Series(dtype=float)

# ==========================================
# Tkinter アプリケーション
# ==========================================
class EyeTrackingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("視線・操作ログ 統合アナライザー (GUIペアリング対応版)")
        self.root.geometry("1400x850")
        
        self.current_dir = "./data/" if os.path.exists("./data/") else os.getcwd()
        self.paired_data = [] # (log_path, gaze_path)
        self.loaded_dfs = []  # [(df_log, df_gaze, label, color), ...]
        
        self.setup_ui()
        if os.path.exists(self.current_dir):
            self.scan_directory(self.current_dir)

    def setup_ui(self):
        # --- 左側パネル (操作・ファイル選択) ---
        left_frame = tk.Frame(self.root, width=350, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)
        
        btn_select_dir = tk.Button(left_frame, text="📂 フォルダを選択", command=self.select_directory, font=("", 12, "bold"))
        btn_select_dir.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_dir = tk.Label(left_frame, text=f"現在: {self.current_dir}", bg="#f0f0f0", anchor="w", justify="left", wraplength=330)
        self.lbl_dir.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(left_frame, text="自動ペアリング結果 (60秒以内):", bg="#f0f0f0", font=("", 10, "bold")).pack(anchor="w")
        
        # ツリービュー (ペア一覧)
        columns = ("type", "file", "info")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", selectmode="extended", height=15)
        self.tree.heading("type", text="状態")
        self.tree.heading("file", text="ファイル (Log基準)")
        self.tree.heading("info", text="詳細/誤差")
        self.tree.column("type", width=40, anchor="center")
        self.tree.column("file", width=180)
        self.tree.column("info", width=100)
        
        # エラー行の背景色設定
        self.tree.tag_configure('error', background='#ffcccc')
        self.tree.tag_configure('ok', background='#e6ffe6')
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # スクロールバー
        scrollbar = ttk.Scrollbar(self.tree, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        tk.Label(left_frame, text="※ 比較したいペアを複数選択(Ctrl+クリック)して\n下のボタンを押してください。", bg="#f0f0f0", justify="left").pack(anchor="w", pady=5)
        
        btn_analyze = tk.Button(left_frame, text="📊 選択したペアを比較・描画", command=self.analyze_selected, font=("", 12, "bold"), bg="#4CAF50", fg="white")
        btn_analyze.pack(fill=tk.X, pady=10)

        # --- 右側パネル (グラフ描画) ---
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.tab_control = ttk.Notebook(self.right_frame)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text="  ① 瞳孔径の推移 (add前後)  ")
        self.tab_control.add(self.tab2, text="  ② 一連の操作・瞬きタイムライン  ")
        self.tab_control.add(self.tab3, text="  ③ マウス軌跡 (2Dマップ)  ")
        self.tab_control.pack(expand=1, fill="both")

    def select_directory(self):
        directory = filedialog.askdirectory(initialdir=self.current_dir)
        if directory:
            self.current_dir = directory
            self.lbl_dir.config(text=f"現在: {self.current_dir}")
            self.scan_directory(directory)

    def scan_directory(self, directory):
        # ツリーをクリア
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        pairs, unpaired = auto_pair_files(directory)
        self.paired_data = pairs # (log, gaze, diff)
        
        # 正常なペアを追加
        for idx, (log_f, gaze_f, diff) in enumerate(pairs):
            iid = f"pair_{idx}"
            self.tree.insert("", tk.END, iid=iid, values=("OK", log_f, f"差 {diff:.0f}秒"), tags=('ok',))
            
        # エラーのファイルを追加
        for type_f, f_name, reason in unpaired:
            self.tree.insert("", tk.END, values=("ERR", f_name, reason), tags=('error',))

    def analyze_selected(self):
        selected_iids = self.tree.selection()
        valid_selections = [iid for iid in selected_iids if iid.startswith("pair_")]
        
        if not valid_selections:
            messagebox.showwarning("選択エラー", "緑色(OK)のペアを少なくとも1つ選択してください。")
            return
            
        # グラフ領域をクリア
        for widget in self.tab1.winfo_children(): widget.destroy()
        for widget in self.tab2.winfo_children(): widget.destroy()
        for widget in self.tab3.winfo_children(): widget.destroy()
        
        self.loaded_dfs = []
        
        # データのロード
        for i, iid in enumerate(valid_selections):
            idx = int(iid.split("_")[1])
            log_f, gaze_f, _ = self.paired_data[idx]
            
            try:
                df_log = pd.read_csv(os.path.join(self.current_dir, log_f))
                df_gaze = pd.read_csv(os.path.join(self.current_dir, gaze_f))
                # 汎用的なラベル（ファイル名の先頭1文字 + log）
                label = f"Data {i+1} ({log_f[:10]}...)"
                color = COLORS[i % len(COLORS)]
                self.loaded_dfs.append((df_log, df_gaze, label, color))
            except Exception as e:
                messagebox.showerror("読み込みエラー", f"{log_f} の読み込みに失敗しました。\n{e}")
                
        if self.loaded_dfs:
            self.draw_tab1()
            self.draw_tab2()
            self.draw_tab3()

    def draw_tab1(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        for df_log, df_gaze, label, color in self.loaded_dfs:
            traj = extract_pupil_trajectory(df_log, df_gaze, 'add')
            if not traj.empty:
                ax.plot(traj.index, traj.values, label=label, marker='o', linewidth=2.5, color=color)
                
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.6, label='カート追加(0ms)')
        ax.set_title("【add】クリック前後の瞳孔径の推移", fontsize=14, fontweight='bold')
        ax.set_xlabel("操作からの相対時間 (ミリ秒)")
        ax.set_ylabel("瞳孔径 (LeftPupil)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab1)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab2(self):
        num_plots = len(self.loaded_dfs)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]
            
        for i, (df_log, df_gaze, label, color) in enumerate(self.loaded_dfs):
            ax = axes[i]
            start_row = df_log[df_log['tag'] == 'start']
            order_row = df_log[df_log['tag'] == 'order']
            
            if start_row.empty or order_row.empty:
                ax.text(0.5, 0.5, "start または order ログが見つかりません", ha='center')
                continue
                
            start_ts = start_row['timestamp'].values[0]
            order_ts = order_row['timestamp'].values[0]
            
            df_gaze_filtered = df_gaze[(df_gaze['Time'] >= start_ts) & (df_gaze['Time'] <= order_ts)].copy()
            time_sec = (df_gaze_filtered['Time'] - start_ts) / 1000.0
            
            ax.plot(time_sec, df_gaze_filtered['LeftPupil'], color=color, alpha=0.6)
            
            # 瞬き（NaN）区間を赤背景
            nan_blocks = np.where(df_gaze_filtered['LeftPupil'].isna())[0]
            for idx in nan_blocks:
                ax.axvline(time_sec.iloc[idx], color='red', alpha=0.03, zorder=1)
                
            # イベントプロット
            df_events = df_log[(df_log['timestamp'] >= start_ts) & (df_log['timestamp'] <= order_ts)]
            for _, row in df_events.iterrows():
                event_time = (row['timestamp'] - start_ts) / 1000.0
                tag = row['tag']
                if tag == 'add':
                    ax.axvline(event_time, color='blue', linestyle='-', alpha=0.7)
                    ax.text(event_time, ax.get_ylim()[1]*0.9, 'Add', color='blue', fontsize=8, rotation=90)
                elif tag == 'open_modal':
                    ax.axvline(event_time, color='purple', linestyle=':', alpha=0.7)
                    ax.text(event_time, ax.get_ylim()[1]*0.9, 'Open', color='purple', fontsize=8, rotation=90)
                    
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
            
        axes[-1].set_xlabel("実験開始からの経過時間 (秒)")
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab2)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def draw_tab3(self):
        """マウス軌跡を2D空間にマッピング"""
        num_plots = len(self.loaded_dfs)
        # 横並びに配置
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        if num_plots == 1:
            axes = [axes]
            
        for i, (df_log, _, label, color) in enumerate(self.loaded_dfs):
            ax = axes[i]
            
            # X, Y を数値に変換（エラーは無視）
            df_log['x_num'] = pd.to_numeric(df_log['x'], errors='coerce')
            df_log['y_num'] = pd.to_numeric(df_log['y'], errors='coerce')
            
            # マウス移動の軌跡を線で描画
            df_mouse = df_log[df_log['tag'] == 'mouse_move'].dropna(subset=['x_num', 'y_num'])
            ax.plot(df_mouse['x_num'], df_mouse['y_num'], color=color, alpha=0.3, linewidth=1, label='マウス軌跡')
            
            # クリックイベント（add）を星マークで描画
            df_add = df_log[df_log['tag'] == 'add'].dropna(subset=['x_num', 'y_num'])
            ax.scatter(df_add['x_num'], df_add['y_num'], color='red', marker='*', s=150, zorder=5, label='カート追加(add)')
            
            # 画面仕様に合わせてY軸を反転 (上が0、下が最大)
            ax.invert_yaxis()
            
            ax.set_title(f"{label}：マウス軌跡", fontsize=11, fontweight='bold')
            ax.set_xlabel("X座標")
            ax.set_ylabel("Y座標")
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.2)
            
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab3)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackingDashboard(root)
    root.mainloop()