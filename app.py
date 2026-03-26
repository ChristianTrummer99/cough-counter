"""
Cough Counter — main application.

Run with: python3 app.py
"""

import os
import sys
import queue
import sqlite3
import datetime
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from audio_classifier import CoughDetector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "yamnet.tflite")
DB_PATH = os.path.join(_DIR, "sessions.db")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "audio_classifier/yamnet/float32/1/yamnet.tflite"
)

# ---------------------------------------------------------------------------
# Colours & fonts
# ---------------------------------------------------------------------------

BG = "#f5f5f7"
HEADER_BG = "#1d1d1f"
HEADER_FG = "#f5f5f7"
GREEN = "#34c759"
RED = "#ff3b30"
BLUE = "#007aff"
SUBTLE = "#8e8e93"
BORDER = "#d1d1d6"
FLASH_BG = "#d1f0db"
FONT = "Helvetica Neue"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time       TEXT NOT NULL,
                end_time         TEXT,
                cough_count      INTEGER DEFAULT 0,
                duration_seconds INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cough_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        self.conn.commit()

    def start_session(self, start_time: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO sessions (start_time, cough_count) VALUES (?, 0)",
            (start_time,),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_cough_count(self, session_id: int, count: int):
        self.conn.execute(
            "UPDATE sessions SET cough_count = ? WHERE id = ?",
            (count, session_id),
        )
        self.conn.commit()

    def end_session(self, session_id: int, end_time: str, cough_count: int, duration: int):
        self.conn.execute(
            "UPDATE sessions SET end_time=?, cough_count=?, duration_seconds=? WHERE id=?",
            (end_time, cough_count, duration, session_id),
        )
        self.conn.commit()

    def record_cough_event(self, session_id: int, timestamp: str):
        self.conn.execute(
            "INSERT INTO cough_events (session_id, timestamp) VALUES (?, ?)",
            (session_id, timestamp),
        )
        self.conn.commit()

    def load_sessions(self, limit: int = 100) -> list:
        cursor = self.conn.execute(
            "SELECT id, start_time, end_time, cough_count, duration_seconds "
            "FROM sessions WHERE end_time IS NOT NULL "
            "ORDER BY start_time DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()

    def load_cough_events(self, session_id: int) -> list:
        cursor = self.conn.execute(
            "SELECT timestamp FROM cough_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class CoughCounterApp:
    def __init__(self, root: tk.Tk, db: Database, detector: CoughDetector):
        self.root = root
        self.db = db
        self.detector = detector

        self.result_queue: queue.Queue = queue.Queue()
        self.detector.result_queue = self.result_queue

        # Session state
        self._session_active = False
        self._session_id: int | None = None
        self._session_start: datetime.datetime | None = None
        self._cough_count = 0

        self._build_ui()
        self._refresh_history()
        self._poll_queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("Cough Counter")
        self.root.geometry("580x680")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self._build_header()
        self._build_status()
        self._build_counter()
        self._build_controls()
        self._build_separator()
        self._build_history()

    def _build_header(self):
        header = tk.Frame(self.root, bg=HEADER_BG, height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Cough Counter",
            bg=HEADER_BG, fg=HEADER_FG,
            font=(FONT, 20, "bold"),
        ).pack(expand=True)

    def _build_status(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(pady=(18, 0))

        self._status_dot = tk.Label(
            frame, text="●", font=(FONT, 13),
            bg=BG, fg=SUBTLE,
        )
        self._status_dot.pack(side=tk.LEFT, padx=(0, 6))

        self._status_label = tk.Label(
            frame, text="Not recording",
            font=(FONT, 13), bg=BG, fg=SUBTLE,
        )
        self._status_label.pack(side=tk.LEFT)

    def _build_counter(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(pady=(8, 0))

        self._count_label = tk.Label(
            frame, text="0",
            font=(FONT, 88, "bold"),
            bg=BG, fg=HEADER_BG,
            width=4, anchor=tk.CENTER,
        )
        self._count_label.pack()

        tk.Label(
            frame, text="coughs this session",
            font=(FONT, 13), bg=BG, fg=SUBTLE,
        ).pack()

        self._timer_label = tk.Label(
            self.root, text="",
            font=(FONT, 12), bg=BG, fg=SUBTLE,
        )
        self._timer_label.pack(pady=(6, 0))

    def _build_controls(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(pady=22)

        self._toggle_btn = tk.Button(
            frame,
            text="Start Session",
            command=self._toggle_session,
            bg=GREEN, fg="white",
            font=(FONT, 14, "bold"),
            relief=tk.FLAT,
            activebackground=GREEN,
            activeforeground="white",
            cursor="hand2",
            padx=36, pady=12,
            bd=0,
        )
        self._toggle_btn.pack()

    def _build_separator(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill=tk.X, padx=28, pady=(4, 0))
        tk.Frame(outer, bg=BORDER, height=1).pack(fill=tk.X)

    def _build_history(self):
        header_row = tk.Frame(self.root, bg=BG)
        header_row.pack(fill=tk.X, padx=28, pady=(14, 8))

        tk.Label(
            header_row, text="Session History",
            font=(FONT, 15, "bold"), bg=BG, fg=HEADER_BG,
        ).pack(side=tk.LEFT)

        table_frame = tk.Frame(self.root, bg=BG)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=28, pady=(0, 20))

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=BG,
            foreground=HEADER_BG,
            rowheight=28,
            fieldbackground=BG,
            font=(FONT, 12),
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#e5e5ea",
            foreground=HEADER_BG,
            font=(FONT, 12, "bold"),
            relief=tk.FLAT,
        )
        style.map("Treeview", background=[("selected", "#cce4ff")])

        cols = ("date", "start", "duration", "coughs", "rate")
        self._tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            height=9,
        )

        headers = {
            "date": ("Date", 90),
            "start": ("Started", 80),
            "duration": ("Duration", 85),
            "coughs": ("Coughs", 65),
            "rate": ("/hr", 60),
        }
        for col, (label, width) in headers.items():
            self._tree.heading(col, text=label)
            self._tree.column(col, width=width, anchor=tk.CENTER)

        sb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<Double-1>", self._on_session_double_click)

    # ------------------------------------------------------------------
    # Session detail chart
    # ------------------------------------------------------------------

    def _on_session_double_click(self, event):
        selection = self._tree.selection()
        if not selection:
            return
        iid = selection[0]
        session_id = getattr(self, "_session_id_map", {}).get(iid)
        if session_id is None:
            return
        self._show_session_chart(session_id)

    def _show_session_chart(self, session_id: int):
        events = self.db.load_cough_events(session_id)
        if not events:
            messagebox.showinfo("No data", "No cough events recorded for this session.")
            return

        timestamps = [datetime.datetime.fromisoformat(ts) for ts in events]
        session_start = timestamps[0]
        session_end = timestamps[-1]
        span = (session_end - session_start).total_seconds()

        bin_seconds = 15 * 60
        bin_label = "15-minute"

        origin = session_start.replace(second=0, microsecond=0)
        bins: dict[datetime.datetime, int] = {}
        for ts in timestamps:
            offset = (ts - origin).total_seconds()
            idx = int(offset // bin_seconds)
            bin_start = origin + datetime.timedelta(seconds=idx * bin_seconds)
            bins[bin_start] = bins.get(bin_start, 0) + 1

        all_bin_starts = []
        t = origin
        end_bound = origin + datetime.timedelta(
            seconds=((session_end - origin).total_seconds() // bin_seconds + 1) * bin_seconds
        )
        while t <= end_bound:
            all_bin_starts.append(t)
            t += datetime.timedelta(seconds=bin_seconds)

        counts = [bins.get(b, 0) for b in all_bin_starts]

        win = tk.Toplevel(self.root)
        win.title(f"Session Detail — {session_start.strftime('%b %-d, %I:%M %p')}")
        win.geometry("700x420")
        win.configure(bg="white")

        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")

        bar_width = bin_seconds / 86400 * 0.8
        ax.bar(
            all_bin_starts, counts,
            width=bar_width, color=BLUE, edgecolor="white", linewidth=0.5,
            zorder=3,
        )

        ax.set_ylabel("Coughs", fontsize=11)
        ax.set_xlabel(f"Time ({bin_label} intervals)", fontsize=11)
        ax.set_title(
            f"{len(timestamps)} coughs — {session_start.strftime('%b %-d, %Y')}",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I:%M %p"))
        fig.autofmt_xdate(rotation=30)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        photo = tk.PhotoImage(data=buf.getvalue())
        label = tk.Label(win, image=photo, bg="white")
        label.image = photo
        label.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _toggle_session(self):
        if self._session_active:
            self._stop_session()
        else:
            self._start_session()

    def _start_session(self):
        self._cough_count = 0
        self._session_start = datetime.datetime.now()
        self._session_active = True
        self._session_id = self.db.start_session(self._session_start.isoformat())

        self._count_label.config(text="0", bg=BG)
        self._toggle_btn.config(text="Stop Session", bg=RED, activebackground=RED)
        self._status_dot.config(fg=GREEN)
        self._status_label.config(text="Recording…", fg=GREEN)

        try:
            self.detector.start()
        except Exception as exc:
            messagebox.showerror("Microphone error", str(exc))
            self._stop_session()
            return

        self._tick_timer()

    def _stop_session(self):
        self._session_active = False
        self.detector.stop()

        end_time = datetime.datetime.now()
        duration = int((end_time - self._session_start).total_seconds())
        self.db.end_session(
            self._session_id, end_time.isoformat(), self._cough_count, duration
        )

        self._toggle_btn.config(text="Start Session", bg=GREEN, activebackground=GREEN)
        self._status_dot.config(fg=SUBTLE)
        self._status_label.config(text="Not recording", fg=SUBTLE)
        self._timer_label.config(text="")
        self._refresh_history()

    # ------------------------------------------------------------------
    # Event loop / queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                item = self.result_queue.get_nowait()
                if item.get("event") == "cough" and self._session_active:
                    self._on_cough()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_cough(self):
        self._cough_count += 1
        self._count_label.config(text=str(self._cough_count), bg=FLASH_BG)
        self.db.update_cough_count(self._session_id, self._cough_count)
        self.db.record_cough_event(
            self._session_id, datetime.datetime.now().isoformat()
        )
        self.root.after(350, lambda: self._count_label.config(bg=BG))

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _tick_timer(self):
        if not self._session_active:
            return
        elapsed = int((datetime.datetime.now() - self._session_start).total_seconds())
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        ts = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        started = self._session_start.strftime("%I:%M %p").lstrip("0")
        self._timer_label.config(text=f"Started {started} · {ts}")
        self.root.after(1000, self._tick_timer)

    # ------------------------------------------------------------------
    # History table
    # ------------------------------------------------------------------

    def _refresh_history(self):
        self._session_id_map: dict[str, int] = {}

        for row in self._tree.get_children():
            self._tree.delete(row)

        for sid, start_iso, end_iso, coughs, duration_secs in self.db.load_sessions():
            dt = datetime.datetime.fromisoformat(start_iso)
            date_str = dt.strftime("%b %-d")
            start_str = dt.strftime("%I:%M %p").lstrip("0")

            secs = duration_secs or 0
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            if h:
                dur_str = f"{h}h {m}m"
            elif m:
                dur_str = f"{m}m {s}s"
            else:
                dur_str = f"{s}s"

            rate = f"{coughs / (secs / 3600):.1f}" if secs >= 60 else "—"

            iid = self._tree.insert(
                "", tk.END, values=(date_str, start_str, dur_str, coughs, rate)
            )
            self._session_id_map[iid] = sid

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        if self._session_active:
            self._stop_session()
        self.detector.close()
        self.db.close()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading YAMNet model (~3 MB) to {MODEL_PATH} …", flush=True)

    def reporthook(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / total_size)) if total_size > 0 else 0
        print(f"\r  {pct}% ", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=reporthook)
    print("\nModel ready.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        ensure_model()
    except Exception as exc:
        print(f"Error downloading model: {exc}", file=sys.stderr)
        print(f"Please download manually:\n  {MODEL_URL}\nand save it to:\n  {MODEL_PATH}")
        sys.exit(1)

    db = Database(DB_PATH)
    result_queue: queue.Queue = queue.Queue()
    detector = CoughDetector(model_path=MODEL_PATH, result_queue=result_queue)

    print("Loading audio classifier…", flush=True)
    try:
        detector.build()
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Ready.")

    root = tk.Tk()
    CoughCounterApp(root=root, db=db, detector=detector)
    root.mainloop()


if __name__ == "__main__":
    main()
