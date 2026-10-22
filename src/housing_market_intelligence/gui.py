"""Small desktop interface for running the Zillow forecasting workflow."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from argparse import Namespace
from pathlib import Path
from tkinter import filedialog, ttk

from .cli import execute


class HousingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Housing Market Intelligence")
        self.geometry("760x500")
        self.data_path = tk.StringVar(value=str(Path("data/zhvi_metro.csv")))
        self.messages: queue.Queue[str] = queue.Queue()
        self._build_form()
        self.after(100, self._read_messages)

    def _build_form(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Zillow Metro ZHVI CSV").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(frame, textvariable=self.data_path).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(frame, text="Browse", command=self._choose_data).grid(
            row=0, column=2
        )
        ttk.Button(frame, text="Run forecast comparison", command=self._run).grid(
            row=1, column=1, sticky="w", pady=12
        )
        self.log = tk.Text(frame, height=18, wrap="word", state="disabled")
        self.log.grid(row=2, column=0, columnspan=3, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)

    def _choose_data(self) -> None:
        chosen = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if chosen:
            self.data_path.set(chosen)

    def _run(self) -> None:
        root = Path(__file__).resolve().parents[2]
        data = Path(self.data_path.get())
        if not data.is_absolute():
            data = root / data
        args = Namespace(
            data=data,
            output_dir=root / "outputs",
            train_fraction=0.7,
            validation_fraction=0.15,
        )
        self._append("Running chronological forecast comparison...\n")
        threading.Thread(target=self._worker, args=(args,), daemon=True).start()

    def _worker(self, args: Namespace) -> None:
        try:
            result = execute(args)
            self.messages.put(
                "Finished. Selected model: "
                f"{result['best_model_by_validation_mae']}\n"
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            self.messages.put(f"Analysis failed: {error}\n")

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _read_messages(self) -> None:
        while not self.messages.empty():
            self._append(self.messages.get_nowait())
        self.after(100, self._read_messages)


def main() -> None:
    app = HousingApp()
    app.mainloop()
