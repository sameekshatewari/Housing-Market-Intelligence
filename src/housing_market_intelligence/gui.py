"""Small desktop interface for running the project without shell arguments."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .cli import execute


class HousingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Housing Market Intelligence")
        self.geometry("760x540")
        self.data_path = tk.StringVar(value=str(Path("data/housing.csv")))
        self.forecast_path = tk.StringVar(value=str(Path("data/future_economic_inputs.csv")))
        self.features = tk.StringVar(
            value="inflation_rate,mortgage_rate,unemployment_rate,fuel_price,interest_rate,synthetic_region_index"
        )
        self.messages: queue.Queue[str] = queue.Queue()
        self._build_form()
        self.after(100, self._read_messages)

    def _build_form(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Historical CSV").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.data_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse", command=self._choose_data).grid(row=0, column=2)
        ttk.Label(frame, text="Future inputs CSV (optional)").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=self.forecast_path).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse", command=self._choose_forecast).grid(row=1, column=2)
        ttk.Label(frame, text="Features (comma-separated)").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.features).grid(row=2, column=1, columnspan=2, sticky="ew", padx=8)
        ttk.Button(frame, text="Run analysis", command=self._run).grid(row=3, column=1, sticky="w", pady=12)
        self.log = tk.Text(frame, height=20, wrap="word", state="disabled")
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

    def _choose_data(self) -> None:
        chosen = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if chosen:
            self.data_path.set(chosen)

    def _choose_forecast(self) -> None:
        chosen = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if chosen:
            self.forecast_path.set(chosen)

    def _run(self) -> None:
        from argparse import Namespace

        root = Path(__file__).resolve().parents[2]
        data = Path(self.data_path.get())
        forecast = Path(self.forecast_path.get()) if self.forecast_path.get().strip() else None
        if not data.is_absolute():
            data = root / data
        if forecast is not None and not forecast.is_absolute():
            forecast = root / forecast
        args = Namespace(
            data=data,
            date_column="date",
            target_column="price",
            features=self.features.get(),
            macro_data=None,
            area_lookup=None,
            forecast_csv=forecast,
            output_dir=root / "outputs",
            train_fraction=0.8,
        )
        self._append("Running model comparison...\n")
        threading.Thread(target=self._worker, args=(args,), daemon=True).start()

    def _worker(self, args: object) -> None:
        try:
            result = execute(args)
            self.messages.put(f"Finished. Best model: {result['best_model_by_mae']}\n")
        except Exception as error:
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


if __name__ == "__main__":
    main()
