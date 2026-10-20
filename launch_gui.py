"""Launch the optional Tk desktop application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from housing_market_intelligence.gui import main


if __name__ == "__main__":
    main()
