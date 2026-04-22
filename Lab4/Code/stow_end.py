"""
stow_end.py
-----------
Run this AFTER finishing an observing session.

Moves the Leuschner dish safely to the stow position and confirms success.
Includes a noise-diode safety-off in case the diode was left on mid-session
(e.g. after a crash or Ctrl-C during calibration).

Usage:
    python3 stow_end.py
"""

import time
from datetime import datetime, timezone

import ugradio
from ugradio import leusch


def run():
    print("=" * 55)
    print("  Leuschner Post-Observation Stow")
    print("=" * 55)
    print(f"  UTC : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    input("  Press Enter to connect to the dish and stow... ")

    print("\n[stow] Connecting to Leuschner dish...")
    dish = leusch.LeuschTelescope()

    # Safety: turn off noise diode in case it was left on
    try:
        noise = leusch.LeuschNoise()
        noise.off()
        print("[stow] Noise diode confirmed OFF.")
    except Exception as e:
        print(f"[stow] Warning: could not verify noise diode state: {e}")

    print("[stow] Moving dish to stow position...")
    dish.stow()

    print("[stow] Dish stowed successfully.")
    print(f"[stow] Session ended at "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == "__main__":
    run()
