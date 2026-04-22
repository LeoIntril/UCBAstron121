"""
maint_start.py
--------------
Run this BEFORE starting an observing session.

Moves the Leuschner dish to the maintenance position (dish face-down) and
holds it there for a configurable duration to drain any trapped water, then
returns to stow so the main observing script can take over.

Usage:
    python3 maint_start.py              # default 3-minute drain
    python3 maint_start.py --wait 5     # 5-minute drain
    python3 maint_start.py --no-stow    # leave in maint position (unusual)
"""

import argparse
import time
from datetime import datetime, timezone

import ugradio
from ugradio import leusch

MAINT_WAIT_SEC_DEFAULT = 180   # 3 minutes


def run(wait_sec=MAINT_WAIT_SEC_DEFAULT, stow_after=True):
    print("=" * 55)
    print("  Leuschner Pre-Observation Maintenance")
    print("=" * 55)
    print(f"  UTC : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Drain hold : {wait_sec} s  ({wait_sec/60:.1f} min)")
    print(f"  Stow after : {stow_after}")
    print()

    input("  Press Enter to connect to the dish and begin maintenance... ")

    print("\n[maint] Connecting to Leuschner dish...")
    dish = leusch.LeuschTelescope()

    print("[maint] Moving to maintenance position (face-down)...")
    dish.maintenance()
    print(f"[maint] Holding for {wait_sec} s to drain trapped water...")

    # Countdown so the operator can see progress
    for remaining in range(wait_sec, 0, -10):
        print(f"[maint]   {remaining} s remaining...")
        time.sleep(min(10, remaining))

    print("[maint] Drain complete.")

    if stow_after:
        print("[maint] Moving to stow position...")
        dish.stow()
        print("[maint] Dish stowed. Ready to start observing.")
    else:
        print("[maint] Leaving dish in maintenance position as requested.")

    print("\n[maint] Pre-observation maintenance complete.")
    print("        You may now run:  python3 observe_hi_survey.py")


def main():
    parser = argparse.ArgumentParser(
        description="Leuschner pre-observation maintenance (water drain)"
    )
    parser.add_argument(
        "--wait", type=int, default=MAINT_WAIT_SEC_DEFAULT,
        metavar="SECONDS",
        help=f"Seconds to hold in maintenance position (default {MAINT_WAIT_SEC_DEFAULT})"
    )
    parser.add_argument(
        "--no-stow", action="store_false", dest="stow_after",
        help="Do not stow after draining (leave in maint position)"
    )
    args = parser.parse_args()
    run(wait_sec=args.wait, stow_after=args.stow_after)


if __name__ == "__main__":
    main()
