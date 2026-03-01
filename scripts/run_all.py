"""
Run All Phases: Full Pipeline
==============================
Executes the entire analysis pipeline from raw data to final figures.
"""

import sys
import os
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    start = time.time()

    print("=" * 70)
    print("EYEMW: Mind Wandering is Not Monolithic")
    print("Full Analysis Pipeline")
    print("=" * 70)

    # Phase 1
    print("\n\n>>> PHASE 1: Data Preparation <<<\n")
    from scripts.phase1_data_preparation import main as phase1
    phase1()

    # Phase 2
    print("\n\n>>> PHASE 2: Latent Profile Analysis <<<\n")
    from scripts.phase2_lpa import main as phase2
    phase2()

    # Phase 3
    print("\n\n>>> PHASE 3: Gaze Signature Discrimination <<<\n")
    from scripts.phase3_gaze_discrimination import main as phase3
    phase3()

    # Phase 4
    print("\n\n>>> PHASE 4: Environmental Robustness <<<\n")
    from scripts.phase4_slicing_analysis import main as phase4
    phase4()

    # Phase 5
    print("\n\n>>> PHASE 5: Final Figures <<<\n")
    from scripts.phase5_figures import main as phase5
    phase5()

    elapsed = time.time() - start
    print(f"\n\nTotal pipeline time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
