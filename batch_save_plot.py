from pathlib import Path
from plot_emg_gait import load_emg_and_labels, plot_emg_with_gait

SUBJECTS = range(1, 41)
MOVEMENTS = ["WAK", "UPS", "DNS", "STDUP"]

TMIN, TMAX = 0.0, 2.2
OUT_BASE = (Path(__file__).resolve().parent / "plots" / "raw")   # will create Sub01/, Sub02/, ...

def main():
    failures = []

    for subj in SUBJECTS:
        for mov in MOVEMENTS:   
            try:
                print(f"Processing Sub{subj:02d} {mov} ...")
                data_df, label_df = load_emg_and_labels(subj, mov)

                plot_emg_with_gait(
                    data_df,
                    label_df,
                    subject=subj,
                    movement=mov,
                    time_window=None,
                    save=True,
                    dpi=300,
                    show=False,
                    out_dir=OUT_BASE,
                )

            except Exception as e:
                failures.append((subj, mov, str(e)))
                print(f"[FAILED] Sub{subj:02d} {mov}: {e}")

    if failures:
        print("\n==== Failures ====")
        for subj, mov, msg in failures:
            print(f"Sub{subj:02d} {mov} -> {msg}")
    else:
        print("\nAll plots saved successfully!")

if __name__ == "__main__":
    main()
