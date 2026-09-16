# Experiment plan: the Section 4.8.2 occlusion figure (F-1)

**Status:** ready to run. Written 2 September 2026.
**Execution:** local, on Enam's machine.
**No GPU. No training. This reads stored CSVs and draws a figure.** Estimated 30 to 60 minutes including the render check.
**Owner decision point:** one, in section 1.4. Do not resolve it yourself.

---

## 0. Read this first

Three things that will otherwise cost a round trip each.

1. **Do NOT use `g3_profiles_chandrop.csv` or `g3_profiles_noaug.csv`.** Those hold the *normalized* profiles, and Section 4.8.2 of the thesis states in print that the normalized measures are artifacts of a collapsed scale and are not interpretable. A figure built on them would contradict the section it illustrates. The figure must be drawn from the **raw** `drop_pp` values in the two `instr/occlusion.csv` files named in section 2. The handoff's Tier 3a note lists all four files; only the raw ones are usable.
2. **Scale for the rendered PDF page, not for the PNG.** Existing generators in `06_Code/figures_rework/` set an explicit `figsize` and 300 dpi and are checked by rendering the thesis, not by opening the PNG. A figure that looks right at full size is usually unreadable at page width.
3. **The Python interpreter is `06_Code/.venv/Scripts/python.exe`.** Every `jobs_*.txt` in `06_Code/` points at a `MSc Python Project/.venv` path that was removed in the August restructure. Those files are stale.

---

## 1. The task

### 1.1 Why it exists

Section 4.8.2 carries the strongest mechanism number in the thesis, a 5.6-fold reduction in single-electrode occlusion cost on 40 subjects out of 40, and it has **no table and no figure**. Every comparable finding in Chapter 4 has an exhibit. This is presentation debt, not a scientific gap.

### 1.2 What to build

**Figure 4.10**, two panels, in the conventions of `06_Code/figures_rework/`.

- **Panel (a): per-channel occlusion cost.** Nine channels on the x axis, mean macro-F1 lost to occluding that channel on the y axis in percentage points, two series: un-augmented residual network and channel-dropout residual network. Error bars are the standard deviation across the 40 subjects. This shows the shape: the un-augmented model leans hard on channel 5 and the augmented one leans on nothing.
- **Panel (b): per-subject total cost.** Forty paired points or lines, un-augmented on the left, channel dropout on the right, one connector per subject. This shows the consistency, which is the 40 out of 40, and it shows the handful of augmented subjects whose total is negative.

One caption, one figure, two panels. Do not split into two numbered figures.

### 1.3 Numbering and placement

- Chapter 4 figures currently end at **Figure 4.9**, so this is **Figure 4.10** and **no renumbering of anything is required.** Confirm that before you write the label.
- It belongs in **Section 4.8.2**, immediately after the paragraph beginning "The mechanism itself can be measured rather than argued", which is the paragraph that quotes 84.2 pp and 15.0 pp.
- House convention, verified in the document: **the table or image element comes first and its caption paragraph comes immediately after it.** Match that.

### 1.4 The decision point, for Enam and not for you

Panel (b) can be drawn as **paired connected lines** (clearer about 40 out of 40, busier) or as a **paired dot plot with the mean marked** (cleaner, less emphatic about the consistency). Build the connected-lines version first, render it, and if it looks crowded at page width, produce the dot-plot alternative as a second PNG and **report both to Enam rather than choosing.**

---

## 2. Inputs, which exist and must not be regenerated

| Artifact | Path | Use |
|---|---|---|
| Un-augmented occlusion, raw | `results_g3_noaug_instr/instr/occlusion.csv` | 360 rows, 40 subjects x 9 channels, column `drop_pp` |
| Channel-dropout occlusion, raw | `results_cd_resnet_nose_chandrop/instr/occlusion.csv` | same shape |
| Paired per-subject summary | `g3_occlusion_per_subject.csv` | cross-check only |
| Regenerator for the statistics | `g3_occlusion_stats.py` | do not re-run unless a gate fails |

**Both arms are the SE-free `resnet` backbone.** That is deliberate and matches the G1 arm. The caption must say so, because the deep model of record is the SE-carrying `resnet_se`, and a reader will otherwise assume the figure is of the model of record.

---

## 3. Gates. Check all four before drawing anything.

1. Each `occlusion.csv` has exactly **360 rows and 40 distinct subjects**.
2. Per-channel means on the un-augmented arm reproduce **[8.68, 7.16, 8.95, 8.07, 9.35, 19.83, 2.77, 10.95, 8.45]** pp, and on the channel-dropout arm **[1.82, 1.43, 1.43, 2.60, 1.27, 1.14, 0.17, 3.24, 1.94]** pp.
3. Summed per subject, the arm means are **84.21 pp** and **15.03 pp**, giving the 5.6x quoted in the thesis.
4. Per-subject totals range **38.71 to 153.23** un-augmented and **-13.37 to 38.63** with channel dropout. The negative minimum is real and panel (b) must not clip it.

**If any gate fails, stop and report.** Do not adjust the figure to match the thesis; the thesis matches the CSVs or something has moved underneath both.

---

## 4. Drawing conventions, copied from `figures_rework/make_fig3_1.py`

- `matplotlib.use('Agg')`, dpi 300, `font.family` sans-serif with `['Liberation Sans', 'DejaVu Sans', 'Arial']`.
- **Black and white only.** No colour anywhere. Distinguish the two series by fill: white with a black edge for un-augmented, solid black or a plain hatch for channel dropout. Square corners, `joinstyle='miter'`, linewidth about 1.3.
- Plain black arrows if any are needed. No shadows, no gradients, no seaborn styling, no grid unless it is a light horizontal one.
- Axis labels in full words with units, for example "macro-F1 lost to occlusion (pp)".
- Write the generator as `06_Code/figures_rework/make_fig4_10.py` with a module docstring naming the figure and the regeneration command, matching the two existing generators.

---

## 5. Wiring it into the thesis

1. Save the PNG beside the generator.
2. Insert the image and its caption into **`01_Thesis/MSc Thesis.docx` first**, then apply the identical change to **`01_Thesis/Results_Chapter.docx`**. The two are in exact sync and must stay so.
3. **Moving an image into a docx:** extract the blob and re-add with `tgt.part.get_or_add_image(BytesIO(blob))`, which assigns an unused part name. Relating the source part object directly produces duplicate zip entries and a subtly malformed file.
4. Add one sentence to the paragraph that quotes 84.2 and 15.0 pp so the figure is cited in the body. A caption with no in-text reference is a defect; the thesis was checked for this and currently has none.
5. Run `01_Thesis/_tools/enforce_chapter_page_breaks.py`. It must report 9 divisions, all starting on a fresh page.
6. Regenerate the render. **`MSc Thesis.pdf` is locked by a stale LibreOffice lock file, so write `MSc Thesis (2026-09-02).pdf`.** Launch `soffice` and wait in the same shell call; a backgrounded job dies with the parent shell.
7. Verify: figure captions run 4.1 to 4.10 with no gaps or duplicates, every in-text reference resolves to a caption, zero em dashes, no blank pages, and the chapter files still match the assembled document paragraph for paragraph.

**Back up first** to `_ARCHIVE/prebackups/<name>_prebackup_fig4_10.docx`.

---

## 6. House rules that apply to the caption text

- **No em dashes anywhere.** Not in the caption, not in the axis labels, not in chat replies.
- Decimals in captions, percentages in prose: write 84.2 pp and 15.0 pp as percentage points, and macro-F1 values to three decimals.
- Do not be over-defensive. State what the figure shows; do not pre-emptively list what it does not show.
- The caption must name the backbone (SE-free `resnet`), the dataset and protocol (SIAT, LOSO, n = 40), and what the error bars are.

---

## 7. What to report back

1. Whether all four gates passed, with the numbers.
2. The two panel-(b) variants if the first looked crowded, for Enam's choice.
3. The new page count and confirmation of no blank pages.
4. Confirmation that figure numbering is 4.1 to 4.10 with no renumbering having been needed.
