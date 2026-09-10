# Stage P-0: literature grounding for channel dropout

No GPU. Retrieval and verification pass run 3 September 2026. **Nothing here has
been written into the thesis.** Section 2.4.4 is voice-audited; the drafted
sentences below are for Enam to review and place, not to paste.

## 1. The four starting sources: retrieval status and what each supports

### 1.1 Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR 15, 1929-1958.

**Retrieved in full** (JMLR open-access PDF, 30 pages). Section 10 "Multiplicative
Gaussian Noise" and Table 10 both read and verified verbatim:

- Table 10, classification error %: MNIST (2 layers, 1024 units) Bernoulli
  dropout 1.08 plus or minus 0.04 against Gaussian dropout 0.95 plus or minus
  0.04; CIFAR-10 (3 conv + 2 FC) Bernoulli 12.6 plus or minus 0.1 against
  Gaussian 12.5 plus or minus 0.1. Averaged over 10 seeds. sigma set to
  sqrt((1 - p) / p) per layer.
- Section 10, verbatim: "multiplying by a random variable drawn from N(1, 1)
  works just as well, or perhaps better than using Bernoulli noise." The paper
  shows both forms can be set to the same mean and variance, that r_g has the
  highest entropy and r_b the lowest given those moments, and that "both these
  extremes work well, although preliminary experimental results ... suggest that
  the high entropy case might work slightly better."

The plan's characterization is exact. This is W-4's result one level up:
mean-and-variance-matched multiplicative noise matches Bernoulli masking at the
unit level in 2014; W-4 finds the same relationship at the input-channel level.

### 1.2 Neverova, Wolf, Taylor, Nebout (2016). ModDrop: adaptive multi-modal gesture recognition. IEEE TPAMI 38(8), 1692-1706.

**Retrieved: abstract and bibliographic record in full** (IEEE Xplore / dblp /
arXiv 1501.00102 preprint). Citation details confirmed: TPAMI vol. 38 no. 8
pp. 1692-1706, 2016, DOI 10.1109/TPAMI.2015.2461544. The mechanism as described
by the plan is confirmed by the abstract: "gradual fusion involving random
dropping of separate channels (dubbed ModDrop) for learning cross-modality
correlations while preserving uniqueness of each modality-specific
representation." Full text not pulled; cite only for the idea of training-time
structured dropping of an entire input stream, which the abstract supports.

### 1.3 Pereira, Chalatsis, Hodossy, Farina (2024). Tackling electrode shift in gesture recognition with HD-EMG electrode subsets. arXiv:2401.02773.

**Retrieved: arXiv abstract in full; confirmed published.** This is **not a
preprint in the end**: it was published at **ICASSP 2024** (IEEE International
Conference on Acoustics, Speech and Signal Processing), IEEE Xplore document
10448329. The IEEE copyright notice on the retrieved copy is therefore correct
and it should be cited as the ICASSP 2024 paper, not as arXiv-only. The plan's
instruction to flag it as a preprint no longer applies.

Abstract confirms the method: "training on a collection of input channel subsets
and augmenting our training distribution with data from different electrode
locations, simultaneously targeting electrode shift and reducing input
dimensionality ... significantly higher intersession performance across subjects
and classification algorithms." The abstract does **not** state the +6.9 %
figure, the "16 times fewer input channels" figure, the subject count, or the
classifier names; those are in the full text, which was not retrieved. **Cite it
only for what the abstract states** (subset-collection training as augmentation
for electrode shift, improved intersession performance) unless the full text is
obtained. It is the direct sEMG precedent for the intervention and the P-5
comparator; Farina is already in the reference list.

### 1.4 Zhang, Zhou, Fang, Gao, Ju (2026). Assessing the resilience of sEMG classifiers to sensor malfunction and signal saturation. Sensors 26(8), 2386.

**Retrieved: abstract and full bibliographic record via PubMed (PMID 42076494,
PMCID PMC13120049, DOI 10.3390/s26082386, published 13 April 2026); MDPI full
text returned HTTP 403 and was not retrieved.** Confirmed from the abstract /
PubMed record: nine subjects, multi-session multi-gesture protocol, 250 ms
window, four time-domain features (RMS, Variance, Zero Crossing, Waveform
Length), classical feature-classifier pipelines, degradations applied at test
time (amplitude saturation and single-sensor failure). It describes itself as
"the first systematic robustness map of a conventional sEMG pipeline under
controlled clipping and single-sensor failure." No training-time augmentation.

**Terminology caution for Enam:** this paper uses the phrase "channel dropout"
for a **test-time** degradation (a sensor going dead), which is the opposite end
from the thesis's training-time augmentation of the same name. If it is cited in
2.4.4 next to the thesis's channel dropout, one sentence should disambiguate, or
an examiner will read a contradiction that is not there.

Cite only for what the title and abstract state: that single-channel-loss
profiling is an accepted way to characterize an sEMG pipeline, which is what
makes section 4.8.2's occlusion protocol non-idiosyncratic. Scope does not
overlap section 4.8.2 (classical vs deep, test-time vs 40-subject LOSO,
no augmentation vs a 5.6-fold cost collapse).

## 2. Further sources found in the same space

Training-time channel or electrode dropping in sEMG / EEG, and occlusion or
channel-ablation profiling on deep sEMG models. Open access, mostly within five
years. None retrieved in full; listed for Enam to pull and judge.

1. **MAST: Masked Autoencoder with Swin Transformer for mitigating electrode
   shift in HD-EMG gesture recognition (arXiv:2410.17261, 2024).** Self-
   supervised pre-training on masked subsets of HD-sEMG channels with four
   masking strategies, one of them "sensor-wise random masking." A direct
   training-time channel-masking precedent in sEMG, framed for electrode shift.
   Belongs in 2.4.4 and 5.7 as a recent instance of the same idea.
2. **"From Unstable Contacts to Stable Control: A Deep Learning Paradigm for
   HD-sEMG in Neurorobotics" (arXiv:2309.11086, 2023).** Deep HD-sEMG under
   unstable electrode contact. Context for 2.4.4.
3. **Attention-DFCNN (Frontiers in Neuroscience, 2024, doi
   10.3389/fnins.2024.1306047).** Deep model built for robustness to HD-sEMG
   electrode shift and electrode damage. Context for 2.4.4 / 5.7.
4. **Myoformer: sEMG missing-signal recovery for gesture recognition
   (Biomedical Signal Processing and Control, 2023).** Detects and repairs
   aberrant / missing channels rather than training against their loss;
   contrast rather than precedent.
5. **EEG side.** Channel dropout is a standard entry in EEG augmentation
   catalogues; e.g. "Data Augmentation: Using Channel-Level Recombination to
   Improve Classification Performance for Motor Imagery EEG" (PMC7990774) and
   review coverage listing channel dropout / perturbation among the six common
   biosignal augmentations. Supports the sentence that the technique is
   established cross-domain, not invented here.

**On an occlusion / channel-ablation profile of a deep sEMG model
specifically:** no prior report found that matches section 4.8.2's protocol (per
held-out subject, per channel, drop in macro-F1 from zeroing one channel, over a
deep model under LOSO). The nearest are generic occlusion-sensitivity for
temporal sensor data (channel importance by zeroing the j-th channel and summing
over time) and Zhang et al. 2026's test-time single-sensor-failure map on
classical pipelines. Section 4.8.2's profile on a deep sEMG model under LOSO
appears to be novel; that is a claim worth making cautiously and only after
Enam has done his own search.

## 3. Drafted sentences, for review and placement, NOT for pasting

House rules observed: no em dashes, decimals for statistics, no over-defensive
phrasing.

**For 2.4.4 (general grounding of dropout as augmentation):**

> Dropout was introduced as a unit-level regularizer that randomly zeros hidden
> activations during training (Srivastava et al., 2014). The same paper reports
> that replacing the Bernoulli mask with mean-one multiplicative Gaussian noise
> of matched variance performs comparably or slightly better (Table 10, MNIST
> 0.95 percent against 1.08 percent error), so the operative quantity at the
> unit level is the injected multiplicative variance rather than the zeroing
> itself.

**For 2.4.4 (structured, input-stream dropping):**

> Structured variants drop whole input streams rather than individual units.
> ModDrop removes entire sensing modalities during training so that a
> multi-modal network tolerates missing modalities at test time (Neverova et
> al., 2016). Channel dropout applied here is the single-modality analogue:
> whole electrode channels, rather than modalities, are removed per training
> sample.

**For 2.4.4 and 5.7 (the direct sEMG precedent):**

> The closest sEMG precedent trains on collections of electrode-channel subsets
> as augmentation to counter electrode shift and reports improved intersession
> performance (Pereira et al., ICASSP 2024). That work selects from a fixed
> vocabulary of subsets and targets within-subject session change; channel
> dropout here draws an independent per-channel mask and targets cross-subject
> transfer.

**For 2.4.4 and 4.8.2 (the measurement precedent):**

> Characterizing an sEMG pipeline by the performance lost when single channels
> are removed is an established diagnostic; Zhang et al. (2026) present such a
> map for classical feature-classifier pipelines under test-time single-sensor
> failure. Section 4.8.2 applies the same style of single-channel-loss profiling
> to a deep model across forty held-out subjects, and additionally measures how
> a training-time augmentation changes that profile.

**For 4.8.2's gain-jitter paragraph and 5.7 (the uncomfortable implication,
section 1.4.2, stated not hidden):**

> That gain jitter matches channel dropout at matched multiplicative variance
> (W-4) is consistent with the unit-level result that mean-one multiplicative
> noise matches Bernoulli dropout of the same variance (Srivastava et al.,
> 2014). Section 5.7 had advanced a domain-specific reading, that channel
> dropout works by simulating electrode liftoff, which predicts that the
> zeroing itself is the operative event. The W-4 result does not support that
> reading, and its agreement with a known property of dropout at a different
> level strengthens rather than weakens it. It was worth testing directly
> because the domain-specific reading was the one the thesis had committed to.

## 4. Where each source belongs (summary)

| Source | Retrieved | Placement |
|---|---|---|
| Srivastava et al. 2014 | full | 2.4.4; cited at 4.8.2 gain-jitter paragraph and 5.7 |
| Neverova et al. 2016 (ModDrop) | abstract + record | 2.4.4 |
| Pereira et al. 2024 (ICASSP, not preprint) | abstract; published status confirmed | 2.4.4 and 5.7; P-5 comparator |
| Zhang et al. 2026 | abstract + PubMed record (MDPI full text 403) | 2.4.4 and 4.8.2 |
| MAST 2024 (arXiv 2410.17261) | listing only | 2.4.4 and 5.7, if Enam pulls it |
| EEG channel-dropout augmentation refs | listing only | 2.4.4, one sentence on cross-domain use |

## P-0 outcome

Grounding located and verified to the extent open access allows. Two of the four
starting sources retrieved in full (Srivastava; Pereira abstract with published
status resolved). One correction to the plan: Pereira et al. is ICASSP 2024, a
peer-reviewed conference paper, not a preprint. One caution: Zhang et al. uses
"channel dropout" for a test-time fault, and needs disambiguating if cited
alongside the thesis term. Section 4.8.2's deep-model occlusion profile under
LOSO appears to have no direct prior, pending Enam's own search.
