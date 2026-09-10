# ICAISD 2026 — format requirements

Source of truth: `template/ICAISD_2026_Template_original.zip`, downloaded 2026-08-04 from
the "Download Templates" page at https://www.icaisd.com/ktgmfiqw (the link is injected by
JS; the real endpoint is `https://www.ais.cn/service/open/archive/down/E5C57EBMFI5H9FJNN4RA879J`).
Archive was built 2025-09-04. Re-check before submitting in case it is revised.

## Verdict: ACM, not IEEE

The pack ships `acmart.cls` **v2.12 (2024/12/28)** plus `sample-sigconf.tex`, and the Word
path is ACM's own `acm_submission_template.docx` / `acm_mat_word_v2.dotm`. Confirms the
earlier guess from ICAISD 2025 having used ACM-ICPS. **Do not use the IEEE template.**

## Submission format is SINGLE column, not sigconf

The checklist is explicit: "should remain in a one-column format—please do not alter any
of the styles or margins." The zip's LaTeX sample is `sigconf` (two-column), which is the
*published* look, not the submission look. For LaTeX, submit as:

    \documentclass[manuscript,screen,review]{acmart}

`sample-sigconf.tex` line 40 says exactly this. Typesetting to the final two-column form is
done by ACM's production editors after acceptance — the checklist says so outright, so do not
burn time fighting the layout.

## Hard requirements

| Item | Requirement |
|---|---|
| Length | **≥ 8 pages single-column** (a minimum, not a cap) |
| Page setup | Letter 21.59 × 27.94 cm; margins T 3.1 / B 5.01 / L 2.54 / R 3.6 cm |
| Keywords | ≥ 3 |
| CCS CONCEPTS | Required, generated from https://dl.acm.org/ccs (pick a 3rd-level term, paste verbatim) |
| References | ≥ 5, recent, must carry years, **authors from ≥ 3 different countries**, cited in text as superscript `[n]` |
| Corresponding author | Footnote "Corresponding author" bottom-left of page 1; all author emails listed; first author's email must be live (copyright forms go there) |
| ORCID | Needed for every author at publication-agreement time — register at https://orcid.org/register |
| Figures | High-res (text legible at 100%), no CJK characters, one caption per figure, multi-panel merged into a single image, caption says "Figure 1." in full with a trailing period, referenced in text in order |
| Tables | Editable format, not images. "Table 1." in full, trailing period, referenced in order |
| Equations | Editable (MathType or Word's editor if going the Word route); numbered (1), (2), … with the number *outside* the equation |
| Files | Site asks for **Word + PDF**. See open question below. |

## Open questions — email icaisd@163.com

1. **Word vs LaTeX.** The site's submission page asks for "full paper in Word + PDF format,"
   but the pack ships a full LaTeX path. The checklist's demands (editable tables, MathType
   equations, Word style tags) are all Word-shaped. Confirm a LaTeX-produced PDF is accepted
   before writing, since the paper is LaTeX-native.
2. **Blinding.** The template's double-blind paragraph is generic ACM boilerplate. ICAISD's own
   checklist requires all author emails *and* a corresponding-author footnote **in the submitted
   file**, which is incompatible with double-blind — so this is almost certainly single-blind.
   Worth one line of confirmation anyway.

## Two notes that affect what we write

- **EI favors empirical work.** The submission notice states EI review "偏重于实证多于综述" —
  weights empirical over survey, and wants papers centered on models, systems, experiments,
  and data analysis. A measured benchmark with a Pareto frontier is exactly that shape.
- **ACM generative-AI policy.** AI tools may not be listed as authors. Any AI-generated text,
  figures, tables, or code must be disclosed proportionally in the Acknowledgments; pure
  grammar/spelling assistance needs no disclosure. The pack includes suggested "Use of AI"
  wording, and notes that heavy AI polishing has previously tripped the Morressier
  similarity check.

## Commercial-venue caveat

The included 投稿须知 (submission notice) is a pre-registration agreement covering fees:
refunds if the paper is not submitted to the publisher within 3 months, a 50%-off registration
at another of the organizer's conferences if EI/Scopus indexing exceeds 12 months, and a 30%
refund if the author terminates. Indexing is explicitly **not guaranteed** — proceedings are
"submitted to" EI Compendex / Scopus, and acceptance by those databases is a separate process.
Budget for a registration fee and treat indexing as probable, not certain.

## Deadlines (site, verified 2026-08-04)

- Paper submission: **2026-09-09**
- Notification: 2026-10-13
- Registration: 2026-10-29
- Conference: 2026-11-13 to 11-15, Shanghai
- Submission portal: https://paper-sub.com/paper/6BBFZN
