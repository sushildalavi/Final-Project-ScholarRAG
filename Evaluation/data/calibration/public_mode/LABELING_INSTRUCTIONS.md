# ScholarRAG Public-Mode Labeling — Instructions for Coders A, B, C

## What this is

ScholarRAG is a research literature assistant. When a user asks a question in **public-research mode**, it retrieves papers from Semantic Scholar, arXiv, CrossRef, OpenAlex, Springer, Elsevier, and IEEE, then generates a cited answer. Each generated sentence (a *claim*) cites one or more evidence snippets (paper abstracts / snippets).

We need to know, for each (claim, evidence) pair, whether the evidence actually supports the claim — or whether the system misattributed or hallucinated. Three coders will label the same set independently; we'll use your labels to compute inter-annotator agreement and re-calibrate the confidence model for public-mode queries.

**This is a different round from the uploaded-mode labeling you did previously.** The claims here come from live public-search results, not PDFs in a user's corpus. Evidence text will look like paper abstracts rather than chunked PDF body text.

---

## Files you received

One file per coder:

- `ScholarRAG_PublicMode_Coder_A.xlsx`
- `ScholarRAG_PublicMode_Coder_B.xlsx`
- `ScholarRAG_PublicMode_Coder_C.xlsx`

**Open only your own file.** The rows and order are identical across all three; that's how we merge labels afterward.

---

## Definitions

### supported
The evidence passage **explicitly states or strongly implies** the specific factual content of the claim. A reviewer reading the evidence would accept it as the source for the claim.

**Quick test:** *Can you point to a sentence in the evidence that a reviewer would accept as grounding for this claim?*

### unsupported
The evidence does **not** substantiate the claim. This includes:

- Evidence is about a different topic or paper than what the claim asserts.
- Claim is a system abstention like *"I only found X mentioned in…"* — always unsupported.
- Evidence is topically related but does not make the specific assertion.
- Evidence contradicts the claim.
- Claim overstates what the evidence says.

**Quick test:** *If the claim used this evidence as its only source, would a reviewer call it hallucinated or misattributed?*

---

## How to label one row

1. Read **col E (`claim_text`)**. What is the specific factual claim?
2. Read **col F (`evidence_text`)**. This is the snippet the system cited. Read the whole thing, not just the first sentence.
3. Decide: does the evidence support the specific claim?
4. Click **col G (`Your Label`)** — pick `supported` or `unsupported` from the dropdown.
5. If uncertain, type a short reason in **col H (`Notes`)**. Examples:
   - *"evidence is a review of Transformers, claim is specifically about BERT pretraining"*
   - *"abstract mentions the topic but not the specific mechanism claimed"*
   - *"system abstention"*
   - *"only partially supported — lean supported"*
6. Move on. ~30–60 seconds per row.

---

## Public-mode specific notes (read these)

The evidence snippets come from external APIs, so a few things differ from the uploaded round:

1. **Evidence is usually an abstract.** Expect 2-4 paragraphs of paper-abstract text. If the abstract mentions the concept in passing but doesn't actually make the claim's specific assertion, label **unsupported**.

2. **Sometimes the retrieval picks a tangentially related paper.** A query about "self-attention in Transformers" might return a paper on "attention in radiology models." If the evidence doesn't directly support the claim, mark **unsupported** even if it sounds topical.

3. **Citations to review papers are common.** A generic review saying *"Transformers use attention mechanisms"* can legitimately support a basic definitional claim — that's **supported**. But if the claim makes a specific sub-assertion (e.g., *"Transformers use 8 attention heads"*) and the review only mentions attention generically, that's **unsupported**.

4. **Acronym / sense collisions.** If the claim is about "BERT" and the evidence is clearly about a different BERT (a person's name, a different system), label **unsupported** and note it.

5. **If the evidence is in another language or looks garbled:** label **unsupported** and note "unreadable evidence."

---

## Blind-labeling rules (these matter for the IAA statistic)

1. **Do NOT share, compare, or discuss labels with the other coders until everyone finishes.**
2. **Do NOT use outside knowledge.** Use only the evidence passage in col F. Do not Google the paper, do not rely on what you remember. Only what's in col F.
3. **Do not skip rows.** Every row needs a label.
4. **Use the dropdown only** for col G. No free text in that column.
5. **Trust your first read.** Don't second-guess unless genuinely ambiguous.
6. **Save often.** Cmd-S / Ctrl-S.

---

## Worked examples

| Claim | Evidence snippet | Correct label | Why |
|-------|-----------------|---------------|-----|
| *"BERT is pre-trained using a masked language model objective."* | *"We introduce BERT, which is pre-trained using a masked language model (MLM) objective that randomly masks tokens."* | **supported** | Evidence explicitly states MLM. |
| *"Self-attention computes weighted sums of Value vectors."* | *"The paper evaluates summarization performance using ROUGE metrics."* | **unsupported** | Evidence is on a different topic. |
| *"The Transformer achieves 28.4 BLEU on WMT 2014 English-German."* | *"The Transformer model uses multi-head attention and outperforms prior work on machine translation."* | **unsupported** | Evidence mentions the model beats prior work but not the specific 28.4 BLEU number. |
| *"Chain-of-thought prompting improves reasoning on arithmetic tasks."* | *"CoT prompting, where models generate intermediate steps, shows large gains on math word problems and arithmetic benchmarks."* | **supported** | Evidence directly supports the specific claim. |
| *"I only found `Self-Attention` mentioned in profile/course context in your uploaded files."* | Any evidence. | **unsupported** | System-level abstention. |

---

## Tabs in your workbook

- **Instructions** — short version of this doc.
- **Rubric** — same definitions and examples as above.
- **Labeling** — the rows. Work here.
- **Summary** — live progress counter. Watch "% complete" climb. If your supported/unsupported ratio drifts far from what you expected, re-read the rubric — you may be getting strict or lenient over time.

---

## Time budget

Depends on how many claims the extraction produces (target is ~150–250 rows). At ~30–60 sec per row, expect **1.5–3 hours**. Break into two sittings if you can.

---

## When you finish

1. Check col G — every row should have a label. Summary tab "Remaining" should read 0.
2. Save the file. Do **not rename it**. Keep `ScholarRAG_PublicMode_Coder_X.xlsx` so we know which rater you are.
3. Send it back via the agreed channel.
4. Do not discuss labels with other coders until the merge + IAA analysis is complete.

---

## FAQ

**Q: The evidence is paywalled / truncated — only an abstract is shown.**
A: Fine. Label based on what's shown. If the visible text doesn't support the claim, it's **unsupported** regardless of whether the full paper might.

**Q: The claim says "X et al., 2017" and the evidence is from "Y et al., 2020". Does that matter?**
A: The attribution itself isn't what you're judging. Judge whether the content of the claim is supported by the content of the evidence. Wrong attribution → you can mark supported if content matches; put a note.

**Q: The evidence mentions the topic 20 times but never the specific assertion in the claim.**
A: **Unsupported.** Topic presence ≠ claim support.

**Q: The claim is very generic ("Transformers are used in NLP") and the evidence talks about a specific Transformer variant. Does that count?**
A: Yes, **supported.** Specific examples support generic claims. (The reverse is not true.)

**Q: Can I go back and change a label?**
A: Yes. Click the cell and change it. No submit button — only your saved state matters.

---

## Contact

Questions about the rubric or specific edge cases: reach out to the project lead **before** labeling further. If you're confused, another rater probably is too, and we want to resolve it once.
