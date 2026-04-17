#!/usr/bin/env python3

from __future__ import annotations

import nbformat
from pathlib import Path


NOTEBOOK = Path("Evaluation/ScholarRAG_Evaluation.ipynb")


def _update_notebook() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    nb.cells[0].source = """# ScholarRAG — Evaluation Dashboard (Post-Fix, Strict-Grounding)

Comprehensive post-fix visualizations for targeted retrieval/abstention validation, strict-grounding faithfulness/coverage tradeoffs, leakage-aware M/S/A calibration, fresh IAA, cross-corpus generalization, and formal significance tests.
"""

    nb.cells[1].source = """import json, csv, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from collections import Counter

sns.set_theme(style="whitegrid", font_scale=1.15)
PAL = ["#FF6F00", "#FFA000", "#FFD54F", "#4FC3F7", "#0288D1", "#01579B", "#E65100"]
sns.set_palette(PAL)

DATA = pathlib.Path("data")
POST = DATA / "post_fix/benchmark_56_strict"
BASE = DATA / "post_fix/benchmark_56_baseline_before_strict"

post_validation = json.loads((DATA / "post_fix/post_fix_validation.json").read_text())
faith = json.loads((POST / "faithfulness_distribution.json").read_text())
faith_base = json.loads((BASE / "faithfulness_distribution.json").read_text())
calib = json.loads((POST / "calibration_robustness.json").read_text())
calib_base = json.loads((BASE / "calibration_robustness.json").read_text())
headline = json.loads((POST / "headline_metrics.json").read_text())
headline_base = json.loads((BASE / "headline_metrics.json").read_text())
judge_raw = json.loads((POST / "judge_eval_post_fix.json").read_text())
judge_base = json.loads((BASE / "judge_eval_post_fix.json").read_text())
iaa_fresh = json.loads((DATA / "post_fix/fresh_iaa_report.json").read_text())
pub_data = json.loads((DATA / "public_search/public_search_eval.json").read_text())
cross = json.loads((DATA / "post_fix/cross_corpus_scifact_lite.json").read_text())
sig = json.loads((DATA / "post_fix/significance_ab_tests.json").read_text())
clean_a = json.loads((DATA / "human_labels/claim_scores_scored_clean_summary.json").read_text())
retrieval_ablation = json.loads((DATA / "post_fix/retrieval_ablation_120q.json").read_text())

claims_csv = pd.read_csv(POST / "claim_annotations.csv")
df_claim_labeled = claims_csv[claims_csv["support_label"].isin(["supported", "unsupported"])].copy()
df_claim_complete = df_claim_labeled.dropna(subset=["msa_M", "msa_S", "msa_A"]).copy()
df_claim_complete["label"] = df_claim_complete["support_label"].str.lower()

claim_rows = []
for c in judge_raw.get("claims", []):
    s = c.get("supported")
    if s is None:
        continue
    claim_rows.append({
        "query_id": str(c.get("query_id") or ""),
        "query": str(c.get("query") or ""),
        "supported": bool(s),
    })
df_claims = pd.DataFrame(claim_rows)

if not df_claims.empty:
    q_stats = df_claims.groupby(["query_id", "query"])["supported"].agg(["mean", "size"]).reset_index()
    q_stats = q_stats.rename(columns={"mean": "support_rate", "size": "n_claims"})
    df_query_scores = q_stats
else:
    df_query_scores = pd.DataFrame(columns=["query_id", "query", "support_rate", "n_claims"])

claim_rows_base = []
for c in judge_base.get("claims", []):
    s = c.get("supported")
    if s is None:
        continue
    claim_rows_base.append({
        "query_id": str(c.get("query_id") or ""),
        "query": str(c.get("query") or ""),
        "supported": bool(s),
    })
df_claims_base = pd.DataFrame(claim_rows_base)
if not df_claims_base.empty:
    df_query_scores_base = (
        df_claims_base.groupby(["query_id", "query"])["supported"].agg(["mean", "size"]).reset_index()
        .rename(columns={"mean": "support_rate", "size": "n_claims"})
    )
else:
    df_query_scores_base = pd.DataFrame(columns=["query_id", "query", "support_rate", "n_claims"])

n_sup = int(df_claims["supported"].sum()) if not df_claims.empty else 0
n_unsup = int((~df_claims["supported"]).sum()) if not df_claims.empty else 0

strict_query_mean = float(df_query_scores["support_rate"].mean()) if not df_query_scores.empty else 0.0
base_query_mean = float(df_query_scores_base["support_rate"].mean()) if not df_query_scores_base.empty else 0.0

judge_a = pd.read_csv(DATA / "post_fix/benchmark_56/judge_human_validation_sample.csv")
judge_b = pd.read_csv(DATA / "post_fix/benchmark_56/judge_human_validation_sample_annotator_b.csv")
iaa_join = (
    judge_a[["sample_id", "human_label"]]
    .rename(columns={"human_label": "human_label_a"})
    .merge(
        judge_b[["sample_id", "human_label"]].rename(columns={"human_label": "human_label_b"}),
        on="sample_id",
        how="inner",
    )
)

for col in ("human_label_a", "human_label_b"):
    iaa_join[col] = iaa_join[col].astype(str).str.lower().str.strip()
iaa_join = iaa_join[iaa_join["human_label_a"].isin(["supported", "unsupported"])]
iaa_join = iaa_join[iaa_join["human_label_b"].isin(["supported", "unsupported"])]

cm_fresh = pd.crosstab(iaa_join["human_label_a"], iaa_join["human_label_b"])
cm_fresh = cm_fresh.reindex(index=["supported", "unsupported"], columns=["supported", "unsupported"], fill_value=0)

def _kappa_band(v: float) -> str:
    if v < 0.20:
        return "slight"
    if v < 0.40:
        return "fair"
    if v < 0.60:
        return "moderate"
    if v < 0.80:
        return "substantial"
    return "almost perfect"

iaa = {
    "n": int(len(iaa_join)),
    "confusion_matrix": {
        "sup_sup": int(cm_fresh.loc["supported", "supported"]),
        "sup_unsup": int(cm_fresh.loc["supported", "unsupported"]),
        "unsup_sup": int(cm_fresh.loc["unsupported", "supported"]),
        "unsup_unsup": int(cm_fresh.loc["unsupported", "unsupported"]),
    },
    "cohens_kappa": float(iaa_fresh["claim_sheet"]["cohen_kappa"]),
    "agreement_count": int((iaa_join["human_label_a"] == iaa_join["human_label_b"]).sum()),
    "agreement_pct": round(float(iaa_fresh["claim_sheet"]["agreement"]) * 100, 1),
    "interpretation": _kappa_band(float(iaa_fresh["claim_sheet"]["cohen_kappa"])),
    "source_note": "annotator-B generated by rubric script (not independent second human)",
}

print(
    f"Loaded strict post-fix artifacts: {len(df_query_scores)} query scores, "
    f"{len(df_claims)} judged claims (baseline: {len(df_claims_base)}), "
    f"{post_validation['adversarial']['count']} adversarial + {post_validation['abstention']['count']} abstention checks"
)
print(
    f"Faithfulness query mean baseline→strict: {base_query_mean:.3f} → {strict_query_mean:.3f}; "
    f"claim volume baseline→strict: {len(df_claims_base)} → {len(df_claims)}"
)
print(
    f"Fresh claim IAA κ={iaa['cohens_kappa']:.4f} on n={iaa['n']} ({iaa['source_note']})"
)
"""

    nb.cells[5].source = """---
## 2. Post-Fix Faithfulness Evaluation (Strict Grounding + Coverage Tradeoff)
"""

    nb.cells[13].source = """---
## 5. Inter-Annotator Agreement (Fresh Claim Sheet)

Claim IAA below uses the fresh sample sheet; annotator-B is rubric-generated and should be presented as synthetic QA, not independent human adjudication.
"""

    nb.cells[14].source = """cm = iaa["confusion_matrix"]
conf_mat = np.array([[cm["sup_sup"], cm["sup_unsup"]],
                     [cm["unsup_sup"], cm["unsup_unsup"]]])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Heatmap
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Supported", "Unsupported"],
            yticklabels=["Supported", "Unsupported"],
            ax=axes[0], cbar_kws={"shrink": 0.8}, linewidths=2, linecolor="white")
axes[0].set_xlabel("Annotator B"); axes[0].set_ylabel("Annotator A")
axes[0].set_title(f"Fresh Claim IAA Confusion (n={iaa['n']})", fontweight="bold")

# Kappa gauge
kappa = iaa["cohens_kappa"]
kappa_ranges = [(0, 0.20, "Slight"), (0.20, 0.40, "Fair"),
                (0.40, 0.60, "Moderate"), (0.60, 0.80, "Substantial"),
                (0.80, 1.00, "Almost\\nPerfect")]
colors_kappa = ["#E53935", "#FB8C00", "#FFD54F", "#66BB6A", "#2E7D32"]

for (lo, hi, lbl), col in zip(kappa_ranges, colors_kappa):
    axes[1].barh(0, hi - lo, left=lo, height=0.5, color=col, edgecolor="white", alpha=0.7)
    axes[1].text((lo + hi)/2, -0.45, lbl, ha="center", va="top", fontsize=9, fontweight="bold")

axes[1].axvline(kappa, color="black", lw=3, zorder=5)
axes[1].plot(kappa, 0, marker="v", color="black", markersize=16, zorder=6)
axes[1].text(kappa, 0.35, f"κ = {kappa:.4f}", ha="center", fontsize=14, fontweight="bold")

axes[1].set_xlim(0, 1); axes[1].set_ylim(-0.8, 0.7)
axes[1].set_xlabel("Cohen's Kappa"); axes[1].set_yticks([])
axes[1].set_title(f"IAA Band: {iaa['interpretation'].title()}", fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig_iaa.png", dpi=180, bbox_inches="tight")
plt.show()

print(f"Agreement: {iaa['agreement_count']}/{iaa['n']} ({iaa['agreement_pct']}%)")
print(f"Cohen's kappa: {kappa:.4f} — {iaa['interpretation']}")
print(f"Disclosure: {iaa['source_note']}")
"""

    nb.cells[19].source = """---
## 7. Combined Post-Fix Summary Dashboard

This combines targeted validation, strict-grounding faithfulness, fresh IAA, cross-corpus generalization, and calibration at a glance.
"""

    nb.cells[20].source = """fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

# (0,0) Targeted validation pass rates
ax1 = fig.add_subplot(gs[0, 0])
vals = [post_validation['adversarial']['pass_rate'], post_validation['abstention']['pass_rate']]
ax1.bar(['Adv', 'Abs'], vals, color=[PAL[4], PAL[0]], edgecolor='white')
for i, v in enumerate(vals):
    ax1.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=10, fontweight='bold')
ax1.set_ylim(0, 1.05); ax1.set_title('Targeted Validation', fontweight='bold')

# (0,1) Strict-grounding query-level faithfulness
ax2 = fig.add_subplot(gs[0, 1])
fs = df_query_scores['support_rate'].to_numpy()
ax2.hist(fs, bins=np.linspace(0, 1, 11), color=PAL[0], edgecolor='white')
ax2.axvline(fs.mean(), color=PAL[5], ls='--', lw=2)
ax2.set_title('Strict Faithfulness (Query-Level)', fontweight='bold')

# (0,2) Cross-corpus recall@10
ax3 = fig.add_subplot(gs[0, 2])
cc_modes = ['dense_lsi', 'bm25', 'hybrid_rrf', 'hybrid_weighted']
cc_modes = [m for m in cc_modes if m in cross['metrics']]
cc_r10 = [cross['metrics'][m]['recall@10']['mean'] for m in cc_modes]
ax3.bar(cc_modes, cc_r10, color=[PAL[3], PAL[0], PAL[5], PAL[4]][:len(cc_modes)], edgecolor='white')
ax3.set_ylim(0, 1.0); ax3.set_title('Cross-Corpus Recall@10', fontweight='bold')
ax3.tick_params(axis='x', rotation=25)

# (0,3) Fresh IAA
ax4 = fig.add_subplot(gs[0, 3])
cm = iaa['confusion_matrix']
conf_mat = np.array([[cm['sup_sup'], cm['sup_unsup']], [cm['unsup_sup'], cm['unsup_unsup']]])
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlOrRd', xticklabels=['Sup', 'Unsup'], yticklabels=['Sup', 'Unsup'], ax=ax4, cbar=False, linewidths=1.5, linecolor='white')
ax4.set_title(f"Fresh IAA κ={iaa['cohens_kappa']:.3f}", fontweight='bold')

# (1,0:2) M/S/A distributions
components = [('msa_M', 'M'), ('msa_S', 'S'), ('msa_A', 'A')]
for i, (col, title) in enumerate(components):
    ax = fig.add_subplot(gs[1, i])
    for label, color in [('supported', PAL[4]), ('unsupported', PAL[6])]:
        subset = df_claim_complete[df_claim_complete['label'] == label][col]
        if not subset.empty:
            ax.hist(subset, bins=12, alpha=0.7, color=color, label=label.capitalize(), edgecolor='white')
    ax.set_title(f"{title} Distribution", fontweight='bold', fontsize=10)
    if i == 0:
        ax.legend(fontsize=8)

# (1,3) Key metrics text
ax5 = fig.add_subplot(gs[1, 3])
ax5.axis('off')
combined_sig = sig['validation_mcnemar']['combined']
metrics_text = (
    f"━━━ Post-Fix Metrics ━━━\\n\\n"
    f"Adversarial pass:  {post_validation['adversarial']['passed']}/{post_validation['adversarial']['count']}\\n"
    f"Abstention pass:   {post_validation['abstention']['passed']}/{post_validation['abstention']['count']}\\n\\n"
    f"Faith mean (strict): {faith['overall']['query_level_bootstrap_ci_95']['mean']:.1%}\\n"
    f"Faith mean (base):   {faith_base['overall']['query_level_bootstrap_ci_95']['mean']:.1%}\\n"
    f"Claims strict/base:  {faith['overall']['n_claims']}/{faith_base['overall']['n_claims']}\\n\\n"
    f"Cross-corpus R@10:\\n"
    f"  BM25 {cross['metrics']['bm25']['recall@10']['mean']:.2f} | Dense {cross['metrics']['dense_lsi']['recall@10']['mean']:.2f}\\n"
    f"  Hybrid-RRF {cross['metrics']['hybrid_rrf']['recall@10']['mean']:.2f}\\n\\n"
    f"Fresh IAA (κ):     {iaa['cohens_kappa']:.4f}\\n"
    f"  note: synthetic annotator-B\\n\\n"
    f"A/B sign test (combined):\\n"
    f"  improved={combined_sig['n_improved']}, regressed={combined_sig['n_regressed']}\\n"
    f"  p(two-sided)={combined_sig['p_value_two_sided']:.5f}"
)
ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
         va='top', ha='left', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1', edgecolor='#FF6F00', alpha=0.9))

fig.suptitle('ScholarRAG — Post-Fix Evaluation Summary', fontsize=16, fontweight='bold', y=1.01)
plt.savefig('figures/fig_summary_dashboard.png', dpi=180, bbox_inches='tight')
plt.show()
"""

    nb.cells[22].source = """# Dataset composition + strict-grounding tradeoff + cross-corpus MRR
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Strict vs baseline tradeoff
axes[0].bar(['Baseline', 'Strict'], [faith_base['overall']['n_claims'], faith['overall']['n_claims']],
            color=[PAL[3], PAL[0]], edgecolor='white', width=0.55)
axes[0].set_ylabel('Judged claims')
axes[0].set_title('Claim Coverage Tradeoff', fontweight='bold')
for i, v in enumerate([faith_base['overall']['n_claims'], faith['overall']['n_claims']]):
    axes[0].text(i, v + 3, str(v), ha='center', fontsize=10, fontweight='bold')

# Complete vs skipped features
axes[1].pie([calib['n_claims_complete'], calib['n_claims_skipped_incomplete_features']],
            labels=[f"Complete ({calib['n_claims_complete']})", f"Skipped ({calib['n_claims_skipped_incomplete_features']})"],
            colors=[PAL[4], PAL[1]], autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Strict Run Feature Completeness', fontweight='bold')

# Cross-corpus MRR by mode
modes = ['dense_lsi', 'bm25', 'hybrid_rrf', 'hybrid_weighted']
modes = [m for m in modes if m in cross['metrics']]
mrr_vals = [cross['metrics'][m]['mrr']['mean'] for m in modes]
axes[2].bar(modes, mrr_vals, color=[PAL[3], PAL[0], PAL[5], PAL[4]][:len(modes)], edgecolor='white')
axes[2].set_ylim(0, 1.0)
axes[2].set_title('Cross-Corpus MRR', fontweight='bold')
axes[2].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig('figures/fig_dataset.png', dpi=180, bbox_inches='tight')
plt.show()

print(f"Strict query mean support: {faith['overall']['query_level_bootstrap_ci_95']['mean']:.3f}")
print(f"Baseline query mean support: {faith_base['overall']['query_level_bootstrap_ci_95']['mean']:.3f}")
print(f"Strict claims: {faith['overall']['n_claims']} (baseline: {faith_base['overall']['n_claims']})")
print(f"Cross-corpus R@10 (BM25 vs Hybrid-RRF): {cross['metrics']['bm25']['recall@10']['mean']:.3f} vs {cross['metrics']['hybrid_rrf']['recall@10']['mean']:.3f}")
"""

    nb.cells[25].source = """---
## 9. Corrected Post-Fix Metrics Table (Presentation-Ready)

Includes strict-grounding tradeoffs, fresh IAA disclosure, cross-corpus generalization, and significance tests.
"""

    nb.cells[26].source = """summary_table = pd.DataFrame([
    {'Category': 'Targeted Validation', 'Metric': 'Adversarial Pass Rate', 'Value': f"{post_validation['adversarial']['passed']}/{post_validation['adversarial']['count']} ({post_validation['adversarial']['pass_rate']:.1%})"},
    {'Category': 'Targeted Validation', 'Metric': 'Abstention Pass Rate',  'Value': f"{post_validation['abstention']['passed']}/{post_validation['abstention']['count']} ({post_validation['abstention']['pass_rate']:.1%})"},

    {'Category': 'Faithfulness (Baseline)', 'Metric': 'Query-level Mean', 'Value': f"{faith_base['overall']['query_level_bootstrap_ci_95']['mean']:.1%}"},
    {'Category': 'Faithfulness (Strict)', 'Metric': 'Query-level Mean', 'Value': f"{faith['overall']['query_level_bootstrap_ci_95']['mean']:.1%}"},
    {'Category': 'Faithfulness (Strict)', 'Metric': 'Query-level 95% CI', 'Value': f"[{faith['overall']['query_level_bootstrap_ci_95']['lo']:.1%}, {faith['overall']['query_level_bootstrap_ci_95']['hi']:.1%}]"},
    {'Category': 'Faithfulness Tradeoff', 'Metric': 'Judged Claims (Base → Strict)', 'Value': f"{faith_base['overall']['n_claims']} → {faith['overall']['n_claims']}"},
    {'Category': 'Faithfulness (Strict)', 'Metric': 'Claim Support Rate', 'Value': f"{faith['overall']['claim_support_rate']:.1%}"},

    {'Category': 'Calibration (Strict)', 'Metric': 'Raw Claims', 'Value': str(calib['n_claims_raw'])},
    {'Category': 'Calibration (Strict)', 'Metric': 'Labeled Claims', 'Value': str(calib['n_claims_labeled'])},
    {'Category': 'Calibration (Strict)', 'Metric': 'Complete M/S/A Claims', 'Value': str(calib['n_claims_complete'])},
    {'Category': 'Calibration (Strict)', 'Metric': 'Skipped Incomplete', 'Value': str(calib['n_claims_skipped_incomplete_features'])},
    {'Category': 'Calibration (Strict)', 'Metric': 'Group-by-paper M+S Accuracy', 'Value': f"{headline['post_fix_calibration']['group_by_paper_M+S']['accuracy']:.3f}"},
    {'Category': 'Calibration (Strict)', 'Metric': 'Group-by-paper M+S Macro-F1', 'Value': f"{headline['post_fix_calibration']['group_by_paper_M+S']['f1_macro']:.3f}"},
    {'Category': 'Calibration (Strict)', 'Metric': 'Group-by-paper M+S Brier', 'Value': f"{headline['post_fix_calibration']['group_by_paper_M+S']['brier']:.3f}"},
    {'Category': 'Calibration (Strict)', 'Metric': 'Group-by-paper M+S ECE', 'Value': f"{headline['post_fix_calibration']['group_by_paper_M+S']['ece']:.3f}"},

    {'Category': 'Fresh IAA (Claims)', 'Metric': "Cohen's Kappa", 'Value': f"{iaa_fresh['claim_sheet']['cohen_kappa']:.4f}"},
    {'Category': 'Fresh IAA (Claims)', 'Metric': 'Agreement', 'Value': f"{iaa_fresh['claim_sheet']['agreement']:.1%}"},
    {'Category': 'Fresh IAA (Disclosure)', 'Metric': 'Annotator-B Source', 'Value': 'Rubric-generated (synthetic), not independent human'},
    {'Category': 'Fresh IAA (Retrieval)', 'Metric': "Kappa (3-class / binary)", 'Value': f"{iaa_fresh['retrieval_sheet']['cohen_kappa_3class']:.4f} / {iaa_fresh['retrieval_sheet']['cohen_kappa_binary']:.4f}"},

    {'Category': 'Cross-Corpus (SciFact-lite)', 'Metric': 'Dense Recall@10 / MRR', 'Value': f"{cross['metrics']['dense_lsi']['recall@10']['mean']:.3f} / {cross['metrics']['dense_lsi']['mrr']['mean']:.3f}"},
    {'Category': 'Cross-Corpus (SciFact-lite)', 'Metric': 'BM25 Recall@10 / MRR', 'Value': f"{cross['metrics']['bm25']['recall@10']['mean']:.3f} / {cross['metrics']['bm25']['mrr']['mean']:.3f}"},
    {'Category': 'Cross-Corpus (SciFact-lite)', 'Metric': 'Hybrid-RRF Recall@10 / MRR', 'Value': f"{cross['metrics']['hybrid_rrf']['recall@10']['mean']:.3f} / {cross['metrics']['hybrid_rrf']['mrr']['mean']:.3f}"},
    {'Category': 'Cross-Corpus (SciFact-lite)', 'Metric': 'Hybrid-Weighted Recall@10 / MRR', 'Value': f"{cross['metrics']['hybrid_weighted']['recall@10']['mean']:.3f} / {cross['metrics']['hybrid_weighted']['mrr']['mean']:.3f}"},

    {'Category': 'Significance', 'Metric': 'Validation sign test (combined, two-sided p)', 'Value': f"{sig['validation_mcnemar']['combined']['p_value_two_sided']:.5f}"},
    {'Category': 'Significance', 'Metric': 'Faithfulness Wilcoxon (one-sided p)', 'Value': f"{sig['faithfulness_wilcoxon']['p_value_one_sided_improvement']:.5f}"},

    {'Category': 'Legacy 634-claim A Fix', 'Metric': 'A mean supported (old → clean)', 'Value': f"{clean_a['msa_A_old_by_label']['mean']['supported']:.3f} → {clean_a['msa_A_clean_by_label']['mean']['supported']:.3f}"},
    {'Category': 'Legacy 634-claim A Fix', 'Metric': 'A mean unsupported (old → clean)', 'Value': f"{clean_a['msa_A_old_by_label']['mean']['unsupported']:.3f} → {clean_a['msa_A_clean_by_label']['mean']['unsupported']:.3f}"},
])

display(summary_table.style.set_properties(**{'text-align': 'left'}).set_caption(
    'ScholarRAG — Corrected Post-Fix Evaluation Metrics (Strict Grounding + Cross-Corpus + Significance)'
).hide(axis='index'))

print('Updated figures saved under Evaluation/figures/ and post-fix detail files under Evaluation/data/post_fix/.')
"""

    nbformat.write(nb, NOTEBOOK)


if __name__ == "__main__":
    _update_notebook()
    print(f"Patched {NOTEBOOK}")
