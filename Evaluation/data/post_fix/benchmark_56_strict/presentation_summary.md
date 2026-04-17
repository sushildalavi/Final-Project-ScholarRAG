# ScholarRAG Post-Fix Evaluation Summary

## Headline

- Targeted adversarial retrieval validation: 20/20 (100.0%)
- Targeted abstention validation: 12/12 (100.0%)
- Post-fix faithfulness query mean: 61.6% with 95% CI [48.9%, 73.2%]
- Post-fix claim support rate: 85.6% over 202 judged sentence-level claims
- Post-fix calibration sample: 187/192 labeled claims had complete live M/S/A values

## Key Caveats

- The targeted validation improved the exact failure mode, but it is not perfect. Remaining misses are still present in the adversarial and abstention sets.
- The post-fix faithfulness query distribution is harsh: median 0.83, with many zero-score queries in the worst decile.
- Sentence-level judge claims were available for 55/56 scored queries. Query-level `overall_score` covers the full scored set and is the more reliable headline.
- Calibration results are now leakage-free, but they are based on a small complete-feature subset: 187 claims spanning 50 queries and 6 papers.

## Leakage-Free Calibration Snapshot

- Group-by-paper M+S accuracy: 77.3%
- Group-by-paper M+S macro-F1: 0.432
- Group-by-paper M+S Brier: 0.220
- Group-by-paper M+S ECE: 0.213
- Group-by-paper M+S ROC-AUC: 0.415
- Group-by-paper M+S PR-AUC: 0.779

## Remaining Targeted Failures


## Worst-Decile Faithfulness Queries

- `q1` (0.00): What is the main idea introduced in Attention Is All You Need?
- `q12` (0.00): How does RAG differ from a standard seq2seq model without retrieval?
- `q15` (0.00): Why does CLIP not need a fixed label set during pretraining?
- `q23` (0.00): Why do deeper plain networks suffer from degradation according to the paper?
- `q26` (0.00): What is the purpose of the skip connections in U-Net?
- `q29` (0.00): How does YOLOv3 perform object detection in a single-stage manner?
- `q31` (0.00): How does YOLOv3 balance speed and detection accuracy?
- `q32` (0.00): Why does YOLOv3 predict at multiple scales?
- `q34` (0.00): Why does the ViT paper argue that convolutions are not strictly necessary for image classification?
- `q38` (0.00): What role does Q-learning play in the DQN paper?

## Retrieval Ablation (120q)

- `dense`: Recall@1=0.750, Recall@10=1.000, MRR=0.837
- `bm25`: Recall@1=0.558, Recall@10=0.983, MRR=0.699
- `hybrid_rrf`: Recall@1=0.725, Recall@10=1.000, MRR=0.819
- `hybrid_rrf_rerank`: Recall@1=0.650, Recall@10=0.983, MRR=0.764

## Judge-vs-Human Validation

- Completed sample size: 50
- Accuracy: 0.860
- Cohen's kappa: 0.72
