# ScholarRAG Post-Fix Evaluation Summary

## Headline

- Targeted adversarial retrieval validation: 20/20 (100.0%)
- Targeted abstention validation: 12/12 (100.0%)
- Post-fix faithfulness query mean: 50.5% with 95% CI [39.7%, 61.8%]
- Post-fix claim support rate: 45.4% over 546 judged sentence-level claims
- Post-fix calibration sample: 539/544 labeled claims had complete live M/S/A values

## Key Caveats

- The targeted validation improved the exact failure mode, but it is not perfect. Remaining misses are still present in the adversarial and abstention sets.
- The post-fix faithfulness query distribution is harsh: median 0.40, with many zero-score queries in the worst decile.
- Sentence-level judge claims were available for 55/56 scored queries. Query-level `overall_score` covers the full scored set and is the more reliable headline.
- Calibration results are now leakage-free, but they are based on a small complete-feature subset: 539 claims spanning 50 queries and 9 papers.

## Leakage-Free Calibration Snapshot

- Group-by-paper M+S accuracy: 53.2%
- Group-by-paper M+S macro-F1: 0.393
- Group-by-paper M+S Brier: 0.252
- Group-by-paper M+S ECE: 0.130
- Group-by-paper M+S ROC-AUC: 0.598
- Group-by-paper M+S PR-AUC: 0.568

## Remaining Targeted Failures


## Worst-Decile Faithfulness Queries

- `q1` (0.00): What is the main idea introduced in Attention Is All You Need?
- `q24` (0.00): How does ResNet make optimization easier?
- `q26` (0.00): What is the purpose of the skip connections in U-Net?
- `q28` (0.00): Why is U-Net especially useful when labeled data is limited?
- `q29` (0.00): How does YOLOv3 perform object detection in a single-stage manner?
- `q31` (0.00): How does YOLOv3 balance speed and detection accuracy?
- `q33` (0.00): How does the Vision Transformer convert an image into tokens?
- `q34` (0.00): Why does the ViT paper argue that convolutions are not strictly necessary for image classification?
- `q40` (0.00): What kind of input representation is used for the Atari agent?
- `q42` (0.00): Why is the method in the GCN paper considered semi-supervised?

## Retrieval Ablation (120q)

- `dense`: Recall@1=0.750, Recall@10=1.000, MRR=0.837
- `bm25`: Recall@1=0.558, Recall@10=0.983, MRR=0.699
- `hybrid_rrf`: Recall@1=0.725, Recall@10=1.000, MRR=0.819
- `hybrid_rrf_rerank`: Recall@1=0.650, Recall@10=0.983, MRR=0.764

## Judge-vs-Human Validation

- Completed sample size: 50
- Accuracy: 0.860
- Cohen's kappa: 0.72
