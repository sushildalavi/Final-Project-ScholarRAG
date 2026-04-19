INSERT INTO confidence_calibration (model_name, label, weights, metrics, dataset_size) VALUES (
  'msa_logistic_public',
  'public_mode',
  '{"w1": 2.168363, "w2": 0.0, "w3": 0.928912, "b": -2.300233}'::jsonb,
  '{"n": 64, "accuracy": 0.8594, "auc": 0.9018, "pr_auc": 0.8532, "brier": 0.1354, "ece": 0.1317}'::jsonb,
  319
);
