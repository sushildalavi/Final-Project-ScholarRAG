INSERT INTO confidence_calibration (model_name, label, weights, metrics, dataset_size) VALUES (
  'msa_logistic_uploaded',
  'uploaded_mode',
  '{"w1": 1.290253, "w2": 0.0, "w3": 0.571947, "b": -1.371641}'::jsonb,
  '{"n": 170, "accuracy": 0.7, "auc": 0.7025, "pr_auc": 0.4622, "brier": 0.2008, "ece": 0.0718}'::jsonb,
  849
);
