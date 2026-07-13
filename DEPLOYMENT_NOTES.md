# Deployment Notes

Running checklist of deployment-related decisions and things to verify
before actually going live. Not a full deployment guide yet - just
tracking stuff as it comes up so we don't have to rediscover it later
(this becomes the real deployment guide for Epic 6 Story 2).

## Target: Hugging Face Spaces (Streamlit SDK)

Picked over Streamlit Community Cloud because:
- Free CPU-basic tier gives 2 vCPU / 16GB RAM vs. Streamlit Cloud's
  guaranteed 1GB - bert-base-uncased alone is ~440MB in memory, on top
  of torch/transformers + tensorflow overhead, so 1GB was a real risk.
- Model weights aren't in git right now (.gitignore excludes
  models/bilstm/* and models/bert_emotion_model_final/* - large
  binaries, need separate hosting). HF Spaces + the HF Hub are built
  for exactly this - push the BERT weights to a Hub model repo and
  point from_pretrained() at the repo ID instead of a local folder.
  BiLSTM's files are small enough to just commit directly if we're ok
  with binary files in git for those.

## Checklist

- [ ] **Re-run the TF/PyTorch isolation tests on the actual HF Spaces
      container before trusting the in-process architecture there too.**
      Locally (Windows, Python 3.12.10, tensorflow-cpu 2.16.1, torch
      2.11.0+cu128, transformers 4.44.2) both test_isolation_minimal.py
      and test_isolation_realistic.py passed 3/3 - see the updated
      notes in scripts/_integration_check_bilstm.py and
      scripts/verify_model_integration.py for the full history. That
      result is Windows-specific though - native library conflicts
      like this one can be platform/build-specific, so a clean result
      here doesn't guarantee a clean result on HF Spaces' Linux
      container. Just copy both test scripts into the Space and run
      them once before relying on the same in-process design there.
- [ ] Push BERT weights to a Hugging Face Hub model repo, update
      bert_predictor.py's MODEL_PATH to the Hub repo ID.
- [ ] Decide how to handle BiLSTM's model.keras / tokenizer.pickle -
      commit directly (small enough) or same Hub approach for
      consistency.
- [ ] logs/ CSV persistence won't survive restarts on non-persistent
      disk (HF Spaces free tier disk is non-persistent) - fine for a
      single demo session, but anything meant to accumulate over time
      needs a different storage plan. Not blocking for assessment demo
      purposes, just don't expect the CSV to survive a Space restart.
- [ ] Gemini API key / provider config via HF Spaces' "Variables and
      Secrets" instead of a local .env.
- [ ] requirements.txt currently has commented-out CUDA extra-index-url
      lines for local RTX 50-series setup - clean those up or make sure
      they don't get uncommented by accident for the deployed build
      (Spaces free tier is CPU only).
