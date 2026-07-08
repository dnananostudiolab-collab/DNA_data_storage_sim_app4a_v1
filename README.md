# DNA Storage Pipeline (File-Container)

This Streamlit app implements the **Case 2** design:

```text
File container bytes
→ automatic container-byte preparation (no active compression)
→ optional Reed–Solomon byte-level ECC
→ SM or R∞ DNA mapping
→ strand preparation
→ DNA error simulation
→ DNA decode
→ optional RS repair
→ decoded output + summarization
```

Important design rule:

- `SM` and `R∞` are the only DNA design/mapping options.
- Reed–Solomon is an **ECC option**, not a third DNA mapping.
- No active compression is used in the current version. After upload, the app automatically prepares the original file-container bytes as the storage payload.
- Panel 2 has no Run Compression button; the payload is prepared automatically after upload.
- Panel 5 shows the decoded output preview/download.
- Panel 6 is summarization-only: encoding statistics, error report, decode/recovery report, and domain-aware quality metrics.

## Quality metrics in Panel 6

- General files: byte accuracy, byte mismatch count, SHA256 match, file type/container match.
- Text: character accuracy, word accuracy, exact text match.
- Image: MSE, MAE, PSNR, approximate SSIM, exact pixel accuracy.
- Audio: WAV duration/sample rate/channels + waveform RMSE/PSNR/SNR. MP3/FLAC/OGG still use container/byte recovery unless additional audio decoding is added.
- Video: sampled-frame PSNR/SSIM if OpenCV can open both videos.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Recommended first test

Use substitution-only errors before testing insertions/deletions:

```text
Substitution: 0.005–0.01
Insertion: 0
Deletion: 0
RS data block size: 64 bytes
RS parity: 64 bytes/block
```

A 3–5% DNA substitution rate is a strong stress test. With SM, base errors can affect many protected bytes, so use high parity or lower DNA error rates for stable recovery.
