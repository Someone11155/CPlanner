# CPlanner

An on-device AI planner for university students (macOS 15+ / iOS 18+). A local LLM automatically classifies tasks into folders, with all inference performed on-device — no data leaves your machine.

## Features

- **On-device classification** — a local LLM sorts task titles into folders (courses / categories).
- **Multi-model backend** — pick a model on first launch, switchable in Settings:
  - **Gemma 4 E2B** — fast, ANE-optimized (~0.7s/task)
  - **Gemma 4 E4B** — more accurate, prefill-chunk accelerated (~1.7s/task once warm)
  - **Mistral 7B** — exposes softmax confidence (%)
- **Manual correction + few-shot learning** — when you re-file a task, the correction is fed back as an in-context example for future classifications.
- **Two-way Apple Calendar sync** — events sync through a dedicated "CPlanner" calendar.
- **Privacy-focused** — the network is used only for the initial model download; all inference is on-device.

## Tech Stack

- **Language / UI:** Swift, SwiftUI
- **AI/ML:** CoreML, [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM) (Gemma 4 chunked SWA), [swift-transformers](https://github.com/huggingface/swift-transformers) (Hub download + tokenizers)
- **Models:** Gemma 4 E2B / E4B (CoreML), Mistral 7B Instruct v0.3 (Int4, CoreML)
- **Platforms:** macOS 15.0+, iOS 18.0+

## Getting Started

1. Clone the repository.
2. Open `CPlanner.xcodeproj` in Xcode.
3. Build & run. **Models are not bundled** — on first launch, choose a model in the picker and it downloads automatically from Hugging Face (~4–8 GB each).
   - E4B is fetched as a full bundle including prefill chunks (`someone15/gemma-4-E4B-coreml`), so prompt processing uses the fast prefill path instead of a token-by-token fallback.

> Large model / tokenizer files are not tracked in git (see `.gitignore`). The first launch of each model pays a one-time on-device ANE compile; later launches load from the OS compile cache and are much faster.

## License

MIT License
