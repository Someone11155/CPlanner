# CPlanner

An on-device AI planner for university students (macOS 15+ / iOS 18+). A local LLM automatically classifies tasks into folders, with all inference performed on-device — no data leaves your machine.

## Download

**[Download the latest release (.dmg)](https://github.com/Someone11155/CPlanner/releases/latest)** — macOS 15+ (Apple Silicon). Current build: `v0.1.0` (early pre-release).

> ⚠️ The app is **ad-hoc signed** (no Apple Developer ID yet), so macOS Gatekeeper warns on first open. To run it: right-click the app → **Open** → **Open** again, or run `xattr -dr com.apple.quarantine /Applications/CPlanner.app`.
>
> On first launch you pick a model and it downloads on-device from Hugging Face (~4–8 GB) — nothing is bundled in the DMG.

See [Implementation Status](#implementation-status) for what works today and what's still pending.

## Features

- **On-device classification** — a local LLM sorts task titles into folders (courses / categories).
- **Gemma 4 E4B** — the single classification model: more accurate, prefill-chunk accelerated (~1.7s/task once warm). First launch goes straight to a download prompt (no model picker). The multi-model selection UI is retained in code for future re-introduction.
- **Manual correction + few-shot learning** — when you re-file a task, the correction is fed back as an in-context example for future classifications.
- **Two-way Apple Calendar sync** — events sync through a dedicated "CPlanner" calendar.
- **Privacy-focused** — the network is used only for the initial model download; all inference is on-device.

## Implementation Status

`v0.1.0` is an early pre-release. The core loop (capture → on-device classify → calendar sync) works; polish and several features are still pending.

**Done**
- On-device classification with Gemma 4 E4B (single model) — first launch prompts download directly; model-selection UI kept dormant in code for future models
- E4B full-bundle (prefill) auto-download from Hugging Face + eager prefill load + per-file resumable downloader (HTTP Range resume, 60s idle timeout drops a stalled connection, persistent staging folder so a restart picks up where it left off)
- Two-way Apple Calendar sync via a dedicated "CPlanner" calendar
- Manual re-filing + few-shot learning from corrections
- Exact-match shortcut (repeat titles skip the LLM)
- Compute-unit benchmark + tuned defaults (ANE for Gemma)
- Calendar UI: squircle cells, per-day task dots/checks, month-swipe transitions, Korean holiday highlighting
- Dark Todomate-style design, in-app Settings (⌘,), rotating tip bar

**Not yet implemented / rough edges**
- Classification settings: the exact-match toggle is UI-only (not yet wired); the confidence threshold is retained as an internal option (hidden from the UI — Gemma's confidence is binary, so it isn't user-facing)
- Prompt-length guard (many folders / long titles could approach the context ceiling)
- Background-execution policy (model memory release, launch-at-login, battery impact)
- Whole-screen UI/UX pass (information density, spacing, light mode)
- Performance optimization (task lookups, per-cell filtering)
- Empty states, keyboard navigation, ad-hoc folder creation, learning-history management UI
- Recurring calendar events (currently skipped with a warning)
- Download progress / cancel / resume UI (resume itself is implemented but not yet verified end-to-end on the full ~8 GB download)
- iOS build (developed and tested on macOS so far)

## Tech Stack

- **Language / UI:** Swift, SwiftUI
- **AI/ML:** CoreML, [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM) (Gemma 4 chunked SWA), [swift-transformers](https://github.com/huggingface/swift-transformers) (Hub download + tokenizers)
- **Models:** Gemma 4 E4B (CoreML)
- **Platforms:** macOS 15.0+, iOS 18.0+

## Getting Started

1. Clone the repository.
2. Open `CPlanner.xcodeproj` in Xcode.
3. Build & run. **Models are not bundled** — on first launch, choose a model in the picker and it downloads automatically from Hugging Face (~4–8 GB each).
   - E4B is fetched as a full bundle including prefill chunks (`someone15/gemma-4-E4B-coreml`), so prompt processing uses the fast prefill path instead of a token-by-token fallback.

> Large model / tokenizer files are not tracked in git (see `.gitignore`). The first launch of each model pays a one-time on-device ANE compile; later launches load from the OS compile cache and are much faster.

## License

MIT License
