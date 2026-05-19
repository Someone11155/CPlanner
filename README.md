# Cplanner

> ⚠️ **유기된 프로젝트입니다.** 더 이상 개발/유지보수되지 않습니다.

AI-powered planner app for university students using local LLM.

## Features
- **Local LLM Integration:** Uses Mistral 7B (via CoreML) for processing tasks entirely on-device.
- **Automated Task Classification:** Intelligently categorizes tasks into appropriate folders.
- **Privacy Focused:** No data leaves your device; all AI inferences are performed locally.
- **SwiftUI Based:** Modern and responsive user interface.

## Tech Stack
- **Language:** Swift
- **UI Framework:** SwiftUI
- **AI/ML:** CoreML, Mistral 7B, Tokenizers (HuggingFace)
- **Platforms:** macOS 15.0+, iOS 18.0+

## Getting Started
1. Clone the repository.
2. Open `CPlanner.xcodeproj` in Xcode.
3. Ensure you have the `StatefulMistral7BInstructInt4.mlpackage` and required tokenizers in the project resources.
4. Build and run.

## License
MIT License
