//
//  ModelInstaller.swift
//  Cplanner
//
//  Gemma 4 E2B (CoreML) 다운로드 + 로드를 john-rocky/CoreML-LLM 라이브러리에 위임.
//  라이브러리가 ~/Documents/Models/gemma4-e2b/ 에 chunk 4개 (swa decode) + embed weights + 토크나이저를
//  자동으로 캐시하고 ANE 컴파일까지 처리한다.
//
//  주: 원래는 Gemma 4 E2B로 가려 했으나 라이브러리 v1.9.0에서 Qwen3.5 경로가 미완성 상태로 확인됨
//  (chunk auto-detect가 chunk1.mlpackage만 찾고 Qwen은 chunk_a 네이밍, Qwen 전용 generator는 dead code).
//  Gemma 4 E2B는 라이브러리의 happy path이며 mlboydaisuke/gemma-4-E2B-coreml repo에 model_config.json,
//  hf_model/tokenizer.json 등 모든 필수 파일이 존재함. 다운로드 5.4GB로 큰 편이지만 작동 보장됨.
//

import Foundation
import Combine
import CoreML
import CoreMLLLM
import os

@MainActor
final class ModelInstaller: ObservableObject {
    static let shared = ModelInstaller()

    enum State: Equatable {
        case checking
        case ready
        case needsDownload
        case downloading(progress: Double, statusText: String)
        case compiling // 라이브러리의 ANE compile 단계 (first run, can take 1-2 min)
        case failed(message: String)
        case skipped
    }

    @Published private(set) var state: State = .checking

    private let logger = Logger(subsystem: "com.cplanner", category: "ModelInstaller")

    /// 라이브러리에 등록된 Gemma 4 E2B (Gemma TOU). 5.4GB, ANE-friendly (~91% ANE residency).
    private let modelInfo: ModelDownloader.ModelInfo = .gemma4e2b

    /// 사용자 설정의 compute units. 변경 시 reload 필요.
    private(set) var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

    private init() {}

    // MARK: - Public API

    func checkInstallation() {
        if ModelDownloader.shared.localModelURL(for: modelInfo) != nil {
            logger.info("Gemma 4 E2B already downloaded — loading.")
            state = .checking
            Task { await loadModel() }
        } else {
            logger.info("Gemma 4 E2B not present — prompting download.")
            state = .needsDownload
        }
    }

    /// 사용자가 "나중에"를 선택. 같은 세션에선 alert 다시 안 뜸.
    func skip() {
        state = .skipped
    }

    func startDownload() {
        Task { await loadModel() }
    }

    /// 컴퓨트 유닛 변경 → 모델 다시 load. 다운로드는 캐시 hit이므로 빠름, ANE 컴파일은 캐시되어 있어 빠를 수 있음.
    func reloadWithComputeUnits(_ units: MLComputeUnits) {
        guard units != computeUnits else { return }
        computeUnits = units
        logger.info("Compute units → \(units.label, privacy: .public). Reloading.")
        Task {
            await LocalLLMService.shared.detachLLM()
            await loadModel()
        }
    }

    // MARK: - Internals

    private func loadModel() async {
        state = .downloading(progress: 0.0, statusText: "Gemma 4 E2B 준비 중…")
        do {
            let units = computeUnits
            let llm = try await CoreMLLLM.load(
                model: modelInfo,
                computeUnits: units,
                onProgress: Self.makeProgressCallback()
            )
            logger.info("Gemma 4 E2B loaded. ctx=\(llm.contextLength, privacy: .public)")
            await LocalLLMService.shared.attachLLM(llm, computeUnits: units)
            state = .ready
        } catch {
            logger.error("Load failed: \(error.localizedDescription, privacy: .public)")
            state = .failed(message: "모델 로드 실패: \(error.localizedDescription)")
        }
    }

    /// 라이브러리에 넘길 progress callback. 캡처를 최소화해 Sendable 제약을 만족.
    /// 클로저는 라이브러리 내부 background thread에서 호출되므로 MainActor로 hop.
    private static func makeProgressCallback() -> @Sendable (String) -> Void {
        return { message in
            Task { @MainActor in
                ModelInstaller.shared.advanceProgress(message: message)
            }
        }
    }

    /// 라이브러리는 String만 callback으로 보냄 → phase-based progress + statusText.
    private func advanceProgress(message: String) {
        let lower = message.lowercased()
        let progress: Double
        let isCompile = lower.contains("compile") || lower.contains("loading chunks")
        if isCompile {
            state = .compiling
            return
        }
        if lower.contains("downloading") {
            progress = 0.30
        } else if lower.contains("reading config") {
            progress = 0.80
        } else if lower.contains("loading tokenizer") {
            progress = 0.90
        } else {
            // 알 수 없는 메시지 — 현재 progress 유지하면서 statusText만 갱신
            if case .downloading(let p, _) = state {
                state = .downloading(progress: p, statusText: message)
                return
            }
            progress = 0.0
        }
        state = .downloading(progress: progress, statusText: message)
    }
}
