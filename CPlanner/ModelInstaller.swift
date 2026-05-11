//
//  ModelInstaller.swift
//  Cplanner
//
//  사용자가 선택한 모델(`ModelKind`)을 다운로드 + 로드 + LocalLLMService에 attach.
//  Phase 1+2 (2026-05-11): Gemma 4 E2B / E4B 두 가지를 `john-rocky/CoreML-LLM` 라이브러리로 처리.
//  Phase 3 (다음 세션): Mistral 7B 분기 추가 — 라이브러리가 Mistral 미지원이라 raw MLModel
//  + MLState 직접 로드. 캐시 위치도 다름 (~/Library/Application Support/CPlanner/StatefulMistral*).
//
//  State machine 시작:
//    - UserDefaults에 selection 있음 → 해당 모델 캐시 확인 → 있으면 load, 없으면 needsDownload
//    - selection 없음 → `awaitingSelection` (ContentView가 picker sheet 띄움)
//
//  사용자가 picker에서 선택 (또는 설정에서 변경) → `selectModel(_:)`:
//    - UserDefaults 갱신
//    - detach 현재 backend
//    - 새 모델 cache 확인 → 있으면 load, 없으면 다운로드 alert
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
        /// 첫 실행 — UserDefaults에 selection 없음. ContentView가 picker sheet 띄움.
        case awaitingSelection
        case checking
        case ready
        case needsDownload
        case downloading(progress: Double, statusText: String)
        case compiling // 라이브러리의 ANE compile 단계 (first run, can take 1-2 min)
        case failed(message: String)
        case skipped
    }

    @Published private(set) var state: State = .awaitingSelection
    /// 현재 활성 / 선택된 모델 — UI 표시 + load 분기에 사용. nil이면 awaitingSelection.
    @Published private(set) var selectedKind: ModelKind?

    private let logger = Logger(subsystem: "com.cplanner", category: "ModelInstaller")

    /// 사용자 설정의 compute units. 모델별 default가 다름 (Gemma=ANE, Mistral=CPU+GPU).
    private(set) var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

    private init() {}

    // MARK: - Public API

    /// 앱 시작 시 호출 (CplannerApp.init). UserDefaults에 selection 있으면 해당 모델 cache 확인 후 자동 진행,
    /// 없으면 awaitingSelection 상태로 picker UI 트리거.
    func checkInstallation() {
        guard let stored = ModelSelection.current, stored.isAvailable else {
            logger.info("No model selected — awaiting user picker.")
            state = .awaitingSelection
            return
        }
        selectedKind = stored
        computeUnits = stored.defaultComputeUnits
        if let info = Self.modelInfo(for: stored),
           ModelDownloader.shared.localModelURL(for: info) != nil {
            logger.info("\(stored.displayName, privacy: .public) already downloaded — loading.")
            state = .checking
            Task { await loadModel() }
        } else {
            logger.info("\(stored.displayName, privacy: .public) not present — prompting download.")
            state = .needsDownload
        }
    }

    /// 사용자가 picker에서 모델 선택 (first-run 또는 설정에서 변경). 비활성 모델(Phase 3 대기)은 거부.
    func selectModel(_ kind: ModelKind) {
        guard kind.isAvailable else {
            logger.warning("\(kind.displayName, privacy: .public) not yet available — ignored.")
            return
        }
        ModelSelection.current = kind
        let isSwitch = selectedKind != nil && selectedKind != kind
        selectedKind = kind
        computeUnits = kind.defaultComputeUnits

        Task {
            if isSwitch {
                await LocalLLMService.shared.detachBackend()
            }
            // 이미 다운로드돼 있으면 바로 load, 아니면 다운로드 진행.
            if let info = Self.modelInfo(for: kind),
               ModelDownloader.shared.localModelURL(for: info) != nil {
                state = .checking
                await loadModel()
            } else {
                // picker 자체가 download consent라 즉시 다운로드 시작.
                await loadModel()
            }
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
            await LocalLLMService.shared.detachBackend()
            await loadModel()
        }
    }

    // MARK: - Internals

    /// ModelKind를 라이브러리 ModelInfo로 매핑. Mistral은 라이브러리 미지원이라 nil
    /// (Phase 3에서 별도 raw load 경로 추가 예정).
    private static func modelInfo(for kind: ModelKind) -> ModelDownloader.ModelInfo? {
        switch kind {
        case .gemma4E2B: return .gemma4e2b
        case .gemma4E4B: return .gemma4e4b
        case .mistral7B: return nil  // Phase 3
        }
    }

    private func loadModel() async {
        guard let kind = selectedKind else {
            logger.error("loadModel called with no selectedKind.")
            state = .awaitingSelection
            return
        }
        guard let info = Self.modelInfo(for: kind) else {
            logger.error("\(kind.displayName, privacy: .public) load path not implemented yet (Phase 3).")
            state = .failed(message: "\(kind.displayName)은 다음 업데이트에서 추가 예정입니다.")
            return
        }
        state = .downloading(progress: 0.0, statusText: "\(kind.displayName) 준비 중…")
        do {
            let units = computeUnits
            let llm = try await CoreMLLLM.load(
                model: info,
                computeUnits: units,
                onProgress: Self.makeProgressCallback()
            )
            logger.info("\(kind.displayName, privacy: .public) loaded. ctx=\(llm.contextLength, privacy: .public)")
            let backend = GemmaBackend(kind: kind, llm: llm)
            await LocalLLMService.shared.attachBackend(backend, computeUnits: units)
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
