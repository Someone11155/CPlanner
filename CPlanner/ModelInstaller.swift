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
@preconcurrency import Hub
import Tokenizers
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
        if cacheReady(for: stored) {
            logger.info("\(stored.displayName, privacy: .public) already downloaded — loading.")
            state = .checking
            Task { await loadModel() }
        } else {
            logger.info("\(stored.displayName, privacy: .public) not present — prompting download.")
            state = .needsDownload
        }
    }

    /// 모델별 캐시 존재 확인.
    /// - Gemma E2B: 라이브러리(`john-rocky/CoreML-LLM`)의 `ModelDownloader.localModelURL(for:)`.
    /// - Gemma E4B: prefill chunk 포함 풀번들(`<App Support>/CPlanner/gemma-4-e4b/`)이 다 있을 때만 ready.
    ///   미설치면 `downloadGemma4E4BBundle()`이 HF `someone15/gemma-4-E4B-coreml`에서 받아옴.
    ///   (라이브러리 download path는 prefill 부재 → 17× 느림이라 사용 안 함.)
    /// - Mistral: `applicationSupportDirectory`에 토크나이저 + 컴파일된 `.mlmodelc`.
    private func cacheReady(for kind: ModelKind) -> Bool {
        switch kind {
        case .gemma4E2B:
            guard let info = Self.modelInfo(for: kind) else { return false }
            return ModelDownloader.shared.localModelURL(for: info) != nil
        case .gemma4E4B:
            return Self.gemma4E4BLocalBundleReady()
        case .mistral7B:
            return Self.mistralResolveLocation() != nil
        }
    }

    /// prefill chunk 포함 풀번들이 올라간 HF repo. mlboydaisuke repo와 달리 prefill chunks 포함.
    nonisolated private static let gemma4E4BRepo = "someone15/gemma-4-E4B-coreml"

    /// prefill chunk 포함 Gemma 4 E4B 풀번들 위치 (decode + prefill chunks + embeddings + sidecars + tokenizer).
    /// 첫 실행 시 `downloadGemma4E4BBundle()`이 HF `someone15/gemma-4-E4B-coreml`에서 여기로 다운로드.
    nonisolated private static func gemma4E4BLocalBundleURL() -> URL? {
        guard let appSupport = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: false
        ) else { return nil }
        return appSupport
            .appendingPathComponent("CPlanner", isDirectory: true)
            .appendingPathComponent("gemma-4-e4b", isDirectory: true)
    }

    /// 자체 번들의 필수 파일 존재 여부 검증. chunks + prefill chunks + tokenizer + config + 핵심 sidecar.
    nonisolated private static func gemma4E4BLocalBundleReady() -> Bool {
        guard let dir = gemma4E4BLocalBundleURL() else { return false }
        let fm = FileManager.default
        let required: [String] = [
            "chunk1.mlmodelc", "chunk2.mlmodelc", "chunk3.mlmodelc", "chunk4.mlmodelc",
            "prefill_chunk1.mlmodelc", "prefill_chunk2.mlmodelc",
            "prefill_chunk3.mlmodelc", "prefill_chunk4.mlmodelc",
            "model_config.json", "hf_model/tokenizer.json",
            "embed_tokens_q8.bin", "embed_tokens_scales.bin",
            "cos_sliding.npy", "sin_sliding.npy"
        ]
        return required.allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    /// 진단용 — 어떤 파일이 누락됐는지 출력.
    nonisolated private static func dumpMissingBundleFiles(at dir: URL) {
        let fm = FileManager.default
        let required: [String] = [
            "chunk1.mlmodelc", "chunk2.mlmodelc", "chunk3.mlmodelc", "chunk4.mlmodelc",
            "prefill_chunk1.mlmodelc", "prefill_chunk2.mlmodelc",
            "prefill_chunk3.mlmodelc", "prefill_chunk4.mlmodelc",
            "model_config.json", "hf_model/tokenizer.json",
            "embed_tokens_q8.bin", "embed_tokens_scales.bin",
            "cos_sliding.npy", "sin_sliding.npy"
        ]
        for f in required {
            let p = dir.appendingPathComponent(f).path
            let ok = fm.fileExists(atPath: p)
            print("[ModelInstaller]   \(ok ? "✓" : "✗") \(f)  (\(p))")
        }
    }

    /// 사용자가 picker에서 모델 선택 (first-run 또는 설정에서 변경).
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
            // 이미 다운로드돼 있으면 바로 load, 아니면 picker 자체가 download consent.
            if cacheReady(for: kind) {
                state = .checking
            }
            await loadModel()
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
        switch kind {
        case .gemma4E2B, .gemma4E4B:
            await loadGemmaBackend(kind: kind)
        case .mistral7B:
            await loadMistralBackend()
        }
    }

    private func loadGemmaBackend(kind: ModelKind) async {
        // E4B는 prefill chunk 포함 풀번들을 HF에서 받아 local path로 직접 load.
        // 라이브러리 download path는 prefill이 없어 17× 느리므로 사용하지 않는다.
        if kind == .gemma4E4B {
            if !Self.gemma4E4BLocalBundleReady() {
                logger.info("E4B 번들 미설치 — HF \(Self.gemma4E4BRepo, privacy: .public)에서 다운로드.")
                do {
                    try await downloadGemma4E4BBundle()
                } catch {
                    logger.error("E4B 번들 다운로드 실패: \(error.localizedDescription, privacy: .public)")
                    state = .failed(message: "Gemma 4 E4B 다운로드 실패: \(error.localizedDescription)")
                    return
                }
            }
            guard let bundleURL = Self.gemma4E4BLocalBundleURL(), Self.gemma4E4BLocalBundleReady() else {
                if let url = Self.gemma4E4BLocalBundleURL() { Self.dumpMissingBundleFiles(at: url) }
                state = .failed(message: "Gemma 4 E4B 번들 검증 실패 — 일부 파일 누락.")
                return
            }
            await loadGemmaFromLocalBundle(kind: kind, directory: bundleURL)
            return
        }
        guard let info = Self.modelInfo(for: kind) else {
            logger.error("\(kind.displayName, privacy: .public) ModelInfo 매핑 누락.")
            state = .failed(message: "\(kind.displayName) 매핑이 없습니다.")
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

    /// 자체 변환 Gemma 4 E4B 번들을 라이브러리의 `CoreMLLLM.load(from:)`으로 직접 load.
    /// 라이브러리는 디렉터리에 chunk1.mlmodelc가 있으면 chunked SWA 경로로 자동 분기.
    private func loadGemmaFromLocalBundle(kind: ModelKind, directory: URL) async {
        logger.info("loadGemmaFromLocalBundle dir=\(directory.path, privacy: .public)")
        state = .compiling
        let units = computeUnits
        let cb = Self.makeProgressCallback()
        cb("Gemma 4 E4B 로컬 번들 로드 중…")
        do {
            let llm = try await CoreMLLLM.load(
                from: directory,
                computeUnits: units,
                onProgress: cb
            )
            logger.info("\(kind.displayName, privacy: .public) (local) loaded. ctx=\(llm.contextLength, privacy: .public)")
            let backend = GemmaBackend(kind: kind, llm: llm)
            await LocalLLMService.shared.attachBackend(backend, computeUnits: units)
            state = .ready
        } catch {
            logger.error("Local bundle load failed: \(error.localizedDescription, privacy: .public)")
            state = .failed(message: "로컬 E4B 번들 로드 실패: \(error.localizedDescription)")
        }
    }

    /// HF `someone15/gemma-4-E4B-coreml`에서 prefill chunk 포함 풀번들을 받아 local bundle 경로로 배치.
    /// `matching: []` → repo 전체 파일 다운로드 (HubApi.getFilenames: 빈 globs면 전 파일 반환).
    /// `<hubBase>/models/<repo>/`에 받은 뒤 번들 경로로 moveItem.
    private func downloadGemma4E4BBundle() async throws {
        guard let bundleURL = Self.gemma4E4BLocalBundleURL() else {
            throw NSError(domain: "ModelInstaller", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "E4B 번들 경로를 만들 수 없습니다."])
        }
        state = .downloading(progress: 0.0, statusText: "Gemma 4 E4B 다운로드 준비 중…")

        let parent = bundleURL.deletingLastPathComponent()  // <App Support>/CPlanner/
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let hubBase = parent.appendingPathComponent("hub-e4b", isDirectory: true)
        try? FileManager.default.removeItem(at: hubBase)
        try FileManager.default.createDirectory(at: hubBase, withIntermediateDirectories: true)

        let api = HubApi(downloadBase: hubBase)
        let repo = Hub.Repo(id: Self.gemma4E4BRepo)
        let snapshotURL = try await api.snapshot(from: repo, matching: []) { @Sendable progress in
            let frac = progress.fractionCompleted
            let stat = "Gemma 4 E4B 다운로드 중 — \(Int(frac * 100))%"
            Task { @MainActor in
                ModelInstaller.shared.state = .downloading(progress: frac, statusText: stat)
            }
        }

        // 받은 스냅샷(repo 루트 = 번들 구조)을 번들 경로로 이동.
        if FileManager.default.fileExists(atPath: bundleURL.path) {
            try FileManager.default.removeItem(at: bundleURL)
        }
        try FileManager.default.moveItem(at: snapshotURL, to: bundleURL)
        try? FileManager.default.removeItem(at: hubBase)
        logger.info("E4B bundle downloaded → \(bundleURL.path, privacy: .public)")
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

    // MARK: - Mistral install / load (라이브러리 미지원이라 별도 경로)
    // 클래스 자체는 @MainActor라 상수도 기본 isolated → resolveLocation 등 nonisolated 컨텍스트 + Sendable 클로저에서 접근하려면 nonisolated 명시.

    nonisolated private static let mistralModelRepo = "apple/mistral-coreml"
    nonisolated private static let mistralTokenizerRepo = "mistralai/Mistral-7B-Instruct-v0.3"
    nonisolated private static let mistralPackageName = "StatefulMistral7BInstructInt4.mlpackage"
    nonisolated private static let mistralCompiledName = "StatefulMistral7BInstructInt4.mlmodelc"
    nonisolated private static let mistralTokenizerFiles = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.model.v3",
        "tokenizer_config.json"
    ]
    /// 모델이 다운로드 데이터의 ~99% — 진행 바를 데이터 비율에 맞춰 가중.
    nonisolated private static let mistralModelWeight: Double = 0.95
    nonisolated private static let mistralTokenizerWeight: Double = 0.05

    /// `~/Library/Application Support/CPlanner/`. 옛 commit `8267940`의 위치와 동일.
    nonisolated static var mistralAppSupportDirectory: URL {
        let fm = FileManager.default
        let base: URL = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let dir = base.appendingPathComponent("CPlanner", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// 토크나이저 + 컴파일된 .mlmodelc가 동시에 있는 디렉터리 반환. 번들 우선, 없으면 App Support.
    nonisolated static func mistralResolveLocation() -> URL? {
        let fm = FileManager.default
        let tokenizerOK: (URL) -> Bool = { dir in
            ["tokenizer.json", "tokenizer_config.json"].allSatisfy {
                fm.fileExists(atPath: dir.appendingPathComponent($0).path)
            }
        }
        let compiledOK: (URL) -> Bool = { dir in
            var isDir: ObjCBool = false
            let url = dir.appendingPathComponent(mistralCompiledName)
            return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }
        if let bundle = Bundle.main.resourceURL, tokenizerOK(bundle), compiledOK(bundle) {
            return bundle
        }
        let appSupport = mistralAppSupportDirectory
        if tokenizerOK(appSupport), compiledOK(appSupport) {
            return appSupport
        }
        return nil
    }

    /// 압축 패키지(.mlpackage)는 받았으나 아직 .mlmodelc로 컴파일 안 된 상태.
    private nonisolated static func mistralHasUnpackedPackage() -> Bool {
        let fm = FileManager.default
        let dir = mistralAppSupportDirectory
        let manifest = dir.appendingPathComponent(mistralPackageName).appendingPathComponent("Manifest.json")
        let tokenizerOK = ["tokenizer.json", "tokenizer_config.json"].allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
        return fm.fileExists(atPath: manifest.path) && tokenizerOK
    }

    /// Mistral 백엔드 전체 경로 — 캐시 hit 시 즉시 로드, 미설치 시 다운로드 → 컴파일 → 로드.
    private func loadMistralBackend() async {
        // 이미 캐시 있으면 다운로드/컴파일 스킵.
        if let location = Self.mistralResolveLocation() {
            await mistralAttach(from: location)
            return
        }
        // .mlpackage 받아놨지만 컴파일 안 된 케이스 (이전 세션 중단).
        if Self.mistralHasUnpackedPackage() {
            state = .compiling
            do {
                try await mistralCompilePackage()
            } catch {
                logger.error("Mistral compile failed: \(error.localizedDescription, privacy: .public)")
                state = .failed(message: "Mistral 컴파일 실패: \(error.localizedDescription)")
                return
            }
            if let location = Self.mistralResolveLocation() {
                await mistralAttach(from: location)
            } else {
                state = .failed(message: "컴파일 후에도 Mistral 파일 검증 실패")
            }
            return
        }

        // 완전 신규 다운로드.
        state = .downloading(progress: 0.0, statusText: "Mistral 7B 모델 준비 중…")
        do {
            try await mistralDownloadModel()
            try await mistralDownloadTokenizer()
        } catch {
            let msg: String
            if let hubError = error as? Hub.HubClientError, case .authorizationRequired = hubError {
                msg = "모델 저장소 접근 권한 거부 — HuggingFace 토큰이 필요할 수 있습니다."
            } else {
                msg = "Mistral 다운로드 실패: \(error.localizedDescription)"
            }
            logger.error("Mistral download failed: \(error.localizedDescription, privacy: .public)")
            state = .failed(message: msg)
            return
        }

        state = .compiling
        do {
            try await mistralCompilePackage()
        } catch {
            logger.error("Mistral compile failed: \(error.localizedDescription, privacy: .public)")
            state = .failed(message: "Mistral 컴파일 실패: \(error.localizedDescription)")
            return
        }
        guard let location = Self.mistralResolveLocation() else {
            state = .failed(message: "컴파일 후에도 Mistral 파일 검증 실패")
            return
        }
        await mistralAttach(from: location)
    }

    private func mistralAttach(from location: URL) async {
        do {
            let tokenizer = try await AutoTokenizer.from(modelFolder: location)
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            let modelURL = location.appendingPathComponent(Self.mistralCompiledName)
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let backend = MistralBackend(tokenizer: tokenizer, model: model)
            await LocalLLMService.shared.attachBackend(backend, computeUnits: computeUnits)
            logger.info("Mistral 7B loaded from \(location.path, privacy: .public). units=\(self.computeUnits.label, privacy: .public)")
            state = .ready
        } catch {
            logger.error("Mistral attach failed: \(error.localizedDescription, privacy: .public)")
            state = .failed(message: "Mistral 로드 실패: \(error.localizedDescription)")
        }
    }

    private func mistralDownloadModel() async throws {
        let appSupport = Self.mistralAppSupportDirectory
        let hubBase = appSupport.appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hubBase, withIntermediateDirectories: true)

        let api = HubApi(downloadBase: hubBase)
        let repo = Hub.Repo(id: Self.mistralModelRepo)
        let glob = "\(Self.mistralPackageName)/*"

        let snapshotURL = try await api.snapshot(from: repo, matching: [glob]) { @Sendable progress in
            let frac = progress.fractionCompleted
            let stat = "Mistral 모델 다운로드 중 — \(Int(frac * 100))%"
            let overall = frac * Self.mistralModelWeight
            Task { @MainActor in
                ModelInstaller.shared.state = .downloading(progress: overall, statusText: stat)
            }
        }

        // Hub은 <hubBase>/models/<repo.id>/ 안에 받음. .mlpackage를 AppSupport 루트로 이동.
        let downloadedPackage = snapshotURL.appendingPathComponent(Self.mistralPackageName)
        let destPackage = appSupport.appendingPathComponent(Self.mistralPackageName)
        if FileManager.default.fileExists(atPath: destPackage.path) {
            try FileManager.default.removeItem(at: destPackage)
        }
        try FileManager.default.moveItem(at: downloadedPackage, to: destPackage)
        try? FileManager.default.removeItem(at: hubBase)
    }

    private func mistralDownloadTokenizer() async throws {
        let dest = Self.mistralAppSupportDirectory
        let total = Self.mistralTokenizerFiles.count
        let perWeight = Self.mistralTokenizerWeight / Double(total)
        for (i, name) in Self.mistralTokenizerFiles.enumerated() {
            let baseProgress = Self.mistralModelWeight + Double(i) * perWeight
            state = .downloading(progress: baseProgress, statusText: "토크나이저 (\(i + 1)/\(total)): \(name)")

            let url = URL(string: "https://huggingface.co/\(Self.mistralTokenizerRepo)/resolve/main/\(name)")!
            let destURL = dest.appendingPathComponent(name)
            try await mistralDownloadFile(url: url, destination: destURL) { written, expected in
                let perFile: Double = expected > 0 ? Double(written) / Double(expected) : 0
                let overall = baseProgress + perFile * perWeight
                let kb = Double(written) / 1024
                let stat: String
                if expected > 0 {
                    let totalKB = Double(expected) / 1024
                    stat = "토크나이저 (\(i + 1)/\(total)): \(name) — \(Int(kb))/\(Int(totalKB)) KB"
                } else {
                    stat = "토크나이저 (\(i + 1)/\(total)): \(name) — \(Int(kb)) KB"
                }
                Task { @MainActor [weak self] in
                    self?.state = .downloading(progress: overall, statusText: stat)
                }
            }
        }
    }

    private func mistralCompilePackage() async throws {
        let appSupport = Self.mistralAppSupportDirectory
        let packageURL = appSupport.appendingPathComponent(Self.mistralPackageName)
        let tempCompiled = try await MLModel.compileModel(at: packageURL)
        let stableURL = appSupport.appendingPathComponent(Self.mistralCompiledName)
        if FileManager.default.fileExists(atPath: stableURL.path) {
            try FileManager.default.removeItem(at: stableURL)
        }
        try FileManager.default.moveItem(at: tempCompiled, to: stableURL)
    }

    private func mistralDownloadFile(url: URL,
                                     destination: URL,
                                     progress: @escaping (Int64, Int64) -> Void) async throws {
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        let delegate = HFDownloadDelegate(progressHandler: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        let temp: URL
        do {
            temp = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<URL, Error>) in
                delegate.continuation = cont
                session.downloadTask(with: url).resume()
            }
        } catch {
            session.invalidateAndCancel()
            throw error
        }
        session.finishTasksAndInvalidate()
        try FileManager.default.moveItem(at: temp, to: destination)
    }
}

private final class HFDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    var continuation: CheckedContinuation<URL, Error>?
    private let progressHandler: (Int64, Int64) -> Void

    init(progressHandler: @escaping (Int64, Int64) -> Void) {
        self.progressHandler = progressHandler
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        progressHandler(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        if let http = downloadTask.response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let body = (try? String(contentsOf: location, encoding: .utf8)) ?? ""
            try? FileManager.default.removeItem(at: location)
            let snippet = body.prefix(200)
            let msg = "HTTP \(http.statusCode)" + (snippet.isEmpty ? "" : " — \(snippet)")
            continuation?.resume(throwing: NSError(domain: "ModelInstaller",
                                                   code: http.statusCode,
                                                   userInfo: [NSLocalizedDescriptionKey: msg]))
            continuation = nil
            return
        }
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: temp)
            continuation?.resume(returning: temp)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error, let cont = continuation {
            cont.resume(throwing: error)
            continuation = nil
        }
    }
}
