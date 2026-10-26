//
//  ModelInstaller.swift
//  Cplanner
//
//  사용자가 선택한 모델(`ModelKind`)을 다운로드 + 로드 + LocalLLMService에 attach.
//  현재는 Gemma 4 E4B 단독 운영 (prefill chunk 포함 풀번들). 모델 선택 인프라(`ModelKind`
//  allCases / `selectModel` / picker UI)는 향후 다중 모델 복원을 위해 그대로 유지.
//
//  State machine 시작 (단일 모델):
//    - selection 있음 → 해당 모델 캐시 확인 → 있으면 load, 없으면 needsDownload
//    - selection 없음(첫 실행) → `defaultModel`(E4B) 자동 선택 → 캐시 확인 → needsDownload(=다운로드 alert)
//      (picker는 띄우지 않음 — `awaitingSelection`/`ModelPickerSheet`는 코드만 dormant 유지)
//
//  설정에서 모델 변경 시 → `selectModel(_:)`:
//    - UserDefaults 갱신 → detach 현재 backend → 새 모델 cache 확인 → load 또는 다운로드 alert
//

import Foundation
import Combine
import CoreML
import CoreMLLLM
import os

/// HF tree API 응답의 파일 엔트리. (`SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`라
/// JSONDecoder가 nonisolated 컨텍스트에서 init(from:)을 호출할 수 있도록 `nonisolated` 명시.)
nonisolated private struct RepoFile: Decodable {
    let path: String
    let size: Int64
    let type: String
}

/// `downloadOnce`의 URLSession data delegate. 수신 데이터를 dest에 append(이어받기) 하거나,
/// 서버가 Range를 무시(200)하면 처음부터 다시 쓰고, 완료/실패를 continuation으로 돌린다.
/// 콜백이 URLSession background queue(직렬)에서만 호출되므로 추가 락 없이 안전하다.
/// 빌드 설정 `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` 대응으로 `nonisolated` + `@unchecked Sendable`.
nonisolated private final class BundleFileDownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    var continuation: CheckedContinuation<Void, Error>?
    private let dest: URL
    private let resumeFrom: Int64
    private let expectedSize: Int64
    private let progress: @Sendable (Int64) -> Void
    private var handle: FileHandle?
    private var written: Int64 = 0
    private var lastReported: Int64 = 0
    private var finished = false

    init(dest: URL, resumeFrom: Int64, expectedSize: Int64,
         progress: @escaping @Sendable (Int64) -> Void) {
        self.dest = dest
        self.resumeFrom = resumeFrom
        self.expectedSize = expectedSize
        self.progress = progress
    }

    private func err(_ msg: String, _ code: Int = -8) -> Error {
        NSError(domain: "ModelInstaller", code: code, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    private func finish(_ result: Result<Void, Error>) {
        guard !finished else { return }
        finished = true
        try? handle?.close()
        handle = nil
        switch result {
        case .success: continuation?.resume(returning: ())
        case .failure(let e): continuation?.resume(throwing: e)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel); finish(.failure(err("응답 형식 오류"))); return
        }
        let fm = FileManager.default
        do {
            if resumeFrom > 0 && http.statusCode == 206 {
                // 부분 이어받기 — 기존 파일 끝에 append.
                handle = try FileHandle(forWritingTo: dest)
                try handle?.seekToEnd()
                written = resumeFrom
            } else if http.statusCode == 416 {
                // Range 불가(부분 파일 손상/크기 불일치) → 지우고 다음 시도에서 처음부터.
                try? fm.removeItem(at: dest)
                completionHandler(.cancel); finish(.failure(err("Range 불가(416) — 재시작", -9))); return
            } else if (200..<300).contains(http.statusCode) {
                // 전체 응답(200, 서버가 Range 무시) 또는 신규 — truncate 후 0부터.
                if !fm.fileExists(atPath: dest.path) {
                    fm.createFile(atPath: dest.path, contents: nil)
                }
                handle = try FileHandle(forWritingTo: dest)
                try handle?.truncate(atOffset: 0)
                written = 0
            } else {
                completionHandler(.cancel)
                finish(.failure(err("HTTP \(http.statusCode)", http.statusCode))); return
            }
            completionHandler(.allow)
        } catch {
            completionHandler(.cancel); finish(.failure(error))
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let handle, !finished else { return }
        do {
            try handle.write(contentsOf: data)
            written += Int64(data.count)
            // ~8MB마다만 보고 — MainActor 상태 갱신 폭주 방지.
            if written - lastReported >= 8_000_000 || written >= expectedSize {
                lastReported = written
                progress(written)
            }
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish(.failure(error)) } else { finish(.success(())) }
    }
}

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

    /// 사용자 설정의 compute units. Gemma E4B는 ANE default.
    private(set) var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

    private init() {}

    // MARK: - Public API

    /// 단일 모델 운영 기본값. 첫 실행/옛 모델 제거 시 자동 선택되며, 다중 모델 복원 시
    /// picker 기본 선택값으로도 재활용할 수 있다.
    nonisolated static let defaultModel: ModelKind = .gemma4E4B

    /// 앱 시작 시 호출 (CplannerApp.init). 저장된 selection이 없거나(첫 실행) 옛 모델이 제거된
    /// 경우 `defaultModel`(E4B)을 자동 선택해 picker 없이 바로 다운로드 동의(`needsDownload`)로 진입한다.
    /// 모델 선택 UI(`awaitingSelection` + `ModelPickerSheet`)는 향후 다중 모델 복원을 위해 코드로만 남겨둠.
    func checkInstallation() {
        let stored = ModelSelection.current ?? Self.defaultModel
        ModelSelection.current = stored  // 첫 실행/마이그레이션 시 영속화
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
    /// - Gemma E4B: prefill chunk 포함 풀번들(`<App Support>/CPlanner/gemma-4-e4b/`)이 다 있을 때만 ready.
    ///   미설치면 `downloadGemma4E4BBundle()`이 HF `someone15/gemma-4-E4B-coreml`에서 받아옴.
    ///   (라이브러리 download path는 prefill 부재 → 17× 느림이라 사용 안 함.)
    private func cacheReady(for kind: ModelKind) -> Bool {
        switch kind {
        case .gemma4E4B:
            return Self.gemma4E4BLocalBundleReady()
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

    /// 풀번들 무결성 검증에 필요한 핵심 파일들. chunks + prefill chunks + tokenizer + config + 핵심 sidecar.
    nonisolated private static let gemma4E4BRequiredFiles: [String] = [
        "chunk1.mlmodelc", "chunk2.mlmodelc", "chunk3.mlmodelc", "chunk4.mlmodelc",
        "prefill_chunk1.mlmodelc", "prefill_chunk2.mlmodelc",
        "prefill_chunk3.mlmodelc", "prefill_chunk4.mlmodelc",
        "model_config.json", "hf_model/tokenizer.json",
        "embed_tokens_q8.bin", "embed_tokens_scales.bin",
        "cos_sliding.npy", "sin_sliding.npy"
    ]

    /// 자체 번들의 필수 파일 존재 여부 검증.
    nonisolated private static func gemma4E4BLocalBundleReady() -> Bool {
        guard let dir = gemma4E4BLocalBundleURL() else { return false }
        let fm = FileManager.default
        return gemma4E4BRequiredFiles.allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    /// 진단용 — 어떤 파일이 누락됐는지 출력.
    nonisolated private static func dumpMissingBundleFiles(at dir: URL) {
        let fm = FileManager.default
        for f in gemma4E4BRequiredFiles {
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

    /// ModelKind를 라이브러리 ModelInfo로 매핑. (E4B는 실제론 풀번들 경로로 로드하므로 이 매핑은
    /// 라이브러리 download path를 쓰는 향후 모델용 — `loadGemmaBackend`의 generic 분기에서 사용.)
    private static func modelInfo(for kind: ModelKind) -> ModelDownloader.ModelInfo? {
        switch kind {
        case .gemma4E4B: return .gemma4e4b
        }
    }

    private func loadModel() async {
        guard let kind = selectedKind else {
            logger.error("loadModel called with no selectedKind.")
            state = .awaitingSelection
            return
        }
        switch kind {
        case .gemma4E4B:
            await loadGemmaBackend(kind: kind)
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
    ///
    /// 과거엔 swift-transformers의 단일 `HubApi.snapshot(matching: [])` 호출을 정체 감지 watchdog로
    /// 감쌌지만, 대용량 파일(prefill 2.3GB·embed 2.8GB) 전송 중 `nw_connection cancelled`로 hang하면
    /// Swift Task의 협조적 cancel이 진행 중인 URLSession 전송까지 전파되지 않아 `snapshotTask.value`가
    /// 영원히 반환되지 않았다(= 5분이 지나도 회복 못 함, 실측). 그래서 파일 단위로 URLSession을 직접
    /// 제어한다: idle 60s(`timeoutIntervalForRequest`)면 OS가 연결을 끊어 hang을 확실히 회복하고,
    /// 부분 파일은 `Range`로 이어받으며, 이미 받은 파일(크기 일치)은 skip한다.
    /// staging(`gemma-4-e4b.download/`)은 launch마다 지우지 않아 재실행 시에도 resume된다.
    private func downloadGemma4E4BBundle() async throws {
        guard let bundleURL = Self.gemma4E4BLocalBundleURL() else {
            throw NSError(domain: "ModelInstaller", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "E4B 번들 경로를 만들 수 없습니다."])
        }
        state = .downloading(progress: 0.0, statusText: "Gemma 4 E4B 파일 목록 확인 중…")

        let fm = FileManager.default
        let parent = bundleURL.deletingLastPathComponent()  // <App Support>/CPlanner/
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent("gemma-4-e4b.download", isDirectory: true)

        // 구버전 snapshot staging(`hub-e4b/models/<repo>/`)이 남아 있으면 그 파일들을 재활용해
        // 이미 받은 decode chunk(~2.2GB) 재다운로드를 피한다(크기 검증 후 skip되므로 안전). 그 뒤 정리.
        let legacy = parent.appendingPathComponent("hub-e4b", isDirectory: true)
            .appendingPathComponent("models/\(Self.gemma4E4BRepo)", isDirectory: true)
        if !fm.fileExists(atPath: staging.path), fm.fileExists(atPath: legacy.path) {
            try? fm.moveItem(at: legacy, to: staging)
        }
        try? fm.removeItem(at: parent.appendingPathComponent("hub-e4b", isDirectory: true))
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)

        let files = try await Self.fetchRepoManifest()
        guard !files.isEmpty else {
            throw NSError(domain: "ModelInstaller", code: -3,
                          userInfo: [NSLocalizedDescriptionKey: "repo 파일 목록이 비어 있습니다."])
        }
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }

        // 이미 완전히 받아둔 파일(크기 일치)의 바이트를 진행률 기준선으로.
        func bytesPresent(_ f: RepoFile) -> Int64 {
            let sz = (try? fm.attributesOfItem(atPath: staging.appendingPathComponent(f.path).path)[.size] as? Int64) ?? nil
            return (sz == f.size) ? f.size : 0
        }
        var completedBytes = files.reduce(Int64(0)) { $0 + bytesPresent($1) }

        for (idx, f) in files.enumerated() {
            let dest = staging.appendingPathComponent(f.path)
            if bytesPresent(f) == f.size { continue }  // 이미 완료 — skip

            let base = completedBytes
            let position = idx + 1, count = files.count
            try await downloadFileWithResume(path: f.path, expectedSize: f.size, to: dest) { writtenInFile in
                let overall = totalBytes > 0 ? Double(base + writtenInFile) / Double(totalBytes) : 0
                let stat = "Gemma 4 E4B 다운로드 중 — \(Int(overall * 100))% (\(position)/\(count))"
                Task { @MainActor in
                    ModelInstaller.shared.state = .downloading(progress: overall, statusText: stat)
                }
            }
            completedBytes += f.size
        }

        // 모든 파일 확보 → 번들 경로로 이동.
        if fm.fileExists(atPath: bundleURL.path) {
            try fm.removeItem(at: bundleURL)
        }
        try fm.moveItem(at: staging, to: bundleURL)
        logger.info("E4B bundle downloaded → \(bundleURL.path, privacy: .public)")
    }

    /// HF tree API로 repo의 전체 파일 경로+크기 매니페스트를 받는다(`?recursive=1`).
    nonisolated private static func fetchRepoManifest() async throws -> [RepoFile] {
        let urlStr = "https://huggingface.co/api/models/\(gemma4E4BRepo)/tree/main?recursive=1"
        guard let url = URL(string: urlStr) else {
            throw NSError(domain: "ModelInstaller", code: -4,
                          userInfo: [NSLocalizedDescriptionKey: "매니페스트 URL 생성 실패."])
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 30
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
        guard (200..<300).contains(code) else {
            throw NSError(domain: "ModelInstaller", code: code,
                          userInfo: [NSLocalizedDescriptionKey: "파일 목록 조회 실패 (HTTP \(code))."])
        }
        let entries = try JSONDecoder().decode([RepoFile].self, from: data)
        return entries.filter { $0.type == "file" && $0.size > 0 }
    }

    /// 파일 하나를 idle-timeout + Range 재개로 받는다. 실패 시 부분 파일을 이어받아 최대 5회 재시도.
    private func downloadFileWithResume(path: String,
                                        expectedSize: Int64,
                                        to dest: URL,
                                        progress: @escaping @Sendable (Int64) -> Void) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let maxAttempts = 5
        var lastError: Error?

        for attempt in 1...maxAttempts {
            let have = (try? fm.attributesOfItem(atPath: dest.path)[.size] as? Int64) ?? 0
            if have == expectedSize { return }                       // 이미 완료
            if have > expectedSize { try? fm.removeItem(at: dest) }  // 손상 → 처음부터

            do {
                try await Self.downloadOnce(repo: Self.gemma4E4BRepo, path: path,
                                            dest: dest, expectedSize: expectedSize, progress: progress)
                let final = (try? fm.attributesOfItem(atPath: dest.path)[.size] as? Int64) ?? 0
                if final == expectedSize { return }
                lastError = NSError(domain: "ModelInstaller", code: -5,
                                    userInfo: [NSLocalizedDescriptionKey: "\(path) 크기 불일치 (\(final)/\(expectedSize))"])
            } catch {
                lastError = error
            }
            if attempt < maxAttempts {
                logger.warning("E4B 파일 \(path, privacy: .public) 실패 \(attempt, privacy: .public)/\(maxAttempts, privacy: .public): \(lastError?.localizedDescription ?? "?", privacy: .public) — 이어받기 재시도")
                try? await Task.sleep(nanoseconds: 2 * 1_000_000_000)
            }
        }
        throw lastError ?? NSError(domain: "ModelInstaller", code: -6,
                                   userInfo: [NSLocalizedDescriptionKey: "\(path) 다운로드 재시도 소진"])
    }

    /// 단일 파일 1회 다운로드. 부분 파일이 있으면 `Range`로 이어받고, idle 60s 무수신이면
    /// URLSession이 연결을 끊어(`timeoutIntervalForRequest`) hang을 OS 레벨에서 확실히 회복한다
    /// — 라이브러리 snapshot의 협조적 Task cancel이 전파 안 되던 문제의 근본 해결.
    nonisolated private static func downloadOnce(repo: String,
                                                 path: String,
                                                 dest: URL,
                                                 expectedSize: Int64,
                                                 progress: @escaping @Sendable (Int64) -> Void) async throws {
        let encoded = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
        guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(encoded)") else {
            throw NSError(domain: "ModelInstaller", code: -7,
                          userInfo: [NSLocalizedDescriptionKey: "파일 URL 생성 실패: \(path)"])
        }
        let have = (try? FileManager.default.attributesOfItem(atPath: dest.path)[.size] as? Int64) ?? 0
        var req = URLRequest(url: url)
        if have > 0 { req.setValue("bytes=\(have)-", forHTTPHeaderField: "Range") }

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60      // idle(무수신) 60s → OS가 끊음. watchdog 대체.
        config.timeoutIntervalForResource = 7200   // 대용량 단일 파일(최대 2.8GB) 허용.
        config.waitsForConnectivity = false
        config.urlCache = nil                       // 8GB가 URL 캐시에 중복 저장되지 않도록.

        let delegate = BundleFileDownloadDelegate(dest: dest, resumeFrom: have,
                                                  expectedSize: expectedSize, progress: progress)
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                delegate.continuation = cont
                session.dataTask(with: req).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
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
