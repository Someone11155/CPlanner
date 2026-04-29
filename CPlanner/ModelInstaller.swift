//
//  ModelInstaller.swift
//  Cplanner
//
//  앱 시작 시 Mistral 모델/토크나이저 설치 여부를 판단하고,
//  필요 시 HuggingFace에서 다운로드한다. 번들에 모델이 있으면 그쪽을
//  우선 사용하고, 없으면 ~/Library/Application Support/CPlanner 로 받는다.
//

import Foundation
import Combine
import CoreML
@preconcurrency import Hub
import os

@MainActor
final class ModelInstaller: ObservableObject {
    static let shared = ModelInstaller()

    enum State: Equatable {
        case checking
        case ready
        case needsDownload
        case downloading(progress: Double, statusText: String)
        case compiling // CoreML이 .mlpackage를 .mlmodelc로 컴파일 중
        case failed(message: String)
        case skipped // 사용자가 "나중에"를 누른 경우, 같은 세션에선 더 이상 묻지 않음
    }

    @Published private(set) var state: State = .checking

    private let logger = Logger(subsystem: "com.cplanner", category: "ModelInstaller")

    private let modelRepo = "apple/mistral-coreml"
    private let tokenizerRepo = "mistralai/Mistral-7B-Instruct-v0.3"
    private let modelDirName = "StatefulMistral7BInstructInt4.mlpackage"

    private let tokenizerFiles = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.model.v3",
        "tokenizer_config.json",
    ]

    private init() {}

    // MARK: - Resolution

    /// 토크나이저 + 컴파일된 .mlmodelc가 있는 디렉터리를 반환. 번들 우선, 없으면 App Support.
    nonisolated func resolveModelLocation() -> URL? {
        if let bundle = Bundle.main.resourceURL,
           hasTokenizer(in: bundle), hasCompiledModel(in: bundle) {
            return bundle
        }
        let appSupport = applicationSupportDirectory
        if hasTokenizer(in: appSupport), hasCompiledModel(in: appSupport) {
            return appSupport
        }
        return nil
    }

    /// .mlpackage는 받았지만 아직 .mlmodelc로 컴파일되지 않은 상태인지.
    private nonisolated func hasUnpackedModelInAppSupport() -> Bool {
        let dir = applicationSupportDirectory
        return hasTokenizer(in: dir) && hasMLPackage(in: dir)
    }

    nonisolated var applicationSupportDirectory: URL {
        let fm = FileManager.default
        let base: URL = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let dir = base.appendingPathComponent("CPlanner", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private nonisolated func hasTokenizer(in dir: URL) -> Bool {
        let fm = FileManager.default
        let req = ["tokenizer.json", "tokenizer_config.json"]
        return req.allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
    }

    private nonisolated func hasCompiledModel(in dir: URL) -> Bool {
        let url = dir.appendingPathComponent("StatefulMistral7BInstructInt4.mlmodelc")
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
        return exists && isDir.boolValue
    }

    private nonisolated func hasMLPackage(in dir: URL) -> Bool {
        let manifest = dir.appendingPathComponent("StatefulMistral7BInstructInt4.mlpackage")
            .appendingPathComponent("Manifest.json")
        return FileManager.default.fileExists(atPath: manifest.path)
    }

    // MARK: - State transitions

    func checkInstallation() {
        if resolveModelLocation() != nil {
            logger.info("Model + tokenizer found, ready.")
            state = .ready
            Task { await LocalLLMService.shared.reload() }
        } else if hasUnpackedModelInAppSupport() {
            logger.info(".mlpackage found but not yet compiled — compiling now.")
            state = .compiling
            Task { await compileAndReady() }
        } else {
            logger.info("Model + tokenizer missing, prompting download.")
            state = .needsDownload
        }
    }

    /// 사용자가 "나중에"를 선택한 경우. 같은 세션에선 alert 다시 뜨지 않게 한다.
    func skip() {
        state = .skipped
    }

    func startDownload() {
        Task { await performDownload() }
    }

    // 모델이 데이터의 ~99%이므로 진행 바를 데이터 비율에 가깝게 가중.
    nonisolated private static let modelWeight: Double = 0.95
    nonisolated private static let tokenizerWeight: Double = 0.05

    private func performDownload() async {
        let dest = applicationSupportDirectory

        state = .downloading(progress: 0.0, statusText: "Mistral 모델 준비 중…")
        do {
            try await downloadModelViaHub()
        } catch {
            let msg: String
            if let hubError = error as? Hub.HubClientError, case .authorizationRequired = hubError {
                msg = "모델 저장소 접근 권한 거부 — HuggingFace 토큰이 필요할 수 있습니다."
            } else {
                msg = "모델 다운로드 실패: \(error.localizedDescription)"
            }
            logger.error("Model snapshot failed: \(error.localizedDescription, privacy: .public)")
            state = .failed(message: msg)
            return
        }

        let tokTotal = tokenizerFiles.count
        let perTokWeight = Self.tokenizerWeight / Double(tokTotal)
        for (i, name) in tokenizerFiles.enumerated() {
            let baseProgress = Self.modelWeight + Double(i) * perTokWeight
            state = .downloading(progress: baseProgress,
                                 statusText: "토크나이저 (\(i + 1)/\(tokTotal)): \(name)")

            let url = URL(string: "https://huggingface.co/\(tokenizerRepo)/resolve/main/\(name)")!
            let destURL = dest.appendingPathComponent(name)
            do {
                try await downloadFile(url: url, destination: destURL) { written, expected in
                    let perFile: Double = expected > 0 ? Double(written) / Double(expected) : 0
                    let overall = baseProgress + perFile * perTokWeight
                    let kb = Double(written) / 1024
                    let stat: String
                    if expected > 0 {
                        let totalKB = Double(expected) / 1024
                        stat = "토크나이저 (\(i + 1)/\(tokTotal)): \(name) — \(Int(kb))/\(Int(totalKB)) KB"
                    } else {
                        stat = "토크나이저 (\(i + 1)/\(tokTotal)): \(name) — \(Int(kb)) KB"
                    }
                    Task { @MainActor [weak self] in
                        self?.state = .downloading(progress: overall, statusText: stat)
                    }
                }
            } catch {
                logger.error("Tokenizer download failed for \(name, privacy: .public): \(error.localizedDescription, privacy: .public)")
                state = .failed(message: "토크나이저 다운로드 실패: \(name)\n\(error.localizedDescription)")
                return
            }
        }

        if hasUnpackedModelInAppSupport() {
            logger.info("Download complete, compiling .mlpackage to .mlmodelc...")
            await compileAndReady()
        } else {
            logger.error("Files downloaded but verification failed")
            state = .failed(message: "다운로드는 완료됐지만 파일 검증에 실패했습니다.")
        }
    }

    private func downloadModelViaHub() async throws {
        let appSupport = applicationSupportDirectory
        let hubBase = appSupport.appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hubBase, withIntermediateDirectories: true)

        let api = HubApi(downloadBase: hubBase)
        let repo = Hub.Repo(id: modelRepo)
        let glob = "\(modelDirName)/*"

        let snapshotURL = try await api.snapshot(from: repo, matching: [glob]) { @Sendable progress in
            let frac = progress.fractionCompleted
            let stat = "Mistral 모델 다운로드 중 — \(Int(frac * 100))%"
            let overall = frac * Self.modelWeight
            Task { @MainActor in
                ModelInstaller.shared.state = .downloading(progress: overall, statusText: stat)
            }
        }

        // Hub은 <hubBase>/models/<repo.id>/ 안에 받는다. .mlpackage를 AppSupport 루트로 옮긴다.
        let downloadedPackage = snapshotURL.appendingPathComponent(modelDirName)
        let destPackage = appSupport.appendingPathComponent(modelDirName)
        if FileManager.default.fileExists(atPath: destPackage.path) {
            try FileManager.default.removeItem(at: destPackage)
        }
        try FileManager.default.moveItem(at: downloadedPackage, to: destPackage)

        // 캐시 디렉터리 정리
        try? FileManager.default.removeItem(at: hubBase)
    }

    private func compileAndReady() async {
        state = .compiling
        do {
            try await compileMLPackage()
        } catch {
            logger.error("Compile failed: \(error.localizedDescription, privacy: .public)")
            state = .failed(message: "모델 컴파일 실패: \(error.localizedDescription)")
            return
        }
        if resolveModelLocation() != nil {
            logger.info("Compile complete, model ready.")
            state = .ready
            await LocalLLMService.shared.reload()
        } else {
            state = .failed(message: "컴파일 후에도 모델 검증 실패")
        }
    }

    private func compileMLPackage() async throws {
        let appSupport = applicationSupportDirectory
        let packageURL = appSupport.appendingPathComponent("StatefulMistral7BInstructInt4.mlpackage")
        let tempCompiled = try await MLModel.compileModel(at: packageURL)
        let stableURL = appSupport.appendingPathComponent("StatefulMistral7BInstructInt4.mlmodelc")
        if FileManager.default.fileExists(atPath: stableURL.path) {
            try FileManager.default.removeItem(at: stableURL)
        }
        try FileManager.default.moveItem(at: tempCompiled, to: stableURL)
    }

    private func downloadFile(url: URL,
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

        // 임시 파일은 호출 종료 후 시스템이 정리하므로, 안정 위치로 옮긴다.
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
