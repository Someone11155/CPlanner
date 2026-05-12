//
//  LocalLLMService.swift
//  Cplanner
//
//  분류 백엔드 추상화 — `LLMBackend` 프로토콜로 모델별 구현체를 swap 가능하게 함.
//  Phase 1+2 (2026-05-11): Gemma 4 E2B / E4B 두 변종을 `GemmaBackend`로 통합 처리.
//  Phase 3 (다음 세션): `MistralBackend` 추가 예정 — raw MLModel + MLState + softmax 신뢰도.
//
//  `LocalLLMService`는 백엔드 무관 coordinator —
//    1) F1.2 정확매치 단축 (모델 무관, 캐시된 corrections 비교)
//    2) 백엔드 위임 (현재 attached된 backend.classify로 라우팅)
//  `ModelInstaller`가 사용자 선택(`ModelKind`)에 따라 `GemmaBackend` 또는 (Phase 3)
//  `MistralBackend`를 만들어 `attachBackend(_:)`로 넘겨준다.
//
//  Gemma confidence semantics: 라이브러리(`john-rocky/CoreML-LLM`)가 raw logits를 노출 안 함 →
//  알파벳 token softmax 불가. binary로 단순화 — 깨끗한 단일 A-Z + 유효 폴더 매핑 = 1.0,
//  그 외(빈/multi-char/알파벳 외/범위 밖) = "일반" + 0.0. Mistral 복원 시(Phase 3)에는
//  softmax 0~1 + 75% threshold가 부활한다.
//

import Foundation
import CoreML
import CoreMLLLM
import Tokenizers
import os

// MLModel은 ObjC class로 SDK가 아직 Sendable 표시를 안 함. MistralBackend actor 안에서만 접근하므로 안전.
extension MLModel: @retroactive @unchecked Sendable {}

nonisolated(unsafe) private let llmLogger = Logger(subsystem: "com.cplanner", category: "LocalLLMService")

/// 분류 라벨 후보 — 대문자 알파벳 26개. 폴더 26개 초과 시 trailing은 분류 불가능 (warning).
nonisolated(unsafe) private let classificationLabelAlphabet: [String] =
    (0..<26).map { i in String(UnicodeScalar(UInt8(0x41 + i))) }

// MARK: - Model kinds

/// 사용자가 선택 가능한 LLM 모델 종류. `UserDefaults`에 `rawValue`로 영속화됨.
/// 프로젝트가 Swift 6 default-isolation = MainActor라 actor 컨텍스트에서 접근 가능하도록 각 멤버 `nonisolated`.
enum ModelKind: String, Codable, CaseIterable, Sendable {
    case gemma4E2B = "gemma-4-e2b"
    case gemma4E4B = "gemma-4-e4b"
    case mistral7B = "mistral-7b"

    /// UI에 표시할 짧은 이름.
    nonisolated var displayName: String {
        switch self {
        case .gemma4E2B: return "Gemma 4 E2B"
        case .gemma4E4B: return "Gemma 4 E4B"
        case .mistral7B: return "Mistral 7B"
        }
    }

    /// 대략 다운로드 크기 (GB).
    nonisolated var sizeGB: Double {
        switch self {
        case .gemma4E2B: return 5.4
        case .gemma4E4B: return 5.5
        case .mistral7B: return 3.8
        }
    }

    /// 사용자에게 보일 짧은 특징 설명.
    nonisolated var summary: String {
        switch self {
        case .gemma4E2B: return "빠름 · ANE 최적화 · 5.4GB"
        case .gemma4E4B: return "더 정확 · 더 큰 모델 · 5.5GB"
        case .mistral7B: return "신뢰도 % 표시 · 다소 느림 · 3.8GB"
        }
    }

    /// 좀 더 자세한 설명 — picker 카드용.
    nonisolated var detail: String {
        switch self {
        case .gemma4E2B:
            return "35 layers, hidden=2048. 분류 1건 ~730ms. ANE 친화적, 가벼움. 신뢰도는 binary (1.0 / 0.0)."
        case .gemma4E4B:
            return "42 layers, hidden=2560. 분류 ~1~2s 예상. 더 큰 컨텍스트 이해, 어려운 입력에 더 강함. 신뢰도는 binary."
        case .mistral7B:
            return "원조 모델. 분류 1건 ~8.8s, ANE 모드 hang 가능 → CPU+GPU 권장. softmax 신뢰도(0~100%) 노출."
        }
    }

    /// 사용 가능한 모델. Phase 3 (2026-05-12)에서 Mistral도 활성화.
    nonisolated var isAvailable: Bool { true }

    /// 각 모델에 권장되는 기본 compute units.
    nonisolated var defaultComputeUnits: MLComputeUnits {
        switch self {
        case .gemma4E2B, .gemma4E4B: return .cpuAndNeuralEngine
        case .mistral7B: return .cpuAndGPU  // Mistral은 ANE 모드에서 hang했었음
        }
    }
}

// MARK: - Selected model persistence

/// 현재 선택된 모델의 영속 저장소 (`UserDefaults`). 첫 실행 시 nil이라
/// `ModelInstaller.state == .awaitingSelection`으로 picker가 뜬다.
enum ModelSelection {
    static let key = "cplanner.selectedModel"

    static var current: ModelKind? {
        get {
            UserDefaults.standard.string(forKey: key).flatMap { ModelKind(rawValue: $0) }
        }
        set {
            if let v = newValue { UserDefaults.standard.set(v.rawValue, forKey: key) }
            else { UserDefaults.standard.removeObject(forKey: key) }
        }
    }
}

// MARK: - Backend protocol

/// 분류 백엔드 추상화. `ModelInstaller`가 사용자 선택한 모델에 맞는 구현체를 생성해
/// `LocalLLMService.attachBackend(_:computeUnits:)`에 넘겨준다.
@available(macOS 15.0, iOS 18.0, *)
protocol LLMBackend: AnyObject, Sendable {
    nonisolated var kind: ModelKind { get }
    /// 실제 LLM 호출. corrections는 이미 ContentView에서 활성 폴더로 필터된 상태.
    /// returns: (선택 폴더 이름 또는 "일반", confidence 0.0~1.0)
    func classify(taskTitle: String, availableFolders: [String], corrections: [Correction]) async -> (folder: String, confidence: Double)
}

// MARK: - Gemma backend (E2B + E4B)

/// Gemma 4 (E2B / E4B 공용). 라이브러리 `CoreMLLLM.generate(messages:maxTokens:1)`로 1-token greedy
/// decode → A-Z letter 추출 → 폴더 매핑. 두 변종은 라이브러리 ModelInfo만 다르고 inference 경로는 동일.
@available(macOS 15.0, iOS 18.0, *)
actor GemmaBackend: LLMBackend {
    nonisolated let kind: ModelKind
    private let llm: CoreMLLLM

    init(kind: ModelKind, llm: CoreMLLLM) {
        precondition(kind == .gemma4E2B || kind == .gemma4E4B, "GemmaBackend requires Gemma kind")
        self.kind = kind
        self.llm = llm
    }

    func classify(taskTitle: String, availableFolders: [String], corrections: [Correction]) async -> (folder: String, confidence: Double) {
        let usableFolders = Array(availableFolders.prefix(classificationLabelAlphabet.count))
        if availableFolders.count > usableFolders.count {
            llmLogger.warning("폴더가 26개를 초과 — 후행 폴더는 분류 불가능.")
        }
        let labels = Array(classificationLabelAlphabet.prefix(usableFolders.count))
        guard !labels.isEmpty else { return ("일반", 0.0) }

        let sanitizedTitle = sanitize(taskTitle)
        var optionsText = ""
        for (i, folder) in usableFolders.enumerated() {
            optionsText += "\(labels[i]). \(folder)\n"
        }

        // System 안내 — Gemma chat template은 라이브러리가 토크나이저 메타로 자동 적용.
        var messages: [CoreMLLLM.Message] = [
            .init(role: .system, content: "할 일의 핵심 키워드(주제 단어)를 보고 가장 의미적으로 가까운 카테고리를 선택하세요. 카테고리 라벨의 위치(A/B/C…)는 매번 다르니 letter 자체가 아니라 카테고리 이름의 의미를 보고 판단해야 합니다. 답은 정확히 대문자 한 글자만.")
        ]

        // Cold-start seed (4개) — corrections 유무와 무관하게 항상 포함.
        // 의도: (a) "대문자 한 글자만" format 학습 (b) 2/3-way 다양 (c) letter↔domain 위치가
        // 매번 다르다(예: "운동"이 A에도 C에도 등장)는 메타 규칙 학습. corrections만으로는 부족.
        messages.append(.init(role: .user, content: "카테고리:\nA. 운동\nB. 공부\n할 일: 헬스장 가기"))
        messages.append(.init(role: .assistant, content: "A"))
        messages.append(.init(role: .user, content: "카테고리:\nA. 한국어\nB. 영어\n할 일: 한글 문법 정리"))
        messages.append(.init(role: .assistant, content: "A"))
        messages.append(.init(role: .user, content: "카테고리:\nA. 음악\nB. 코딩\n할 일: 알고리즘 문제 풀기"))
        messages.append(.init(role: .assistant, content: "B"))
        messages.append(.init(role: .user, content: "카테고리:\nA. 영화\nB. 책\nC. 음식\n할 일: 라면 끓이기"))
        messages.append(.init(role: .assistant, content: "C"))

        // 사용자 corrections — 도메인-specific mapping 학습용. cold-start 뒤에 덧붙임.
        for c in corrections {
            guard let folderIdx = usableFolders.firstIndex(of: c.folderName) else { continue }
            let label = labels[folderIdx]
            let safeTitle = sanitize(c.taskTitle)
            messages.append(.init(role: .user, content: "카테고리:\n\(optionsText)할 일: \(safeTitle)"))
            messages.append(.init(role: .assistant, content: label))
        }
        messages.append(.init(role: .user, content: "카테고리:\n\(optionsText)할 일: \(sanitizedTitle)"))

        do {
            llm.reset()  // 이전 턴의 KV cache 오염 방지
            let output = try await llm.generate(messages, maxTokens: 1)
            let clean = output.trimmingCharacters(in: .whitespacesAndNewlines)
                              .trimmingCharacters(in: .punctuationCharacters)
                              .uppercased()
            if clean.count == 1, let index = labels.firstIndex(of: clean) {
                let result = usableFolders[index]
                llmLogger.info("[\(self.kind.displayName, privacy: .public)] 분류 성공: \(sanitizedTitle, privacy: .public) -> \(result, privacy: .public) (\(clean, privacy: .public))")
                return (result, 1.0)
            }
            llmLogger.info("[\(self.kind.displayName, privacy: .public)] 매치 안 됨 — '일반'으로 fallback. raw='\(output, privacy: .public)' clean='\(clean, privacy: .public)'")
            return ("일반", 0.0)
        } catch {
            llmLogger.error("[\(self.kind.displayName, privacy: .public)] inference 에러: \(error.localizedDescription, privacy: .public)")
            return ("일반", 0.0)
        }
    }

    private func sanitize(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<start_of_turn>", with: "")
            .replacingOccurrences(of: "<end_of_turn>", with: "")
            .replacingOccurrences(of: "<bos>", with: "")
            .replacingOccurrences(of: "<eos>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Mistral backend

/// Mistral 7B Instruct v0.3 (apple/mistral-coreml int4 stateful 변종).
/// 라이브러리(`john-rocky/CoreML-LLM`)가 Mistral 미지원이라 raw `MLModel` + `MLState` 직접 운용.
///
/// **Inference 경로:** prompt를 토큰 단위로 prefill (state 누적) → 마지막 토큰 위치의 logits 추출
/// → 알파벳 letter token IDs(SentencePiece variants 포함)에 대해서만 logit 비교 → softmax로 신뢰도.
///
/// **신뢰도 semantics:** softmax over labels-only (vocab 32K 전체가 아님) — 사용자 입장에선
/// "후보 폴더 N개 중 winner의 상대적 확신도". 75% threshold 미만이면 결과 폐기 + "일반"으로 fallback.
///
/// 옛 commit `8267940`의 `LocalLLMService.classifyTask` 220줄을 actor화. F1.2 정확매치 단축은
/// `LocalLLMService` coordinator 쪽에서 처리하므로 여기선 logits 경로만 담당.
@available(macOS 15.0, iOS 18.0, *)
actor MistralBackend: LLMBackend {
    nonisolated let kind: ModelKind = .mistral7B

    private let tokenizer: Tokenizer
    private let model: MLModel
    /// SentencePiece 변형(`A`, ` A`, `▁A`, `\nA`)으로 인코딩한 letter→token ID 후보 맵.
    /// classify에서 logits[id]만 비교하면 vocab 전체 sort 없이 argmax 가능.
    private let alphabetTokenIDs: [String: [Int]]

    /// 사용자 결정 (2026-05-04): softmax winner가 다른 폴더들보다 ~7-8배 확신 있을 때만 통과.
    nonisolated private static let confidenceThreshold: Double = 0.75

    init(tokenizer: Tokenizer, model: MLModel) {
        self.tokenizer = tokenizer
        self.model = model
        var map: [String: [Int]] = [:]
        for letter in classificationLabelAlphabet {
            var ids = Set<Int>()
            // SentencePiece는 위치/공백 prefix에 따라 다른 token ID. 후보 폭넓게.
            let variants = [letter, " " + letter, "\n" + letter, "▁" + letter]
            for v in variants {
                let tokens = tokenizer.encode(text: v)
                if tokens.count == 1 {
                    ids.insert(tokens[0])
                } else if tokens.count == 2 {
                    // BOS + letter token 케이스
                    ids.insert(tokens[1])
                }
            }
            if !ids.isEmpty { map[letter] = Array(ids) }
        }
        self.alphabetTokenIDs = map
        llmLogger.info("[Mistral 7B] 알파벳 토큰 맵 빌드: \(map.count, privacy: .public) letters covered")
    }

    func classify(taskTitle: String, availableFolders: [String], corrections: [Correction]) async -> (folder: String, confidence: Double) {
        let usableFolders = Array(availableFolders.prefix(classificationLabelAlphabet.count))
        if availableFolders.count > usableFolders.count {
            llmLogger.warning("폴더가 26개를 초과 — 후행 폴더는 분류 불가능.")
        }
        let labels = Array(classificationLabelAlphabet.prefix(usableFolders.count))
        guard !labels.isEmpty else { return ("일반", 0.0) }

        let sanitizedTitle = sanitize(taskTitle)
        var optionsText = ""
        for (i, folder) in usableFolders.enumerated() {
            optionsText += "\(labels[i]). \(folder)\n"
        }

        // 사용자 corrections few-shot
        var fewShotExamples = ""
        for c in corrections {
            guard let folderIdx = usableFolders.firstIndex(of: c.folderName) else { continue }
            let label = labels[folderIdx]
            let safeTitle = sanitize(c.taskTitle)
            fewShotExamples += "[INST] 카테고리:\n\(optionsText)할 일: \(safeTitle) [/INST] \(label) </s> "
        }

        // F1.3 — corrections가 있으면 cold-start seed 생략 (corrections가 format demonstration 충분).
        // corrections == 0인 cold start만 시드 유지.
        let staticSeed = corrections.isEmpty
            ? "[INST] 분류 전문가로서 할 일을 카테고리 중 하나로 분류하세요. 대문자 알파벳 한 글자만 답하세요.\n카테고리:\nA. 운동\nB. 공부\n할 일: 헬스장 가기 [/INST] A </s> "
            : ""

        let prompt = """
        \(staticSeed)\(fewShotExamples)[INST] 카테고리:
        \(optionsText)
        할 일: \(sanitizedTitle) [/INST]
        """

        let inputTokens = tokenizer.encode(text: prompt)

        do {
            let state = model.makeState()
            var finalLogits: MLMultiArray?

            for (i, tokenID) in inputTokens.enumerated() {
                let inputIdsMA = try MLMultiArray(shape: [1, 1], dataType: .int32)
                inputIdsMA[0] = NSNumber(value: Int32(tokenID))

                // causalMask: fp16 [1, 1, 1, keyLen], 모든 위치 0 (마스크 없음)
                let keyLen = i + 1
                let maskMA = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: keyLen)], dataType: .float16)
                let maskBytes = maskMA.dataPointer.bindMemory(to: UInt16.self, capacity: keyLen)
                for j in 0..<keyLen { maskBytes[j] = 0 }

                let inputs: [String: Any] = [
                    "inputIds": inputIdsMA,
                    "causalMask": maskMA
                ]

                let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
                let prediction = try await model.prediction(from: provider, using: state)

                if i == inputTokens.count - 1 {
                    finalLogits = prediction.featureValue(for: "logits")?.multiArrayValue
                }
            }

            // F1.1 — Fast logits path: 알파벳 token IDs만 비교 → argmax + softmax.
            if let logits = finalLogits, !alphabetTokenIDs.isEmpty {
                let logitsCount = logits.count
                var letterLogits: [Float] = []
                letterLogits.reserveCapacity(labels.count)
                for letter in labels {
                    var best: Float = -.greatestFiniteMagnitude
                    if let ids = alphabetTokenIDs[letter] {
                        for id in ids where id < logitsCount {
                            let s = logits[id].floatValue
                            if s > best { best = s }
                        }
                    }
                    letterLogits.append(best)
                }
                if let bestScore = letterLogits.max(), bestScore > -.greatestFiniteMagnitude,
                   let bestIdx = letterLogits.firstIndex(of: bestScore) {
                    // softmax — numerically stable: subtract max
                    let expValues = letterLogits.map { Double(exp(Double($0 - bestScore))) }
                    let sumExp = expValues.reduce(0.0, +)
                    let confidence = sumExp > 0 ? expValues[bestIdx] / sumExp : 0.0
                    let result = usableFolders[bestIdx]
                    if confidence < Self.confidenceThreshold {
                        llmLogger.info("[Mistral 7B] 신뢰도 부족 (\(Int(confidence * 100), privacy: .public)% < \(Int(Self.confidenceThreshold * 100), privacy: .public)%) — '일반'으로 fallback: \(sanitizedTitle, privacy: .public) (would have been \(result, privacy: .public))")
                        return ("일반", confidence)
                    }
                    llmLogger.info("[Mistral 7B] 분류 성공 (fast): \(sanitizedTitle, privacy: .public) -> \(result, privacy: .public) (\(labels[bestIdx], privacy: .public), \(Int(confidence * 100), privacy: .public)%)")
                    return (result, confidence)
                }
                llmLogger.warning("[Mistral 7B] fast path 매치 0건 — top-K fallback")
            }

            // Fallback: 알파벳 맵이 비었거나 fast path 실패 시 top-K 탐색. 신뢰도 보수적 0.5 → threshold 미만 → 일반.
            if let logits = finalLogits {
                let topTokens = getTopK(from: logits, k: 10)
                for token in topTokens {
                    let word = tokenizer.decode(tokens: [token])
                    let clean = word.trimmingCharacters(in: .whitespacesAndNewlines)
                                    .trimmingCharacters(in: .punctuationCharacters)
                                    .uppercased()
                    if clean.count == 1, let index = labels.firstIndex(of: clean) {
                        let result = usableFolders[index]
                        let confidence = 0.5
                        if confidence < Self.confidenceThreshold {
                            llmLogger.info("[Mistral 7B] 신뢰도 부족 (top-K \(Int(confidence * 100), privacy: .public)%) — '일반'으로 fallback: \(sanitizedTitle, privacy: .public)")
                            return ("일반", confidence)
                        }
                        llmLogger.info("[Mistral 7B] 분류 성공 (top-K): \(sanitizedTitle, privacy: .public) -> \(result, privacy: .public)")
                        return (result, confidence)
                    }
                }
            }
        } catch {
            llmLogger.error("[Mistral 7B] 추론 에러: \(error.localizedDescription, privacy: .public)")
        }

        llmLogger.warning("[Mistral 7B] 분류 실패, '일반' 반환")
        return ("일반", 0.0)
    }

    private func getTopK(from logits: MLMultiArray, k: Int) -> [Int] {
        var topTokens = [(index: Int, score: Float)]()
        topTokens.reserveCapacity(logits.count)
        for i in 0..<logits.count {
            topTokens.append((index: i, score: logits[i].floatValue))
        }
        topTokens.sort { $0.score > $1.score }
        return topTokens.prefix(k).map { $0.index }
    }

    private func sanitize(_ text: String) -> String {
        text
            .replacingOccurrences(of: "[INST]", with: "")
            .replacingOccurrences(of: "[/INST]", with: "")
            .replacingOccurrences(of: "<s>", with: "")
            .replacingOccurrences(of: "</s>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - LocalLLMService (coordinator)

@available(macOS 15.0, iOS 18.0, *)
actor LocalLLMService {
    static let shared = LocalLLMService()

    private var backend: (any LLMBackend)?
    private(set) var currentComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    private(set) var currentModelKind: ModelKind?

    private init() {}

    // MARK: - Lifecycle (driven by ModelInstaller)

    /// ModelInstaller가 라이브러리/raw load 완료 후 호출.
    func attachBackend(_ backend: any LLMBackend, computeUnits: MLComputeUnits) {
        self.backend = backend
        self.currentModelKind = backend.kind
        self.currentComputeUnits = computeUnits
        llmLogger.info("[\(backend.kind.displayName, privacy: .public)] attached. units=\(computeUnits.label, privacy: .public)")
    }

    /// 컴퓨트 유닛 변경 / 모델 swap 직전에 호출.
    func detachBackend() {
        if let k = currentModelKind {
            llmLogger.info("[\(k.displayName, privacy: .public)] detached.")
        }
        self.backend = nil
        self.currentModelKind = nil
    }

    /// AppSettingsView에서 사용자가 compute units picker 변경 시 호출.
    func setComputeUnits(_ units: MLComputeUnits) async {
        guard units != currentComputeUnits else { return }
        currentComputeUnits = units
        await MainActor.run {
            ModelInstaller.shared.reloadWithComputeUnits(units)
        }
    }

    // MARK: - Classification

    /// 분류 결과 + 신뢰도. F1.2 정확매치 단축은 backend 무관이라 여기서 처리,
    /// 그 외는 attached backend에 위임. backend가 nil이면 "일반"+0.0.
    func classifyTask(taskTitle: String, availableFolders: [String], corrections: [Correction] = []) async -> (folder: String, confidence: Double) {
        // F1.2 — 정확 매치 단축: 동일(정규화된) 제목 이력이 있고 그 폴더가 살아있으면 LLM 호출 생략.
        let normalizedNew = taskTitle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        for c in corrections {
            let normalizedOld = c.taskTitle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if normalizedNew == normalizedOld, availableFolders.contains(c.folderName) {
                llmLogger.info("정확 매치 단축: \(taskTitle, privacy: .public) -> \(c.folderName, privacy: .public)")
                return (c.folderName, 1.0)
            }
        }

        guard let backend = self.backend else {
            llmLogger.warning("backend not attached — '일반' 반환")
            return ("일반", 0.0)
        }

        return await backend.classify(taskTitle: taskTitle, availableFolders: availableFolders, corrections: corrections)
    }

    nonisolated func validateFileContext(fileName: String, folderName: String) -> Bool {
        let lower = fileName.lowercased()
        let junk = [".dmg", ".exe", ".mp4", ".zip"]
        return !junk.contains(where: { lower.hasSuffix($0) }) && lower.count >= 2
    }
}

/// UI 표시용 라벨.
extension MLComputeUnits {
    nonisolated var label: String {
        switch self {
        case .cpuOnly: return "CPU만"
        case .cpuAndGPU: return "CPU + GPU"
        case .all: return "ANE + GPU + CPU"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        @unknown default: return "알 수 없음 (\(rawValue))"
        }
    }
}
