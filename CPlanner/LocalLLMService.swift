//
//  LocalLLMService.swift
//  Cplanner
//
//  Created by iLo on 2026-04-08.
//

import Foundation
import CoreML
import Tokenizers
import Hub
import os

// MLModel is an ObjC class not yet annotated as Sendable in the CoreML SDK.
// All access is serialised through LocalLLMService (an actor), so this is safe.
extension MLModel: @retroactive @unchecked Sendable {}

nonisolated(unsafe) private let llmLogger = Logger(subsystem: "com.cplanner", category: "LocalLLMService")

/// Single uppercase letters A–Z used as classification targets.
/// 26 folders is more than enough in practice; if a user defines more
/// than 26, the trailing folders are silently unreachable from the
/// classifier and we log a warning.
nonisolated(unsafe) private let classificationLabelAlphabet: [String] =
    (0..<26).map { i in String(UnicodeScalar(UInt8(0x41 + i))) }

@available(macOS 15.0, iOS 18.0, *)
actor LocalLLMService {
    static let shared = LocalLLMService()

    private var tokenizer: Tokenizer?
    private var monoModel: MLModel?
    private var loadTask: Task<Void, Never>?
    /// 알파벳 letter('A'..'Z')별 token ID 후보. SentencePiece 변형 (`A`, ` A`, `▁A`, `\nA`)를
    /// 모두 시도해 모은 set. setupMistral() 끝에서 한 번만 빌드하고 classifyTask()에서 재사용.
    /// 비어 있으면(빌드 실패 등) 기존 top-K fallback 경로 사용.
    private var alphabetTokenIDs: [String: [Int]] = [:]

    private init() {}

    /// Kick off model loading. Safe to call multiple times — subsequent
    /// calls observe the in-flight `loadTask` rather than starting a new
    /// load. Callers that just need the model ready before use should
    /// `await` `classifyTask`, which calls `ensureLoaded()` internally.
    func preload() {
        ensureLoadTaskStarted()
    }

    /// 모델 파일이 새로 설치된 직후 다시 로드하기 위한 리셋.
    func reload() {
        loadTask = nil
        tokenizer = nil
        monoModel = nil
        alphabetTokenIDs = [:]
        ensureLoadTaskStarted()
    }

    private func ensureLoadTaskStarted() {
        if loadTask == nil {
            loadTask = Task { [weak self] in
                await self?.setupMistral()
            }
        }
    }

    private func ensureLoaded() async {
        ensureLoadTaskStarted()
        await loadTask?.value
    }

    private func setupMistral() async {
        llmLogger.info("[Mistral 7B] 모델 로딩 시작")
        let location = await MainActor.run { ModelInstaller.shared.resolveModelLocation() }
        guard let location else {
            llmLogger.warning("모델/토크나이저가 설치되지 않음 — classifyTask는 '일반'을 반환")
            return
        }
        do {
            self.tokenizer = try await AutoTokenizer.from(modelFolder: location)

            let configML = MLModelConfiguration()
            configML.computeUnits = .all

            let modelURL = location.appendingPathComponent("StatefulMistral7BInstructInt4.mlmodelc")
            self.monoModel = try MLModel(contentsOf: modelURL, configuration: configML)
            buildAlphabetTokenMap()
            llmLogger.info("[Mistral 7B] 로딩 완료")
        } catch {
            llmLogger.error("초기화 실패: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// A~Z 각 letter에 대해 SentencePiece 변형으로 인코딩해 token ID를 모음.
    /// classifyTask() 끝에서 logits[id]만 비교하면 vocab 32K 전체 sort 없이 argmax 가능.
    private func buildAlphabetTokenMap() {
        guard let tokenizer = self.tokenizer else { return }
        var map: [String: [Int]] = [:]
        for letter in classificationLabelAlphabet {
            var ids = Set<Int>()
            // SentencePiece는 위치/공백 prefix에 따라 다른 token ID가 나올 수 있어 후보 폭넓게 시도
            let variants = [letter, " " + letter, "\n" + letter, "▁" + letter]
            for v in variants {
                let tokens = tokenizer.encode(text: v)
                if tokens.count == 1 {
                    ids.insert(tokens[0])
                } else if tokens.count == 2 {
                    // BOS + letter token 케이스 — 마지막이 실제 letter
                    ids.insert(tokens[1])
                }
                // 3개 이상이면 letter가 multi-token으로 분리된 비정상 상황 → 신뢰 못함, skip
            }
            if !ids.isEmpty {
                map[letter] = Array(ids)
            }
        }
        self.alphabetTokenIDs = map
        llmLogger.info("[Mistral 7B] 알파벳 토큰 맵 빌드: \(map.count, privacy: .public) letters covered")
    }

    func classifyTask(taskTitle: String, availableFolders: [String], corrections: [Correction] = []) async -> String {
        // F1.2 — 정확 매치 단축: 동일(정규화된) 제목의 분류 이력이 있고 그 폴더가 현재도 살아있으면 LLM 호출 생략
        // 정규화는 trim + lowercase까지만 (보수적). 공백/구두점 제거는 의도하지 않은 매치 위험.
        let normalizedNew = taskTitle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        for c in corrections {
            let normalizedOld = c.taskTitle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if normalizedNew == normalizedOld, availableFolders.contains(c.folderName) {
                llmLogger.info("[Mistral] 정확 매치 단축 — LLM 호출 생략: \(taskTitle, privacy: .public) -> \(c.folderName, privacy: .public)")
                return c.folderName
            }
        }

        await ensureLoaded()
        guard let tokenizer = self.tokenizer, let model = self.monoModel else {
            return "일반"
        }

        // Strip Mistral instruct delimiters from the user-supplied title
        // so a malicious task name cannot break out of the prompt.
        let sanitizedTitle = sanitize(taskTitle)

        let usableFolders = Array(availableFolders.prefix(classificationLabelAlphabet.count))
        if availableFolders.count > usableFolders.count {
            llmLogger.warning("More than \(classificationLabelAlphabet.count) folders provided; trailing entries will not be classifiable.")
        }
        let labels = Array(classificationLabelAlphabet.prefix(usableFolders.count))

        guard !labels.isEmpty else { return "일반" }

        var optionsText = ""
        for (i, folder) in usableFolders.enumerated() {
            optionsText += "\(labels[i]). \(folder)\n"
        }

        // 사용자 수정 이력을 few-shot 예시로 누적 — 같은 카테고리 옵션 + 정답 라벨 형태
        var fewShotExamples = ""
        for c in corrections {
            guard let folderIdx = usableFolders.firstIndex(of: c.folderName) else { continue }
            let label = labels[folderIdx]
            let safeTitle = sanitize(c.taskTitle)
            fewShotExamples += "[INST] 카테고리:\n\(optionsText)할 일: \(safeTitle) [/INST] \(label) </s> "
        }

        // F1.3 — 사용자 corrections가 있으면 정적 시드 예시 생략 (보수적 임계값: corrections.count >= 1).
        // corrections가 형식 demonstration을 충분히 제공하므로 운동/공부 시드는 noise + 25토큰만 차지.
        // corrections == 0인 cold start만 시드 유지 (안전망).
        let staticSeed = corrections.isEmpty
            ? "[INST] 분류 전문가로서 할 일을 카테고리 중 하나로 분류하세요. 대문자 알파벳 한 글자만 답하세요.\n카테고리:\nA. 운동\nB. 공부\n할 일: 헬스장 가기 [/INST] A </s> "
            : ""
        // Mistral Instruct v3 포맷 — (조건부 정적 시드) + 사용자 수정 예시 + 실제 분류 턴
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
                // inputIds: int32 [1, 1]
                let inputIdsMA = try MLMultiArray(shape: [1, 1], dataType: .int32)
                inputIdsMA[0] = NSNumber(value: Int32(tokenID))

                // causalMask: fp16 [1, 1, 1, keyLen], 모든 위치 0 (마스크 없음)
                let keyLen = i + 1
                let maskMA = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: keyLen)], dataType: .float16)
                let maskBytes = maskMA.dataPointer.bindMemory(to: UInt16.self, capacity: keyLen)
                for j in 0..<keyLen { maskBytes[j] = 0 } // fp16 zero == 0x0000

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

            // F1.1 — Fast logits path: 알파벳 token ID들의 logits만 비교 → argmax. vocab 32K 전체 sort 회피.
            if let logits = finalLogits, !alphabetTokenIDs.isEmpty {
                var bestLetterIdx: Int? = nil
                var bestScore: Float = -.greatestFiniteMagnitude
                let logitsCount = logits.count
                for (idx, letter) in labels.enumerated() {
                    guard let ids = alphabetTokenIDs[letter] else { continue }
                    for id in ids where id < logitsCount {
                        let score = logits[id].floatValue
                        if score > bestScore {
                            bestScore = score
                            bestLetterIdx = idx
                        }
                    }
                }
                if let idx = bestLetterIdx {
                    let result = usableFolders[idx]
                    llmLogger.info("[Mistral] 분류 성공 (fast): \(sanitizedTitle, privacy: .public) -> \(result, privacy: .public) (\(labels[idx], privacy: .public))")
                    return result
                }
                llmLogger.warning("[Mistral] fast path 매치 0건 — top-K fallback")
            }

            // Fallback: 알파벳 맵이 비었거나 fast path 실패 시 기존 top-K 경로
            if let logits = finalLogits {
                let topTokens = getTopK(from: logits, k: 10)
                for token in topTokens {
                    let word = tokenizer.decode(tokens: [token])
                    let clean = word.trimmingCharacters(in: .whitespacesAndNewlines)
                                    .trimmingCharacters(in: .punctuationCharacters)
                                    .uppercased()
                    if clean.count == 1, let index = labels.firstIndex(of: clean) {
                        let result = usableFolders[index]
                        llmLogger.info("[Mistral] 분류 성공 (top-K): \(sanitizedTitle, privacy: .public) -> \(result, privacy: .public) (\(clean, privacy: .public))")
                        return result
                    }
                }
            }
        } catch {
            llmLogger.error("추론 중 에러: \(error.localizedDescription, privacy: .public)")
        }

        llmLogger.warning("[Mistral] 분류 실패, '일반'으로 반환")
        return "일반"
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

    nonisolated func validateFileContext(fileName: String, folderName: String) -> Bool {
        let lower = fileName.lowercased()
        let junk = [".dmg", ".exe", ".mp4", ".zip"]
        return !junk.contains(where: { lower.hasSuffix($0) }) && lower.count >= 2
    }
}
