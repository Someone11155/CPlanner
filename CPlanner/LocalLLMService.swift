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

    private init() {}

    /// Kick off model loading. Safe to call multiple times — subsequent
    /// calls observe the in-flight `loadTask` rather than starting a new
    /// load. Callers that just need the model ready before use should
    /// `await` `classifyTask`, which calls `ensureLoaded()` internally.
    func preload() {
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
        do {
            guard let resourceURL = Bundle.main.resourceURL else {
                llmLogger.error("Bundle.main.resourceURL is nil")
                return
            }
            self.tokenizer = try await AutoTokenizer.from(modelFolder: resourceURL)

            let configML = MLModelConfiguration()
            configML.computeUnits = .all

            let modelName = "StatefulMistral7BInstructInt4"
            guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
                llmLogger.error("모델 파일을 찾을 수 없습니다 (mlmodelc).")
                return
            }
            self.monoModel = try MLModel(contentsOf: modelURL, configuration: configML)
            llmLogger.info("[Mistral 7B] 로딩 완료")
        } catch {
            llmLogger.error("초기화 실패: \(error.localizedDescription, privacy: .public)")
        }
    }

    func classifyTask(taskTitle: String, availableFolders: [String]) async -> String {
        await ensureLoaded()
        guard let tokenizer = self.tokenizer, let model = self.monoModel else {
            return "일반"
        }

        // Strip Mistral instruct delimiters from the user-supplied title
        // so a malicious task name cannot break out of the prompt.
        let sanitizedTitle = taskTitle
            .replacingOccurrences(of: "[INST]", with: "")
            .replacingOccurrences(of: "[/INST]", with: "")
            .replacingOccurrences(of: "<s>", with: "")
            .replacingOccurrences(of: "</s>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

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

        // Mistral Instruct v3 포맷
        let prompt = """
        [INST] 분류 전문가로서 할 일을 카테고리 중 하나로 분류하세요. 대문자 알파벳 한 글자만 답하세요.
        카테고리:
        A. 운동
        B. 공부
        할 일: 헬스장 가기 [/INST] A </s> [INST] 카테고리:
        \(optionsText)
        할 일: \(sanitizedTitle) [/INST]
        """

        let inputTokens = tokenizer.encode(text: prompt)

        do {
            let state = model.makeState()
            var finalLogits: MLMultiArray?

            for (i, tokenID) in inputTokens.enumerated() {
                let inputIds = MLShapedArray<Int32>(scalars: [Int32(tokenID)], shape: [1, 1])
                // causalMask의 마지막 축(key length)은 KV cache 누적 위치(i+1)와 일치해야 한다.
                // 모든 위치는 0.0으로 마스크되지 않음 — 단일 query 토큰이 자신을 포함한
                // 이전 모든 토큰을 attend.
                let keyLen = i + 1
                let maskValues = [Float16](repeating: 0.0, count: keyLen)
                let causalMask = MLShapedArray<Float16>(scalars: maskValues, shape: [1, 1, 1, keyLen])

                let inputs: [String: Any] = [
                    "inputIds": MLMultiArray(inputIds),
                    "causalMask": MLMultiArray(causalMask)
                ]

                let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
                let prediction = try await model.prediction(from: provider, using: state)

                if i == inputTokens.count - 1 {
                    finalLogits = prediction.featureValue(for: "logits")?.multiArrayValue
                }
            }

            if let logits = finalLogits {
                let topTokens = getTopK(from: logits, k: 10)
                for token in topTokens {
                    let word = tokenizer.decode(tokens: [token])
                    let clean = word.trimmingCharacters(in: .whitespacesAndNewlines)
                                    .trimmingCharacters(in: .punctuationCharacters)
                                    .uppercased()
                    if clean.count == 1, let index = labels.firstIndex(of: clean) {
                        let result = usableFolders[index]
                        llmLogger.info("[Mistral] 분류 성공: \(sanitizedTitle, privacy: .public) -> \(result, privacy: .public) (\(clean, privacy: .public))")
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

    nonisolated func validateFileContext(fileName: String, folderName: String) -> Bool {
        let lower = fileName.lowercased()
        let junk = [".dmg", ".exe", ".mp4", ".zip"]
        return !junk.contains(where: { lower.hasSuffix($0) }) && lower.count >= 2
    }
}
