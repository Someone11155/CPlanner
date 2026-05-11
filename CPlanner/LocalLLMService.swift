//
//  LocalLLMService.swift
//  Cplanner
//
//  Gemma 4 E2B (CoreML) 기반 폴더 분류 서비스.
//  ModelInstaller가 john-rocky/CoreML-LLM 라이브러리로 모델을 로드한 뒤 attachLLM()으로 넘겨준다.
//  이 actor는 그 instance를 보관하며 classifyTask 호출 시 Message API로 1-token greedy decode를 실행한다.
//  Chat template은 라이브러리가 토크나이저 메타데이터로 자동 적용 (Gemma는 <start_of_turn>...<end_of_turn>).
//
//  Confidence 정책 (Mistral 시절 softmax-based 75% threshold에서 변경됨):
//  라이브러리가 raw logits를 노출하지 않아 알파벳 token softmax를 못 함. 따라서 binary로 단순화 —
//  출력이 깨끗한 단일 A-Z 문자이고 그 라벨이 사용 가능한 폴더 인덱스에 매핑되면 confidence = 1.0,
//  그 외 모든 경우(빈 출력, multi-char, 알파벳 외, 범위 밖 letter)는 "일반" + 0.0.
//  Exact-match shortcut(과거 corrections와 동일 제목)도 1.0.
//

import Foundation
import CoreML
import CoreMLLLM
import Tokenizers
import os

nonisolated(unsafe) private let llmLogger = Logger(subsystem: "com.cplanner", category: "LocalLLMService")

/// 분류 라벨 후보 — 대문자 알파벳 26개. 폴더 26개 초과 시 trailing은 분류 불가능 (warning).
nonisolated(unsafe) private let classificationLabelAlphabet: [String] =
    (0..<26).map { i in String(UnicodeScalar(UInt8(0x41 + i))) }
    
@available(macOS 15.0, iOS 18.0, *)
actor LocalLLMService {
    static let shared = LocalLLMService()

    private var llm: CoreMLLLM?
    private(set) var currentComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    private init() {}

    // MARK: - Lifecycle (driven by ModelInstaller)

    /// ModelInstaller가 라이브러리 load 완료 후 호출.
    func attachLLM(_ llm: CoreMLLLM, computeUnits: MLComputeUnits) {
        self.llm = llm
        self.currentComputeUnits = computeUnits
        llmLogger.info("[Gemma 4] attached. ctx=\(llm.contextLength, privacy: .public) units=\(computeUnits.label, privacy: .public)")
    }

    /// 컴퓨트 유닛 변경 등 reload 직전에 호출 — 기존 instance 해제.
    func detachLLM() {
        self.llm = nil
        llmLogger.info("[Gemma 4] detached.")
    }

    /// AppSettingsView/SettingsView에서 사용자가 picker 변경 시 호출.
    func setComputeUnits(_ units: MLComputeUnits) async {
        guard units != currentComputeUnits else { return }
        currentComputeUnits = units
        await MainActor.run {
            ModelInstaller.shared.reloadWithComputeUnits(units)
        }
    }

    // MARK: - Classification

    /// 분류 결과 + 신뢰도 (binary: 1.0 = 깨끗한 매치, 0.0 = "일반" fallback).
    func classifyTask(taskTitle: String, availableFolders: [String], corrections: [Correction] = []) async -> (folder: String, confidence: Double) {
        // F1.2 — 정확 매치 단축: 동일(정규화된) 제목 이력이 있고 그 폴더가 살아있으면 LLM 호출 생략.
        let normalizedNew = taskTitle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        for c in corrections {
            let normalizedOld = c.taskTitle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if normalizedNew == normalizedOld, availableFolders.contains(c.folderName) {
                llmLogger.info("[Gemma 4] 정확 매치 단축: \(taskTitle, privacy: .public) -> \(c.folderName, privacy: .public)")
                return (c.folderName, 1.0)
            }
        }

        guard let llm = self.llm else {
            llmLogger.warning("[Gemma 4] llm not attached — '일반' 반환")
            return ("일반", 0.0)
        }

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

        // Messages — system 안내 + few-shot (corrections 또는 cold-start seed) + 실제 분류 turn.
        // Gemma 토크나이저가 chat template을 적용 (라이브러리가 내부에서 처리).
        var messages: [CoreMLLLM.Message] = [
            .init(role: .system, content: "할 일의 핵심 키워드(주제 단어)를 보고 가장 의미적으로 가까운 카테고리를 선택하세요. 카테고리 라벨의 위치(A/B/C…)는 매번 다르니 letter 자체가 아니라 카테고리 이름의 의미를 보고 판단해야 합니다. 답은 정확히 대문자 한 글자만.")
        ]

        // Cold-start seed (4개) — corrections 유무와 무관하게 항상 포함.
        // 의도: (a) "대문자 한 글자만" format 학습 (b) 2/3-way 다양 (c) letter↔domain 위치가
        // 매번 다르다(예: "운동"이 A에도 C에도 등장)는 메타 규칙 학습. 사용자 corrections 1개
        // 만으로는 이 모두를 학습하기에 부족 — vault 05-06 측정: 1개 example = 60% / 4개 = 88%.
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
            // 이전 턴의 KV cache 오염 방지 — 매번 fresh state.
            llm.reset()
            let output = try await llm.generate(messages, maxTokens: 1)
            let clean = output.trimmingCharacters(in: .whitespacesAndNewlines)
                              .trimmingCharacters(in: .punctuationCharacters)
                              .uppercased()
            if clean.count == 1, let index = labels.firstIndex(of: clean) {
                let result = usableFolders[index]
                llmLogger.info("[Gemma 4] 분류 성공: \(sanitizedTitle, privacy: .public) -> \(result, privacy: .public) (\(clean, privacy: .public))")
                return (result, 1.0)
            }
            llmLogger.info("[Gemma 4] 매치 안 됨 — '일반'으로 fallback. raw='\(output, privacy: .public)' clean='\(clean, privacy: .public)'")
            return ("일반", 0.0)
        } catch {
            llmLogger.error("[Gemma 4] inference 에러: \(error.localizedDescription, privacy: .public)")
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
