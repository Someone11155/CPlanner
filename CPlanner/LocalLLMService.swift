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

@available(macOS 15.0, iOS 18.0, *)
final class LocalLLMService: @unchecked Sendable {
    static let shared = LocalLLMService()
    
    private var tokenizer: Tokenizer?
    private var monoModel: MLModel?
    private var isReady = false
    
    private init() {
        Task { await setupMistral() }
    }
    
    private func setupMistral() async {
        print("⏳ [Mistral 7B] 모델 로딩 시작...")
        do {
            guard let resourceURL = Bundle.main.resourceURL else { return }
            self.tokenizer = try await AutoTokenizer.from(modelFolder: resourceURL)
            
            let configML = MLModelConfiguration()
            configML.computeUnits = .all
            
            let modelName = "StatefulMistral7BInstructInt4"
            if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
                self.monoModel = try MLModel(contentsOf: modelURL, configuration: configML)
                print("✅ [Mistral 7B] 로딩 완료!")
                self.isReady = true
            } else {
                print("🚨 모델 파일을 찾을 수 없습니다 (mlmodelc).")
            }
        } catch {
            print("🚨 초기화 실패: \(error)")
        }
    }
    
    func classifyTask(taskTitle: String, availableFolders: [String]) async -> String {
        guard isReady, let tokenizer = self.tokenizer, let model = self.monoModel else {
            return "일반"
        }
        
        // 1. 프롬프트 생성 (Few-shot 포함)
        var optionsText = ""
        let labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for (i, folder) in availableFolders.enumerated() {
            if i < labels.count { optionsText += "\(labels[i]). \(folder)\n" }
        }
        
        // Mistral Instruct v3 포맷 준수
        let prompt = """
        [INST] 분류 전문가로서 할 일을 카테고리 중 하나로 분류하세요. 대문자 알파벳 한 글자만 답하세요.
        카테고리:
        A. 운동
        B. 공부
        할 일: 헬스장 가기 [/INST] A </s> [INST] 카테고리:
        \(optionsText)
        할 일: \(taskTitle) [/INST] 
        """
        
        let inputTokens = tokenizer.encode(text: prompt)
        
        do {
            let state = model.makeState()
            var finalLogits: MLMultiArray?
            
            // 2. NPU 추론 실행
            for (i, tokenID) in inputTokens.enumerated() {
                let inputIds = MLShapedArray<Int32>(scalars: [Int32(tokenID)], shape: [1, 1])
                let causalMask = MLShapedArray<Float16>(scalars: [0.0], shape: [1, 1, 1, 1])
                
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
            
            // 3. 결과 분석 및 추출
            if let logits = finalLogits {
                let topTokens = getTopK(from: logits, k: 10)
                
                for token in topTokens {
                    let word = tokenizer.decode(tokens: [token])
                    let clean = word.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                                    .trimmingCharacters(in: CharacterSet.punctuationCharacters)
                                    .uppercased()
                    
                    if clean.count == 1, labels.contains(clean) {
                        if let index = labels.firstIndex(of: clean), index < availableFolders.count {
                            let result = availableFolders[index]
                            print("🎯 [Mistral] 분류 성공: \(taskTitle) -> \(result) (\(clean))")
                            return result
                        }
                    }
                }
            }
        } catch {
            print("🚨 추론 중 에러: \(error)")
        }
        
        print("⚠️ [Mistral] 분류 실패: '일반'으로 반환")
        return "일반"
    }
    
    private func getTopK(from logits: MLMultiArray, k: Int) -> [Int] {
        var topTokens = [(index: Int, score: Float)]()
        for i in 0..<logits.count {
            topTokens.append((index: i, score: logits[i].floatValue))
        }
        topTokens.sort { $0.score > $1.score }
        return topTokens.prefix(k).map { $0.index }
    }
    
    func validateFileContext(fileName: String, folderName: String) async -> Bool {
        let lower = fileName.lowercased()
        let junk = [".dmg", ".exe", ".mp4", ".zip"]
        return !junk.contains(where: { lower.hasSuffix($0) }) && lower.count >= 2
    }
}
