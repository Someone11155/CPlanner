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
class LocalLLMService {
    static let shared = LocalLLMService()
    
    private var tokenizer: Tokenizer?
    private var monoModel: MLModel?
    private var isReady = false
    
    private init() {
        Task { await setupMistral() }
    }
    
    private func setupMistral() async {
        print("🚀 [Mistral 7B] 똑똑한 친구 깨우는 중...")
        do {
            guard let resourceURL = Bundle.main.resourceURL else { return }
            self.tokenizer = try await AutoTokenizer.from(modelFolder: resourceURL)
            print("✅ [Mistral 7B] 번역기 준비 끝!")
            
            let configML = MLModelConfiguration()
            configML.computeUnits = .all
            configML.allowLowPrecisionAccumulationOnGPU = true
            
            let modelName = "StatefulMistral7BInstructInt4"
            if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
                self.monoModel = try MLModel(contentsOf: modelURL, configuration: configML)
                print("✅ [Mistral 7B] 두뇌(MONO) 로딩 완료! 준비 끝!")
                self.isReady = true
            } else {
                print("🚨 모델 파일을 못 찾겠어.")
            }
        } catch {
            print("🚨 초기화 실패: \(error)")
        }
    }
    
    private func runPredictionSync(model: MLModel, provider: MLFeatureProvider, state: MLState) throws -> MLFeatureProvider {
        return try model.prediction(from: provider, using: state)
    }
    
    private func getTopK(from logits: MLMultiArray, k: Int = 30) -> [Int] {
        let count = logits.count
        var topTokens = [(index: Int, score: Float)]()
        
        for i in 0..<count {
            topTokens.append((index: i, score: logits[i].floatValue))
        }
        
        topTokens.sort { $0.score > $1.score }
        return topTokens.prefix(k).map { $0.index }
    }
    
    func classifyTask(taskTitle: String, availableFolders: [String]) async -> String {
        guard isReady, let tokenizer = self.tokenizer, let model = self.monoModel else {
            print("⚠️ Mistral이 아직 준비 안 됐어.")
            return "일반"
        }
        
        print("🤖 [Mistral 7B] '\(taskTitle)' 분류 시작!")
        print("📋 모델 입력 정보: \(model.modelDescription.inputDescriptionsByName.keys)")
        print("📋 모델 출력 정보: \(model.modelDescription.outputDescriptionsByName.keys)")
        
        var optionsText = ""
        let labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for (i, folder) in availableFolders.enumerated() {
            if i < labels.count {
                optionsText += "\(labels[i]). \(folder)\n"
            }
        }
        
        // <s>[INST] 형식과 명확한 지시사항, 그리고 마지막에 공백 한 칸을 추가하여 AI가 바로 답변을 시작하도록 유도합니다.
        let prompt = "<s>[INST] You are a category sorter. Reply with EXACTLY ONE letter from the Categories list.\n\nCategories:\n\(optionsText)\nTask: \(taskTitle) [/INST] "
        
        let inputTokens = tokenizer.encode(text: prompt)
        let seqLen = inputTokens.count
        print("🔢 프롬프트 길이: \(seqLen) 토큰")
        
        do {
            // "Cannot retrieve vector from IRValue format int32" 에러가 발생하면 
            // 모델이 Int64(long)를 기대하는 것일 수 있으므로 로그를 보고 타입을 조정해야 할 수 있습니다.
            let inputIdsArray = try MLMultiArray(shape: [1, 1], dataType: .int32) 
            let causalMaskArray = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
            causalMaskArray[0] = 0.0
            
            let state = model.makeState()
            var finalLogits: MLMultiArray?
            
            print("⏳ NPU 가동! 추론 중...")
            
            for i in 0..<seqLen {
                inputIdsArray[0] = NSNumber(value: inputTokens[i])
                let inputs: [String: Any] = ["inputIds": inputIdsArray, "causalMask": causalMaskArray]
                let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
                let prediction = try await model.prediction(from: provider, using: state)
                
                if i == seqLen - 1 {
                    finalLogits = prediction.featureValue(for: "logits")?.multiArrayValue
                }
            }
            
            if let logits = finalLogits {
                let topTokens = getTopK(from: logits, k: 10)
                print("🧠 AI가 생성한 상위 토큰들:")
                
                for token in topTokens {
                    let decodedWord = tokenizer.decode(tokens: [token])
                    // 앞뒤 공백 제거 및 대문자 변환
                    let cleanWord = decodedWord.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
                    print("  - [\(token)]: \"\(decodedWord)\" -> 처리됨: \"\(cleanWord)\"")
                    
                    // A~J 중 하나만 포함된 답변을 찾습니다.
                    if cleanWord.count == 1, let char = cleanWord.first, "ABCDEFGHIJKLMNOPQRSTUVWXYZ".contains(char) {
                        let letter = String(char)
                        if let index = labels.firstIndex(of: letter), index < availableFolders.count {
                            let matchedFolder = availableFolders[index]
                            print("🎯 매칭 성공: \(letter) -> \(matchedFolder)")
                            return matchedFolder
                        }
                    }
                }
            }
            
            print("🤔 적절한 카테고리를 찾지 못했습니다. '일반'으로 분류합니다.")
            return "일반"
            
        } catch {
            print("🚨 추론 중 에러 발생: \(error)")
            return "일반"
        }
    }
    
    func validateFileContext(fileName: String, folderName: String) async -> Bool {
        let lowerFileName = fileName.lowercased()
        let junkExtensions = [".dmg", ".exe", ".mp4", ".zip"]
        if junkExtensions.contains(where: { lowerFileName.hasSuffix($0) }) { return false }
        return lowerFileName.count >= 2
    }
}
