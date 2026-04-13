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
        
        var optionsText = ""
        let labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for (i, folder) in availableFolders.enumerated() {
            if i < labels.count {
                optionsText += "\(labels[i]). \(folder)\n"
            }
        }
        
        let prompt = """
        <s>[INST] You are a category sorter. Reply with EXACTLY ONE letter from the Categories list.
        
        Categories:
        A. 운동
        B. 공부
        Task: 헬스장 가기 [/INST]A</s>[INST] Categories:
        \(optionsText)
        Task: \(taskTitle) [/INST]
        """
        
        let inputTokens = tokenizer.encode(text: prompt)
        let seqLen = inputTokens.count
        print("🔢 프롬프트 길이: \(seqLen) 토큰")
        
        do {
            let inputIdsArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            let causalMaskArray = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
            causalMaskArray[0] = 0.0
            
            let state = model.makeState()
            var finalLogits: MLMultiArray?
            
            print("⏳ NPU 가동! 질문을 친구에게 던지는 중...")
            
            for i in 0..<seqLen {
                inputIdsArray[0] = NSNumber(value: inputTokens[i])
                let inputs: [String: Any] = ["inputIds": inputIdsArray, "causalMask": causalMaskArray]
                let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
                let prediction = try runPredictionSync(model: model, provider: provider, state: state)
                
                if i == seqLen - 1 {
                    finalLogits = prediction.featureValue(for: "logits")?.multiArrayValue
                }
                
                if i % 10 == 0 {
                    await Task.yield()
                }
            }
            
            print("🧠 친구의 생각 분석 중...")
            
            if let logits = finalLogits {
                let topTokens = getTopK(from: logits, k: 30)
                
                for token in topTokens {
                    let decodedWord = tokenizer.decode(tokens: [token])
                    
                    // 🔥 방어막 1: 아예 특수 토큰(</s>, <bbox> 등) 문자열이 껴있으면 분석하기도 전에 버립니다!
                    if decodedWord.contains("<") || decodedWord.contains(">") { continue }
                    
                    let upperWord = decodedWord.uppercased()
                    let lettersOnly = upperWord.filter { "ABCDEFGHIJKLMNOPQRSTUVWXYZ".contains($0) }
                    
                    if lettersOnly.count == 1 {
                        let cleanLetter = String(lettersOnly)
                        
                        // 🔥 방어막 2: 뽑아낸 글자가 진짜 보기 배열(A, B, C...) 안에 존재하는지 검사!
                        if labels.contains(cleanLetter) {
                            print("🎯 AI의 속마음에서 정답 알파벳을 찾아냈어: [\(cleanLetter)]")
                            
                            if let index = labels.firstIndex(of: cleanLetter), index < availableFolders.count {
                                let matchedFolder = availableFolders[index]
                                print("✅ 최종 폴더 배정 완료: \(matchedFolder)")
                                return matchedFolder
                            }
                            break
                        }
                    }
                }
            }
            
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
