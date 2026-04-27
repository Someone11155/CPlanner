//
//  NLPTestView.swift
//  Cplanner
//
//  Created by iLo on 2026-04-08.
//

import SwiftUI

struct NLPTestView: View {
    @State private var taskTitle: String = ""
    @State private var resultFolder: String = "대기 중..."
    @State private var isProcessing: Bool = false
    
    // 테스트용 폴더 목록
    let mockFolders = ["운영체제", "환경과사회", "디자인", "소프트웨어공학", "희곡교육론"]

    var body: some View {
        VStack(spacing: 20) {
            Text("🧠 Mistral 7B 분류 테스트")
                .font(.largeTitle)
                .bold()
            
            // =========================================================
            // 🔥 새로 추가된 UI: 분류 가능한 카테고리를 가로 스크롤 태그로 보여주기!
            // =========================================================
            VStack(alignment: .leading, spacing: 8) {
                Text("📁 분류 가능한 카테고리")
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .padding(.horizontal)
                
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(mockFolders, id: \.self) { folder in
                            Text(folder)
                                .font(.system(size: 14, weight: .medium))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.blue.opacity(0.15))
                                .foregroundColor(.blue)
                                .cornerRadius(12)
                        }
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.bottom, 10)
            // =========================================================
            
            TextField("할 일을 입력하세요 (예: 그리스 비극 조사)", text: $taskTitle)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding(.horizontal)
            
            Button(action: {
                // UI 업데이트 (분석 중 표시)
                isProcessing = true
                resultFolder = "Mistral이 맥락을 이해하는 중... 🤔"
                
                let currentTitle = taskTitle
                let folders = mockFolders
                
                // 백그라운드 스레드에서 무거운 7B 모델 연산 실행
                Task.detached(priority: .userInitiated) {
                    let result = await LocalLLMService.shared.classifyTask(taskTitle: currentTitle, availableFolders: folders)
                    
                    // 연산이 끝나면 다시 메인 화면으로 돌아와 결과 보여주기
                    await MainActor.run {
                        resultFolder = "📁 배정된 폴더: \(result)"
                        isProcessing = false
                    }
                }
            }) {
                Text(isProcessing ? "분석 중..." : "AI 분류 실행")
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(isProcessing ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .disabled(isProcessing || taskTitle.isEmpty)
            .padding(.horizontal)
            
            Text(resultFolder)
                .font(.title2)
                .foregroundColor(resultFolder.contains("📁") ? .green : .primary)
                .padding()
            
            Spacer()
        }
        .padding(.vertical)
        .frame(width: 450, height: 420) // 태그가 들어갈 공간을 위해 창 크기를 살짝 키웠습니다.
    }
}
