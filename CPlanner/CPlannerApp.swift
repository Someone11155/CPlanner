//
// CplannerApp.swift
// Cplanner
//
// Created by iLo on 2026-03-31.
//

import SwiftUI

@main
struct CplannerApp: App {
    
    // 앱이 켜지자마자 백그라운드에서 무거운 Gemma 모델을 미리 로딩하도록 지시합니다.
    init() {
        _ = LocalLLMService.shared
    }
    
    var body: some Scene {
        WindowGroup {
            // 원래 있던 ContentView() 대신 우리가 만든 테스트 화면을 띄웁니다!
            // (나중에 테스트가 끝나면 다시 ContentView()로 돌려놓으시면 됩니다.)
            NLPTestView()
        }
    }
}
