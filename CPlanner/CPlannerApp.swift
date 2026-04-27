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
            ContentView()
        }
    }
}
