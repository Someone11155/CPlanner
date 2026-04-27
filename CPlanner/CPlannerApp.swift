//
// CplannerApp.swift
// Cplanner
//
// Created by iLo on 2026-03-31.
//

import SwiftUI

@main
struct CplannerApp: App {

    init() {
        // 앱 시작과 동시에 Mistral 7B 모델을 백그라운드에서 미리 로딩.
        // 이후 첫 분류 호출은 await로 로딩 완료를 보장한다.
        Task { await LocalLLMService.shared.preload() }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
