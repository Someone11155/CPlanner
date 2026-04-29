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
        // 모델/토크나이저 설치 여부를 확인. 설치돼 있으면 자동으로 백그라운드 로딩이 트리거되고,
        // 없으면 ContentView에서 다운로드 alert를 띄운다.
        ModelInstaller.shared.checkInstallation()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .defaultSize(width: 1000, height: 750)
    }
}
