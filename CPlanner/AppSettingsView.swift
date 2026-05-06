//
//  AppSettingsView.swift
//  CPlanner
//
//  macOS 표준 Preferences/Settings 윈도우. CPlannerApp.swift의 `Settings { ... }` scene이
//  이 뷰를 띄우며, 메뉴바 `CPlanner > Settings...` 또는 ⌘, 단축키로 접근.
//
//  ContentView 안의 톱니 아이콘으로 띄우는 SettingsView (감시 폴더 + 벤치마크)와는 별개 —
//  여기는 "앱 자체"에 대한 설정 (신뢰도 임계값, TipBar 옵션, 외부 링크 등).
//
//  영속화는 @AppStorage 사용. 키 명명 규약: `cplanner.app.<area>.<option>`.
//

import SwiftUI

struct AppSettingsView: View {
    var body: some View {
        TabView {
            GeneralSettingsTab()
                .tabItem { Label("일반", systemImage: "gearshape") }
            ClassificationSettingsTab()
                .tabItem { Label("분류", systemImage: "wand.and.stars") }
            AboutTab()
                .tabItem { Label("정보", systemImage: "info.circle") }
        }
        .frame(width: 480, height: 320)
    }
}

// MARK: - 일반 (General)

struct GeneralSettingsTab: View {
    @AppStorage("cplanner.app.tip.enabled") private var tipEnabled: Bool = true
    @AppStorage("cplanner.app.tip.intervalSeconds") private var tipIntervalSeconds: Double = 7

    var body: some View {
        Form {
            Section {
                Toggle("앱 하단에 팁 표시", isOn: $tipEnabled)
                HStack {
                    Text("팁 순환 주기")
                    Slider(value: $tipIntervalSeconds, in: 3...20, step: 1) {
                        Text("팁 순환 주기")
                    } minimumValueLabel: {
                        Text("3초").font(.caption2).foregroundColor(.secondary)
                    } maximumValueLabel: {
                        Text("20초").font(.caption2).foregroundColor(.secondary)
                    }
                    .disabled(!tipEnabled)
                    Text("\(Int(tipIntervalSeconds))초")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 36, alignment: .trailing)
                }
            } header: {
                Text("팁 바")
            } footer: {
                Text("앱 하단에 분류·캘린더·학습 관련 팁이 순환 표시돼요. 끄면 공간이 사라져요.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - 분류 (Classification)

struct ClassificationSettingsTab: View {
    @AppStorage("cplanner.app.classification.confidenceThreshold") private var confidenceThreshold: Double = 0.75
    @AppStorage("cplanner.app.classification.exactMatchShortcut") private var exactMatchShortcut: Bool = true

    var body: some View {
        Form {
            Section {
                HStack {
                    Text("신뢰도 임계값")
                    Slider(value: $confidenceThreshold, in: 0.5...0.95, step: 0.05) {
                        Text("신뢰도 임계값")
                    } minimumValueLabel: {
                        Text("50%").font(.caption2).foregroundColor(.secondary)
                    } maximumValueLabel: {
                        Text("95%").font(.caption2).foregroundColor(.secondary)
                    }
                    Text("\(Int(confidenceThreshold * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 40, alignment: .trailing)
                }
            } header: {
                Text("자동 분류")
            } footer: {
                Text("Gemma 4 E2B로 전환된 후 분류 신뢰도가 이진(binary)으로 단순화됐어요 — 깨끗한 단일 알파벳 매치는 100%, 그 외는 '일반'(0%). 이 슬라이더는 향후 logit-level 신뢰도 복원 시 재활성화 예정.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Section {
                Toggle("같은 제목 분류 이력 즉시 재사용 (정확매치 단축)", isOn: $exactMatchShortcut)
            } header: {
                Text("성능 최적화")
            } footer: {
                Text("켜져 있으면 동일한 제목의 task를 다시 입력했을 때 LLM 호출 없이 이전 분류 결과를 즉시 재사용해요. 거의 모든 경우 켜두는 게 좋아요. (이 옵션은 향후 구현 예정)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - 정보 (About)

struct AboutTab: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "calendar.badge.clock")
                .font(.system(size: 56))
                .foregroundColor(.accentColor)
                .padding(.top, 24)
            Text("CPlanner")
                .font(.title)
                .fontWeight(.bold)
            Text("대학생을 위한 온디바이스 AI 플래너")
                .font(.callout)
                .foregroundColor(.secondary)
            Divider().padding(.horizontal, 64)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("버전")
                    Spacer()
                    Text(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "—")
                        .foregroundColor(.secondary)
                }
                HStack {
                    Text("빌드")
                    Spacer()
                    Text(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—")
                        .foregroundColor(.secondary)
                }
                HStack {
                    Text("플랫폼")
                    Spacer()
                    Text("macOS 15+ / iOS 18+")
                        .foregroundColor(.secondary)
                }
            }
            .font(.callout)
            .frame(maxWidth: 280)
            Spacer()
        }
        .padding()
    }
}
