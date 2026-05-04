//
// ContentView.swift
// Cplanner
//
// Created by iLo on 2026-03-26.
//

import SwiftUI
import Combine
import CoreML
import EventKit
import os

private let taskManagerLogger = Logger(subsystem: "com.cplanner", category: "TaskManager")

// --- 1. 모델 정의 ---
struct TaskItem: Identifiable, Codable, Equatable {
    var id = UUID()
    var title: String
    var isCompleted: Bool = false
    var targetFolder: String
    var date: Date
    var eventIdentifier: String? = nil
}

struct Correction: Codable, Equatable, Sendable {
    var taskTitle: String
    var folderName: String
}

struct FolderRule: Identifiable, Codable {
    var id = UUID()
    var folderName: String
    var bookmark: Data

    /// Resolves the security-scoped bookmark.
    /// Caller MUST call `startAccessingSecurityScopedResource()` on the
    /// returned URL before reading the folder, and the matching
    /// `stopAccessingSecurityScopedResource()` once finished.
    func resolveURL() throws -> (url: URL, isStale: Bool) {
        var isStale = false
        let url = try URL(
            resolvingBookmarkData: bookmark,
            options: [.withSecurityScope],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        )
        return (url, isStale)
    }
}

// --- 2. 메인 매니저 ---
class TaskManager: ObservableObject {
    private static let tasksKey = "cplanner.tasks"
    private static let folderRulesKey = "cplanner.folderRules"
    private static let correctionsKey = "cplanner.corrections"

    @Published var tasks: [TaskItem] = [] {
        didSet { persist(tasks, forKey: Self.tasksKey) }
    }
    @Published var folderRules: [FolderRule] = [] {
        didSet { persist(folderRules, forKey: Self.folderRulesKey) }
    }
    @Published var corrections: [Correction] = [] {
        didSet { persist(corrections, forKey: Self.correctionsKey) }
    }
    @Published var lastCalendarError: String?

    private let eventStore = EKEventStore()
    private var monitors: [UUID: (monitor: FolderMonitor, scopedURL: URL)] = [:]
    private var calendarAccessGranted = false
    private var cplannerCalendar: EKCalendar?
    private var isApplyingLocalChange = false
    nonisolated(unsafe) private var calendarChangeObserver: NSObjectProtocol?

    init() {
        // Load persisted state. didSet observers do NOT fire during init,
        // so these assignments don't write back to UserDefaults.
        if let data = UserDefaults.standard.data(forKey: Self.tasksKey),
           let decoded = try? JSONDecoder().decode([TaskItem].self, from: data) {
            self.tasks = decoded
        }
        if let data = UserDefaults.standard.data(forKey: Self.folderRulesKey),
           let decoded = try? JSONDecoder().decode([FolderRule].self, from: data) {
            self.folderRules = decoded
            for rule in self.folderRules {
                startMonitoring(rule: rule)
            }
        }
        if let data = UserDefaults.standard.data(forKey: Self.correctionsKey),
           let decoded = try? JSONDecoder().decode([Correction].self, from: data) {
            self.corrections = decoded
        }
        Task { [weak self] in
            await self?.bootstrapCalendar()
        }
    }

    deinit {
        if let observer = calendarChangeObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        for entry in monitors.values {
            entry.scopedURL.stopAccessingSecurityScopedResource()
        }
    }

    private func persist<T: Encodable>(_ value: T, forKey key: String) {
        do {
            let data = try JSONEncoder().encode(value)
            UserDefaults.standard.set(data, forKey: key)
        } catch {
            taskManagerLogger.error("Failed to persist \(key, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    // 할 일 추가 (AI 분류 적용)
    func addTask(title: String, date: Date) {
        // 1) 즉시 placeholder 폴더로 UI 반영
        let placeholder = TaskItem(title: title, targetFolder: "분류 중…", date: date)
        let taskID = placeholder.id
        tasks.append(placeholder)

        // 2) 백그라운드: 분류 + 캘린더 저장 + task 갱신
        let folderNames = self.folderRules.map { $0.folderName }
        let activeCorrections = Array(self.corrections.filter { folderNames.contains($0.folderName) }.suffix(5))
        Task { [weak self] in
            guard let self else { return }
            let detectedFolder = await LocalLLMService.shared.classifyTask(taskTitle: title, availableFolders: folderNames, corrections: activeCorrections)
            if let idx = self.tasks.firstIndex(where: { $0.id == taskID }), self.tasks[idx].targetFolder == "분류 중…" {
                // 사용자가 그동안 폴더를 직접 골랐으면 자동 분류 결과로 덮어쓰지 않음
                self.tasks[idx].targetFolder = detectedFolder
            }

            if self.calendarAccessGranted, let cal = self.cplannerCalendar {
                let event = EKEvent(eventStore: self.eventStore)
                event.title = title
                let day = Calendar.current.startOfDay(for: date)
                event.startDate = day
                event.endDate = day
                event.isAllDay = true
                event.alarms = [] // 캘린더 source의 기본 alarm이 URL을 포함해 sandbox 경고 발생 — 명시적 클리어
                event.calendar = cal

                self.isApplyingLocalChange = true
                do {
                    try self.eventStore.save(event, span: .thisEvent)
                    if let idx = self.tasks.firstIndex(where: { $0.id == taskID }) {
                        self.tasks[idx].eventIdentifier = event.eventIdentifier
                    }
                    self.lastCalendarError = nil
                } catch {
                    self.lastCalendarError = "캘린더 저장 실패: \(error.localizedDescription)"
                    taskManagerLogger.error("Calendar save failed: \(error.localizedDescription, privacy: .public)")
                }
                DispatchQueue.main.async { [weak self] in self?.isApplyingLocalChange = false }
            }
        }
    }

    /// 사용자가 직접 폴더를 변경 — task에 반영 + 같은 제목의 기존 correction 대체 후 추가.
    /// 다음 분류부터 in-context few-shot 예시로 사용된다.
    func userPickedFolder(taskID: UUID, folder: String) {
        guard let idx = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        tasks[idx].targetFolder = folder
        let title = tasks[idx].title
        corrections.removeAll { $0.taskTitle == title }
        corrections.append(Correction(taskTitle: title, folderName: folder))
        // 무한 누적 방지
        if corrections.count > 50 {
            corrections.removeFirst(corrections.count - 50)
        }
    }

    func deleteTask(id: UUID) {
        guard let idx = tasks.firstIndex(where: { $0.id == id }) else { return }
        let task = tasks[idx]
        if let eid = task.eventIdentifier,
           calendarAccessGranted,
           let event = eventStore.event(withIdentifier: eid) {
            isApplyingLocalChange = true
            do {
                try eventStore.remove(event, span: .thisEvent)
            } catch {
                taskManagerLogger.error("Calendar delete failed: \(error.localizedDescription, privacy: .public)")
            }
            DispatchQueue.main.async { [weak self] in self?.isApplyingLocalChange = false }
        }
        tasks.remove(at: idx)
    }

    // 감시 규칙 추가
    func addRule(name: String, bookmark: Data) {
        let newRule = FolderRule(folderName: name, bookmark: bookmark)
        folderRules.append(newRule)
        startMonitoring(rule: newRule)
    }

    func deleteRule(id: UUID) {
        folderRules.removeAll { $0.id == id }
        if let entry = monitors[id] {
            entry.monitor.stopMonitoring()
            entry.scopedURL.stopAccessingSecurityScopedResource()
        }
        monitors.removeValue(forKey: id)
    }

    // 폴더 감시 시작
    private func startMonitoring(rule: FolderRule) {
        do {
            let (url, isStale) = try rule.resolveURL()
            if isStale {
                taskManagerLogger.warning("Bookmark for \(rule.folderName, privacy: .public) is stale; user should reselect the folder.")
            }
            guard url.startAccessingSecurityScopedResource() else {
                taskManagerLogger.error("Failed to start security-scoped access for \(url.path, privacy: .public)")
                return
            }
            let monitor = FolderMonitor(url: url)
            let ruleID = rule.id
            monitor.folderDidChange = { [weak self] in
                self?.handleNewFileDetected(ruleID: ruleID)
            }
            monitor.startMonitoring()
            monitors[ruleID] = (monitor: monitor, scopedURL: url)
        } catch {
            taskManagerLogger.error("Failed to resolve bookmark for \(rule.folderName, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    // 파일 감지 시 AI 검증 로직 실행
    private func handleNewFileDetected(ruleID: UUID) {
        guard let rule = folderRules.first(where: { $0.id == ruleID }),
              let scopedURL = monitors[ruleID]?.scopedURL else { return }

        Task { [weak self] in
            guard let self else { return }
            let contents: [URL]
            do {
                contents = try FileManager.default.contentsOfDirectory(
                    at: scopedURL,
                    includingPropertiesForKeys: [.contentModificationDateKey],
                    options: [.skipsHiddenFiles]
                )
            } catch {
                taskManagerLogger.error("Failed to enumerate folder \(scopedURL.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
                return
            }

            let newest = contents.max { a, b in
                let aDate = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let bDate = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return aDate < bDate
            }
            guard let fileName = newest?.lastPathComponent else { return }

            let isValid = LocalLLMService.shared.validateFileContext(fileName: fileName, folderName: rule.folderName)
            guard isValid else { return }

            if let idx = self.tasks.firstIndex(where: { $0.targetFolder == rule.folderName && !$0.isCompleted }) {
                self.tasks[idx].isCompleted = true
            }
        }
    }

    private func bootstrapCalendar() async {
        do {
            let granted = try await eventStore.requestFullAccessToEvents()
            calendarAccessGranted = granted
            guard granted else {
                lastCalendarError = "캘린더 접근 권한이 거부되었습니다."
                return
            }
            guard let cal = getOrCreateCPlannerCalendar() else { return }
            cplannerCalendar = cal
            subscribeToCalendarChanges()
            await syncFromCalendar()
        } catch {
            lastCalendarError = "캘린더 권한 요청 실패: \(error.localizedDescription)"
            taskManagerLogger.error("Calendar access request failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func getOrCreateCPlannerCalendar() -> EKCalendar? {
        if let existing = eventStore.calendars(for: .event).first(where: { $0.title == "CPlanner" }) {
            return existing
        }
        let source: EKSource? = eventStore.sources.first(where: { $0.sourceType == .calDAV })
            ?? eventStore.sources.first(where: { $0.sourceType == .local })
        guard let source else {
            lastCalendarError = "캘린더 source를 찾을 수 없습니다."
            taskManagerLogger.error("No suitable EKSource for CPlanner calendar")
            return nil
        }
        let cal = EKCalendar(for: .event, eventStore: eventStore)
        cal.title = "CPlanner"
        cal.source = source
        cal.cgColor = NSColor.systemBlue.cgColor
        do {
            try eventStore.saveCalendar(cal, commit: true)
            return cal
        } catch {
            lastCalendarError = "CPlanner 캘린더 생성 실패: \(error.localizedDescription)"
            taskManagerLogger.error("Failed to create CPlanner calendar: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    private func subscribeToCalendarChanges() {
        calendarChangeObserver = NotificationCenter.default.addObserver(
            forName: .EKEventStoreChanged,
            object: eventStore,
            queue: nil
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.isApplyingLocalChange { return }
                await self.syncFromCalendar()
            }
        }
    }

    private func syncFromCalendar() async {
        guard calendarAccessGranted, let cal = cplannerCalendar else { return }
        let calendar = Calendar.current
        let now = Date()
        let start = calendar.date(byAdding: .month, value: -6, to: now) ?? now
        let end = calendar.date(byAdding: .month, value: 24, to: now) ?? now
        let predicate = eventStore.predicateForEvents(withStart: start, end: end, calendars: [cal])
        let events = eventStore.events(matching: predicate)

        var resultTasks = tasks
        var seenEventIds = Set<String>()
        var taskIdxByEventId: [String: Int] = [:]
        for (i, t) in resultTasks.enumerated() {
            if let eid = t.eventIdentifier { taskIdxByEventId[eid] = i }
        }

        for event in events {
            guard !event.hasRecurrenceRules else {
                taskManagerLogger.warning("Skipping recurring event: \(event.title ?? "(제목 없음)", privacy: .public)")
                continue
            }
            guard let eid = event.eventIdentifier else { continue }
            seenEventIds.insert(eid)
            let normalizedDate = calendar.startOfDay(for: event.startDate)
            let title = event.title ?? "(제목 없음)"

            if let idx = taskIdxByEventId[eid] {
                if resultTasks[idx].title != title { resultTasks[idx].title = title }
                if !calendar.isDate(resultTasks[idx].date, inSameDayAs: normalizedDate) {
                    resultTasks[idx].date = normalizedDate
                }
            } else {
                let folder: String
                if folderRules.count >= 2 {
                    let names = folderRules.map { $0.folderName }
                    folder = await LocalLLMService.shared.classifyTask(taskTitle: title, availableFolders: names)
                } else {
                    folder = "(미분류)"
                }
                let newTask = TaskItem(title: title, targetFolder: folder, date: normalizedDate, eventIdentifier: eid)
                resultTasks.append(newTask)
            }
        }

        resultTasks.removeAll { task in
            guard let eid = task.eventIdentifier else { return false }
            return !seenEventIds.contains(eid)
        }

        if resultTasks != tasks {
            tasks = resultTasks
        }
    }
}

// --- 3. 커스텀 캘린더 뷰 ---
struct CustomCalendarView: View {
    @ObservedObject var taskManager: TaskManager
    @Binding var selectedDate: Date

    private let calendar = Calendar.current
    private let calendarHeaderColor = Color(NSColor.windowBackgroundColor)
    private let selectionColor = Color.blue
    let weekdayNames = ["일", "월", "화", "수", "목", "금", "토"]

    func generateDaysInMonth() -> [Date] {
        guard let monthRange = calendar.range(of: .day, in: .month, for: selectedDate),
              let firstOfMonth = calendar.date(from: calendar.dateComponents([.year, .month], from: selectedDate)) else { return [] }
        let firstWeekday = calendar.component(.weekday, from: firstOfMonth)
        let daysToPrepend = firstWeekday - 1
        var days: [Date] = []
        for i in 0..<daysToPrepend {
            if let date = calendar.date(byAdding: .day, value: -i-1, to: firstOfMonth) { days.append(date) }
        }
        days.reverse()
        for i in 0..<monthRange.count {
            if let date = calendar.date(byAdding: .day, value: i, to: firstOfMonth) { days.append(date) }
        }
        return days
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(selectedDate, formatter: DateFormatter.monthYearFormatter).font(.title3).fontWeight(.bold)
                Spacer()
                HStack(spacing: 5) {
                    Button(action: { changeMonth(value: -1) }) { Image(systemName: "chevron.left") }
                    Button(action: { changeMonth(value: 1) }) { Image(systemName: "chevron.right") }
                }.buttonStyle(.plain).font(.title3)
            }.padding().background(calendarHeaderColor)

            Divider()
            Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(weekdayNames, id: \.self) { weekday in
                        Text(weekday).font(.caption).fontWeight(.medium)
                            .foregroundColor(weekday == "일" ? .red : (weekday == "토" ? .blue : .gray))
                            .frame(maxWidth: .infinity).frame(height: 30)
                    }
                }.background(calendarHeaderColor)
            }
            Divider()

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 0), count: 7), spacing: 0) {
                ForEach(generateDaysInMonth(), id: \.self) { date in
                    let weekday = calendar.component(.weekday, from: date)
                    let isWeekend = weekday == 1 || weekday == 7
                    let isSelectedDay = calendar.isDate(date, inSameDayAs: selectedDate)
                    let inSameMonth = calendar.isDate(date, equalTo: selectedDate, toGranularity: .month)
                    VStack {
                        ZStack {
                            if isSelectedDay {
                                Circle().fill(selectionColor).frame(width: 30, height: 30)
                            }
                            Text("\(calendar.component(.day, from: date))")
                                .font(.system(size: 14))
                                .fontWeight(isSelectedDay ? .bold : .medium)
                                .foregroundColor(inSameMonth ? (isSelectedDay ? .white : .primary) : .secondary.opacity(0.5))
                        }
                        if hasTasks(date: date) {
                            Circle().fill(inSameMonth ? (isSelectedDay ? .white : .blue) : .secondary.opacity(0.5)).frame(width: 4, height: 4)
                        }
                    }
                    .frame(height: 50).frame(maxWidth: .infinity)
                    .background(isWeekend ? Color.gray.opacity(0.06) : Color.clear)
                    .overlay(Rectangle().stroke(Color.gray.opacity(0.1), lineWidth: 0.5))
                    .contentShape(Rectangle())
                    .onTapGesture { self.selectedDate = date }
                }
            }.background(Color(NSColor.controlBackgroundColor))

            Divider()
            Spacer()
        }.background(Color(NSColor.controlBackgroundColor))
    }

    func changeMonth(value: Int) {
        if let newDate = calendar.date(byAdding: .month, value: value, to: selectedDate) { selectedDate = newDate }
    }
    func hasTasks(date: Date) -> Bool { taskManager.tasks.contains { calendar.isDate($0.date, inSameDayAs: date) } }
}

// --- 3-1. 하단 팁 바 ---
struct TipBar: View {
    static let tips: [String] = [
        "분류 결과를 수정하려면 폴더 아이콘을 클릭하세요",
        "수동으로 분류하면 다음 자동 분류의 학습 예시로 사용돼요",
        "캘린더에서 추가한 일정도 자동으로 동기화돼요",
        "AI 분류는 인터넷 없이 기기 안에서 동작해요",
        "설정에서 폴더 감시 규칙을 추가할 수 있어요",
        "감시 폴더에 파일이 들어오면 할 일이 자동으로 완료돼요",
        "달력의 작은 점은 그날 할 일이 있다는 뜻이에요",
        "할 일을 우클릭하면 삭제할 수 있어요"
    ]

    @State private var currentTip: String = TipBar.tips.randomElement() ?? ""
    private let timer = Timer.publish(every: 10, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 6) {
            Text("💡")
            Text(currentTip)
                .font(.caption)
                .foregroundColor(.secondary)
                .id(currentTip)
                .transition(.opacity)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.windowBackgroundColor))
        .onReceive(timer) { _ in
            guard TipBar.tips.count > 1 else { return }
            var next = TipBar.tips.randomElement() ?? currentTip
            while next == currentTip {
                next = TipBar.tips.randomElement() ?? currentTip
            }
            withAnimation(.easeInOut(duration: 0.3)) {
                currentTip = next
            }
        }
    }
}

// --- 4. 메인 UI 화면 ---
struct ContentView: View {
    @StateObject private var taskManager = TaskManager()
    @StateObject private var modelInstaller = ModelInstaller.shared
    @State private var selectedDate = Date()
    @State private var isAddingTask = false
    @State private var showSettings = false
    @State private var newTaskTitle = ""

    var body: some View {
        VStack(spacing: 0) {
        HStack(spacing: 0) {
            // 좌측 캘린더
            CustomCalendarView(taskManager: taskManager, selectedDate: $selectedDate)
                .frame(width: 300)

            Divider()

            // 우측 할 일 목록
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("\(selectedDate.formatted(.dateTime.month().day())) 과제 목록").font(.title).fontWeight(.heavy)
                    Spacer()
                    Button(action: { showSettings.toggle() }) { Image(systemName: "gearshape.fill").font(.title2).foregroundColor(.secondary) }.buttonStyle(.plain).padding(.trailing, 10)
                    Button(action: { withAnimation { isAddingTask.toggle() } }) { Image(systemName: isAddingTask ? "xmark.circle.fill" : "plus.circle.fill").font(.title).foregroundColor(isAddingTask ? .gray : .blue) }.buttonStyle(.plain)
                }.padding(.horizontal).padding(.top, 25).padding(.bottom, 15)

                if let calendarError = taskManager.lastCalendarError {
                    Text("⚠️ \(calendarError)")
                        .font(.caption)
                        .foregroundColor(.orange)
                        .padding(.horizontal)
                        .padding(.bottom, 5)
                }

                if isAddingTask {
                    VStack(spacing: 12) {
                        TextField("할 일 제목 (예: 운영체제 과제)", text: $newTaskTitle)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit {
                                if !newTaskTitle.isEmpty { taskManager.addTask(title: newTaskTitle, date: selectedDate); newTaskTitle = ""; withAnimation { isAddingTask = false } }
                            }
                        HStack {
                            Text("💡 AI가 맥락을 분석하여 폴더를 자동 지정합니다.").font(.caption).foregroundColor(.secondary)
                            Spacer()
                            Button("완료") {
                                if !newTaskTitle.isEmpty { taskManager.addTask(title: newTaskTitle, date: selectedDate); newTaskTitle = ""; withAnimation { isAddingTask = false } }
                            }.buttonStyle(.borderedProminent)
                        }
                    }.padding().background(Color(NSColor.windowBackgroundColor)).cornerRadius(10).padding(.horizontal).padding(.bottom, 10)
                }

                List {
                    ForEach(taskManager.tasks.filter { Calendar.current.isDate($0.date, inSameDayAs: selectedDate) }) { task in
                        if let index = taskManager.tasks.firstIndex(where: { $0.id == task.id }) {
                            HStack(spacing: 10) {
                                Image(systemName: taskManager.tasks[index].isCompleted ? "checkmark.circle.fill" : "circle")
                                    .foregroundColor(taskManager.tasks[index].isCompleted ? .blue : .gray).font(.title3)
                                    .onTapGesture { taskManager.tasks[index].isCompleted.toggle() }
                                Text(task.title)
                                    .strikethrough(taskManager.tasks[index].isCompleted)
                                    .foregroundColor(taskManager.tasks[index].isCompleted ? .gray : .primary)
                                    .font(.headline)
                                    .lineLimit(1)
                                Spacer()
                                Text(task.targetFolder)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .lineLimit(1)
                                Menu {
                                    ForEach(taskManager.folderRules) { rule in
                                        Button(rule.folderName) { taskManager.userPickedFolder(taskID: task.id, folder: rule.folderName) }
                                    }
                                    if taskManager.folderRules.isEmpty {
                                        Text("폴더 규칙이 없습니다 — 설정에서 추가").foregroundColor(.secondary)
                                    }
                                } label: {
                                    Image(systemName: "folder")
                                        .font(.title3)
                                        .foregroundColor(.secondary)
                                }
                                .menuStyle(.borderlessButton)
                                .menuIndicator(.hidden)
                                .fixedSize()
                                .help("분류 폴더 변경")
                            }
                            .padding(.vertical, 6)
                            .contextMenu { Button(role: .destructive) { taskManager.deleteTask(id: task.id) } label: { Label("삭제", systemImage: "trash") } }
                        }
                    }
                }.listStyle(.inset)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity).background(Color(NSColor.textBackgroundColor))
        }
        Divider()
        TipBar()
        }
        .sheet(isPresented: $showSettings) { SettingsView(taskManager: taskManager) }
        .alert("Mistral 모델 다운로드 필요", isPresented: needsDownloadBinding) {
            Button("다운로드 (~ 4 GB)") { modelInstaller.startDownload() }
            Button("나중에", role: .cancel) { modelInstaller.skip() }
        } message: {
            Text("Mistral 7B 모델과 토크나이저가 설치돼 있지 않아 폴더 자동 분류가 비활성 상태입니다.\nHuggingFace에서 받으시겠습니까?")
        }
        .sheet(isPresented: isDownloadingBinding) {
            DownloadProgressView().interactiveDismissDisabled()
        }
        .alert("다운로드 실패", isPresented: hasFailureBinding) {
            Button("재시도") { modelInstaller.startDownload() }
            Button("닫기", role: .cancel) { modelInstaller.skip() }
        } message: {
            Text(failureMessage)
        }
    }

    private var needsDownloadBinding: Binding<Bool> {
        Binding(
            get: { if case .needsDownload = modelInstaller.state { return true }; return false },
            set: { _ in }
        )
    }
    private var isDownloadingBinding: Binding<Bool> {
        Binding(
            get: {
                switch modelInstaller.state {
                case .downloading, .compiling: return true
                default: return false
                }
            },
            set: { _ in }
        )
    }
    private var hasFailureBinding: Binding<Bool> {
        Binding(
            get: { if case .failed = modelInstaller.state { return true }; return false },
            set: { _ in }
        )
    }
    private var failureMessage: String {
        if case .failed(let msg) = modelInstaller.state { return msg }
        return ""
    }
}

// --- 4-1. 모델 다운로드/컴파일 진행 시트 ---
struct DownloadProgressView: View {
    @ObservedObject private var installer = ModelInstaller.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            switch installer.state {
            case .downloading(let progress, let status):
                Text("Mistral 모델 다운로드 중").font(.title2).fontWeight(.bold)
                ProgressView(value: progress).progressViewStyle(.linear)
                Text(status).font(.caption).foregroundColor(.secondary).lineLimit(2)
                Text("\(Int(progress * 100))% — 창을 닫지 마세요.").font(.caption2).foregroundColor(.secondary)
            case .compiling:
                Text("모델 컴파일 중").font(.title2).fontWeight(.bold)
                ProgressView().progressViewStyle(.linear)
                Text("CoreML이 .mlpackage를 .mlmodelc로 컴파일하고 있습니다 (수십 초~수 분 소요).").font(.caption).foregroundColor(.secondary)
            default:
                ProgressView().progressViewStyle(.linear)
            }
        }
        .padding(24)
        .frame(width: 480)
    }
}

// --- 5. 설정 뷰 ---
struct SettingsView: View {
    @ObservedObject var taskManager: TaskManager
    @Environment(\.dismiss) var dismiss
    @State private var newFolderName = ""
    @State private var selectedURL: URL? = nil
    @State private var addRuleError: String? = nil
    @State private var benchmarkRunning = false
    @State private var benchmarkProgress = 0
    @State private var benchmarkResult: String? = nil
    @State private var benchmarkTask: Task<Void, Never>? = nil
    // 세션 한정 — UserDefaults persistence 제거됨. 앱 재시작 시 .cpuAndGPU로 시작.
    // 2026-05-04 벤치마크에서 .cpuAndGPU가 .all보다 19% 빠름 (8.83s vs 10.88s/분류).
    @State private var computeUnits: MLComputeUnits = .cpuAndGPU

    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 15) {
            Text("⚙️ 감시 규칙 및 실제 폴더 연결").font(.title2).fontWeight(.bold)

            List {
                ForEach(taskManager.folderRules) { rule in
                    HStack {
                        VStack(alignment: .leading) {
                            Text("📁 \(rule.folderName)").font(.headline)
                            if let resolved = try? rule.resolveURL() {
                                Text(resolved.url.lastPathComponent).font(.caption).foregroundColor(.secondary)
                            } else {
                                Text("(경로 해석 실패 — 폴더를 다시 선택하세요)").font(.caption).foregroundColor(.red)
                            }
                        }
                        Spacer()
                        Button("삭제") { taskManager.deleteRule(id: rule.id) }.buttonStyle(.plain).foregroundColor(.red)
                    }.padding(.vertical, 4)
                }
            }.listStyle(.bordered).frame(height: 150)

            Divider()
            Text("새 감시 폴더 추가").font(.headline)
            VStack(spacing: 10) {
                HStack { Text("분류할 별명:"); TextField("예: 운영체제", text: $newFolderName).textFieldStyle(.roundedBorder) }
                HStack {
                    Text("실제 폴더:"); Text(selectedURL?.lastPathComponent ?? "선택 안 됨").foregroundColor(selectedURL == nil ? .red : .blue)
                    Spacer()
                    Button("폴더 찾기") { selectFolderFromMac() }
                }
                if let addRuleError {
                    Text("⚠️ \(addRuleError)").font(.caption).foregroundColor(.orange)
                }
                Button("추가하고 감시 시작하기") {
                    if !newFolderName.isEmpty, let url = selectedURL {
                        do {
                            let bookmark = try url.bookmarkData(
                                options: .withSecurityScope,
                                includingResourceValuesForKeys: nil,
                                relativeTo: nil
                            )
                            taskManager.addRule(name: newFolderName, bookmark: bookmark)
                            newFolderName = ""; selectedURL = nil; addRuleError = nil
                        } catch {
                            addRuleError = "북마크 생성 실패: \(error.localizedDescription)"
                        }
                    }
                }.buttonStyle(.borderedProminent).disabled(newFolderName.isEmpty || selectedURL == nil)
            }.padding().background(Color(NSColor.controlBackgroundColor)).cornerRadius(8)

            Divider()
            Text("속도 벤치마크").font(.headline)
            VStack(alignment: .leading, spacing: 8) {
                Picker("Compute Units:", selection: $computeUnits) {
                    Text("CPU + GPU (권장)").tag(MLComputeUnits.cpuAndGPU)
                    Text("ANE + GPU + CPU (.all)").tag(MLComputeUnits.all)
                }
                .pickerStyle(.menu)
                .disabled(benchmarkRunning)
                .onChange(of: computeUnits) { _, new in
                    Task { await LocalLLMService.shared.setComputeUnits(new) }
                }
                Text("바꾸면 모델 재로드 필요 — 첫 분류 전 잠시 대기")
                    .font(.caption2).foregroundColor(.secondary)
                Text("CPU only / ANE-only 모드는 Stateful Mistral과 호환 안 됨 (2026-05-04 확인)")
                    .font(.caption2).foregroundColor(.secondary)

                if taskManager.folderRules.count < 2 {
                    Text("폴더 규칙이 2개 이상 있어야 분류기가 동작해요. 위에서 폴더를 추가해 주세요.")
                        .font(.caption).foregroundColor(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    HStack {
                        Button(benchmarkRunning ? "중단" : "벤치마크 실행 (\(Self.benchmarkIterations)회 랜덤)") {
                            if benchmarkRunning {
                                benchmarkTask?.cancel()
                            } else {
                                runBenchmark()
                            }
                        }.buttonStyle(.borderedProminent)
                        Text("프리셋 \(Self.benchmarkPresets.count)개에서 \(Self.benchmarkIterations)개 랜덤 추출")
                            .font(.caption2).foregroundColor(.secondary)
                        Spacer()
                    }
                    if benchmarkRunning {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text("진행 \(benchmarkProgress)/\(Self.benchmarkIterations)…")
                                .font(.caption).foregroundColor(.secondary)
                        }
                    }
                    if let result = benchmarkResult {
                        Text(result)
                            .font(.caption).foregroundColor(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }.padding().background(Color(NSColor.controlBackgroundColor)).cornerRadius(8)

            HStack { Spacer(); Button("닫기") { dismiss() }.keyboardShortcut(.escape) }
        }.padding()
        }
        .frame(width: 500, height: 600)
    }

    private func selectFolderFromMac() {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
        if panel.runModal() == .OK { self.selectedURL = panel.url }
    }

    private static let benchmarkIterations = 5
    /// 벤치마크용 프리셋 task 풀. 사용자의 실제 폴더 카테고리(희곡교육론 / SW프로그래밍의기초 / 국어교과교육론)에서
    /// 자연스럽게 나올 만한 할 일 15개. classifyTask가 실제로 LLM 추론 경로(F1.1 fast logits, F1.3 조건부 시드)를
    /// 거치는지 확인용 — corrections에 없는 항목이라 F1.2 정확매치 단축은 거의 발동 안 함.
    private static let benchmarkPresets: [String] = [
        // 희곡교육론
        "셰익스피어 작품 분석 과제",
        "교실 연극 시나리오 정리",
        "현대 희곡 감상문 작성",
        "낭독극 대본 준비",
        "연극 교육 사례 조사",
        // SW프로그래밍의기초
        "C언어 포인터 실습",
        "for 반복문 연습 문제",
        "재귀 함수 과제 풀이",
        "배열 정렬 코드 작성",
        "코딩 테스트 한 문제 풀기",
        // 국어교과교육론
        "국어 교과서 단원 분석",
        "수업 지도안 작성",
        "문법 단원 자료 정리",
        "교생실습 일지 작성",
        "국어과 교육과정 비교 정리"
    ]

    private func runBenchmark() {
        let corrections = taskManager.corrections
        let folders = taskManager.folderRules.map { $0.folderName }
        guard folders.count >= 2 else { return }
        // 매 실행마다 프리셋에서 5개 랜덤 추출 (no replacement)
        let iterations = Array(Self.benchmarkPresets.shuffled().prefix(Self.benchmarkIterations))

        benchmarkRunning = true
        benchmarkProgress = 0
        benchmarkResult = nil

        benchmarkTask = Task { @MainActor in
            var total = 0
            var totalTime: Double = 0
            let startWall = Date()
            for title in iterations {
                if Task.isCancelled { break }
                // 실제 addTask 경로와 동일하게 corrections.suffix(5) 필터 적용 — 현실적인 측정
                let context = Array(corrections.filter { folders.contains($0.folderName) }.suffix(5))
                let t0 = Date()
                _ = await LocalLLMService.shared.classifyTask(
                    taskTitle: title,
                    availableFolders: folders,
                    corrections: context
                )
                let dt = Date().timeIntervalSince(t0)
                totalTime += dt
                total += 1
                benchmarkProgress = total
            }
            let cancelled = Task.isCancelled
            let avg = total > 0 ? totalTime / Double(total) : 0
            let wall = Date().timeIntervalSince(startWall)
            let units = await LocalLLMService.shared.currentComputeUnits.label
            benchmarkResult = cancelled
                ? String(format: "[%@] 중단됨 (진행 %d/%d) · 평균 %.2fs · 누적 %.1fs",
                         units, total, Self.benchmarkIterations, avg, wall)
                : String(format: "[%@] %d회 평균 추론 시간 %.2fs · 총 %.1fs",
                         units, total, avg, wall)
            benchmarkRunning = false
        }
    }
}

extension DateFormatter {
    static let monthYearFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "YYYY년 MM월"
        return formatter
    }()
}
