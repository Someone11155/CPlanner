//
// ContentView.swift
// Cplanner
//
// Created by iLo on 2026-03-26.
//

import SwiftUI
import Combine
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

    @Published var tasks: [TaskItem] = [] {
        didSet { persist(tasks, forKey: Self.tasksKey) }
    }
    @Published var folderRules: [FolderRule] = [] {
        didSet { persist(folderRules, forKey: Self.folderRulesKey) }
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
        Task { [weak self] in
            guard let self else { return }
            let folders = self.folderRules.map { $0.folderName }
            let detectedFolder = await LocalLLMService.shared.classifyTask(taskTitle: title, availableFolders: folders)
            var newTask = TaskItem(title: title, targetFolder: detectedFolder, date: date)

            if self.calendarAccessGranted, let cal = self.cplannerCalendar {
                let event = EKEvent(eventStore: self.eventStore)
                event.title = title
                let day = Calendar.current.startOfDay(for: date)
                event.startDate = day
                event.endDate = day
                event.isAllDay = true
                event.calendar = cal

                self.isApplyingLocalChange = true
                do {
                    try self.eventStore.save(event, span: .thisEvent)
                    newTask.eventIdentifier = event.eventIdentifier
                    self.lastCalendarError = nil
                } catch {
                    self.lastCalendarError = "캘린더 저장 실패: \(error.localizedDescription)"
                    taskManagerLogger.error("Calendar save failed: \(error.localizedDescription, privacy: .public)")
                }
                DispatchQueue.main.async { [weak self] in self?.isApplyingLocalChange = false }
            }

            self.tasks.append(newTask)
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

// --- 4. 메인 UI 화면 ---
struct ContentView: View {
    @StateObject private var taskManager = TaskManager()
    @State private var selectedDate = Date()
    @State private var isAddingTask = false
    @State private var showSettings = false
    @State private var newTaskTitle = ""

    var body: some View {
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
                            HStack {
                                Image(systemName: taskManager.tasks[index].isCompleted ? "checkmark.circle.fill" : "circle")
                                    .foregroundColor(taskManager.tasks[index].isCompleted ? .blue : .gray).font(.title3)
                                    .onTapGesture { taskManager.tasks[index].isCompleted.toggle() }
                                VStack(alignment: .leading) {
                                    Text(task.title).strikethrough(taskManager.tasks[index].isCompleted).foregroundColor(taskManager.tasks[index].isCompleted ? .gray : .primary).font(.headline)
                                    Text("📁 자동 분류됨: \(task.targetFolder)").font(.caption).foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                            .padding(.vertical, 6)
                            .contextMenu { Button(role: .destructive) { taskManager.deleteTask(id: task.id) } label: { Label("삭제", systemImage: "trash") } }
                        }
                    }
                }.listStyle(.inset)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity).background(Color(NSColor.textBackgroundColor))
        }
        .sheet(isPresented: $showSettings) { SettingsView(taskManager: taskManager) }
    }
}

// --- 5. 설정 뷰 ---
struct SettingsView: View {
    @ObservedObject var taskManager: TaskManager
    @Environment(\.dismiss) var dismiss
    @State private var newFolderName = ""
    @State private var selectedURL: URL? = nil
    @State private var addRuleError: String? = nil

    var body: some View {
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

            HStack { Spacer(); Button("닫기") { dismiss() }.keyboardShortcut(.escape) }
        }.padding().frame(width: 500, height: 450)
    }

    private func selectFolderFromMac() {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
        if panel.runModal() == .OK { self.selectedURL = panel.url }
    }
}

extension DateFormatter {
    static let monthYearFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "YYYY년 MM월"
        return formatter
    }()
}
