import PhotosUI
import SwiftUI
import UIKit

@MainActor
final class CreateFlowViewModel: ObservableObject {
    enum Stage: Equatable {
        case references
        case uploading
        case ready
        case previewGenerating
        case previewReady
        case purchasing
        case setGenerating
        case delivered
        case failed
    }

    let theme: PortraitTheme?
    let source: ProjectSource
    let sharedRecipeID: String?
    let intent: PortraitIntent
    private let api: APIClient
    @Published var stage: Stage = .references
    @Published var referenceImages: [Data] = []
    @Published var inspirationImage: Data?
    @Published var gender = "female"
    @Published var consent = false
    @Published var adultConfirmed = false
    @Published var rightsConfirmed = false
    @Published var project: PortraitProject?
    @Published var heroImage: UIImage?
    @Published var deliveredSet: PortraitPhotoSet?
    @Published var qualityFeedback: [ReferenceRoleFeedback] = []
    @Published var localPhotoCues: [ReferencePhotoCue] = []
    @Published var referenceImportMessage: String?
    @Published var inspirationImportMessage: String?
    @Published private(set) var isPreparingReferences = false
    @Published private(set) var isPreparingInspiration = false
    @Published var errorMessage: String?
    @Published var progressMessage = "正在准备你的私人写真创作"
    @Published var previewRetriesRemaining = 1
    private var referenceImportGeneration = UUID()
    private var inspirationImportGeneration = UUID()

    init(
        theme: PortraitTheme?,
        source: ProjectSource,
        sharedRecipeID: String? = nil,
        intent: PortraitIntent? = nil,
        api: APIClient = .shared
    ) {
        self.theme = theme
        self.source = source
        self.sharedRecipeID = sharedRecipeID
        self.intent = intent ?? .authentic
        self.api = api
        self.rightsConfirmed = source != .privateInspiration
        if let presentation = theme?.presentation,
           presentation == "female" || presentation == "male" {
            self.gender = presentation
        }
    }

    var canSubmit: Bool {
        referenceImages.count >= 4 && referenceImages.count <= 6 && consent && adultConfirmed
            && rightsConfirmed && (source != .privateInspiration || inspirationImage != nil)
            && !isPreparingReferences && !isPreparingInspiration
    }

    func loadReferenceItems(_ items: [PhotosPickerItem]) async {
        let generation = UUID()
        referenceImportGeneration = generation
        referenceImages = []
        qualityFeedback = []
        localPhotoCues = []
        referenceImportMessage = nil
        isPreparingReferences = !items.isEmpty
        defer {
            if referenceImportGeneration == generation { isPreparingReferences = false }
        }
        let selectedItems = Array(items.prefix(6))
        let prepared = await Self.jpegData(selectedItems)
        guard referenceImportGeneration == generation else { return }
        referenceImages = prepared
        localPhotoCues = Array(repeating: .reading, count: prepared.count)
        let cues = await Self.preflightCues(for: prepared)
        guard referenceImportGeneration == generation else { return }
        localPhotoCues = cues
        let failedCount = selectedItems.count - prepared.count
        referenceImportMessage = Self.importMessage(
            failedCount: failedCount,
            subject: "身份照片"
        )
    }

    func loadGuidedReferences(_ cameraPhotos: [Data]) async {
        let generation = UUID()
        referenceImportGeneration = generation
        referenceImages = []
        qualityFeedback = []
        localPhotoCues = []
        referenceImportMessage = nil
        isPreparingReferences = !cameraPhotos.isEmpty
        defer {
            if referenceImportGeneration == generation { isPreparingReferences = false }
        }

        let prepared = await Self.jpegData(cameraPhotos)
        guard referenceImportGeneration == generation else { return }
        referenceImages = prepared
        localPhotoCues = Array(repeating: .reading, count: prepared.count)
        localPhotoCues = await Self.preflightCues(for: prepared)
        referenceImportMessage = Self.importMessage(
            failedCount: cameraPhotos.count - prepared.count,
            subject: "相机照片"
        )
    }

    func loadInspirationItem(_ item: PhotosPickerItem?) async {
        let generation = UUID()
        inspirationImportGeneration = generation
        inspirationImage = nil
        inspirationImportMessage = nil
        isPreparingInspiration = item != nil
        defer {
            if inspirationImportGeneration == generation { isPreparingInspiration = false }
        }
        guard let item else {
            return
        }
        let prepared = await Self.jpegData([item]).first
        guard inspirationImportGeneration == generation else { return }
        inspirationImage = prepared
        inspirationImportMessage = prepared == nil
            ? "这张灵感图无法读取，请选择 JPEG、HEIC 或 PNG 图片。"
            : nil
    }

    func checkReferences(session: AppSession) async {
        guard canSubmit else { return }
        stage = .uploading
        errorMessage = nil
        var draftProjectID: String?
        var cleanupToken: String?
        var shouldCleanupDraft = true
        do {
            let token = try await session.ensureGuest()
            cleanupToken = token
            progressMessage = "正在为你打开一间私人影棚"
            let project = try await api.createProject(
                token: token,
                source: source,
                themeID: theme?.themeId,
                gender: gender,
                sharedRecipeID: sharedRecipeID
            )
            draftProjectID = project.projectId
            self.project = project
            if source == .privateInspiration, let inspirationImage {
                progressMessage = "正在理解灵感图的光线、服装和氛围"
                let response = try await api.uploadInspiration(
                    projectID: project.projectId,
                    token: token,
                    image: inspirationImage
                )
                guard response.analysisStatus == "analyzed" else {
                    throw APIError.server(409, response.message)
                }
            }
            progressMessage = "正在认识你的五官和神情"
            let references = try await api.uploadReferences(
                projectID: project.projectId,
                token: token,
                images: referenceImages,
                gender: gender
            )
            qualityFeedback = references.referenceQuality.roleCoverage ?? []
            guard references.referenceQuality.pass == true else {
                let identityMismatch = references.referenceQuality.issues?
                    .contains("reference_identity_mismatch") == true
                let anglesTooSimilar = references.referenceQuality.issues?
                    .contains("insufficient_pose_diversity") == true
                throw APIError.server(
                    422,
                    identityMismatch
                        ? "这些照片可能不是同一个人。请只使用同一位成年人的照片。"
                        : anglesTooSimilar
                            ? "照片很清晰，但角度太相似。请补一张正面照，并分别向左右轻转各拍一张。"
                            : "请替换标记出的照片，然后重新检查。"
                )
            }
            shouldCleanupDraft = false
            stage = .ready
        } catch {
            if shouldCleanupDraft, let draftProjectID, let cleanupToken {
                try? await api.deleteProject(draftProjectID, token: cleanupToken)
                if project?.projectId == draftProjectID { project = nil }
            }
            errorMessage = error.localizedDescription
            stage = .references
        }
    }

    func startPreview(session: AppSession) async {
        guard let project, let token = session.token else { return }
        errorMessage = nil
        progressMessage = "正在创作你的第一张写真"
        stage = .previewGenerating
        do {
            _ = try await api.startPreview(projectID: project.projectId, token: token)
            try await pollPreview(projectID: project.projectId, token: token)
        } catch {
            errorMessage = error.localizedDescription
            stage = .ready
        }
    }

    func changeReferences(session: AppSession) async {
        if let project, let token = session.token {
            try? await api.deleteProject(project.projectId, token: token)
        }
        project = nil
        qualityFeedback = []
        localPhotoCues = []
        errorMessage = nil
        stage = .references
    }

    func purchase(session: AppSession, store: StoreKitManager) async {
        guard let project else { return }
        stage = .purchasing
        errorMessage = nil
        do {
            try await store.purchase(projectID: project.projectId, session: session)
            stage = .setGenerating
            progressMessage = "正在完成余下的写真故事"
            guard let token = session.token else { throw APIError.missingToken }
            try await pollDelivery(projectID: project.projectId, token: token)
        } catch {
            errorMessage = error.localizedDescription
            stage = .previewReady
        }
    }

    func retryPreview(session: AppSession, reason: String) async {
        guard let project, let token = session.token, previewRetriesRemaining > 0 else { return }
        let previousHero = heroImage
        errorMessage = nil
        progressMessage = "正在根据你的反馈，让新成片更像你"
        stage = .previewGenerating
        do {
            let response = try await api.retryPreview(
                projectID: project.projectId,
                token: token,
                reason: reason
            )
            previewRetriesRemaining = response.retriesRemaining
            try await pollPreview(projectID: project.projectId, token: token)
        } catch {
            heroImage = previousHero
            errorMessage = error.localizedDescription
            stage = .previewReady
        }
    }

    func recordPreviewConfirmation(session: AppSession) async {
        guard let project, let token = session.token else { return }
        _ = try? await api.confirmPreview(projectID: project.projectId, token: token)
    }

    private func pollPreview(projectID: String, token: String) async throws {
        for _ in 0..<240 {
            try Task.checkCancellation()
            let current = try await api.project(projectID, token: token)
            project = current
            previewRetriesRemaining = current.previewRetriesRemaining ?? previewRetriesRemaining
            switch current.status {
            case .previewReady:
                let data = try await api.heroData(projectID: projectID, token: token)
                guard let image = UIImage(data: data) else { throw APIError.invalidResponse }
                heroImage = image
                stage = .previewReady
                return
            case .failed:
                throw APIError.server(
                    422,
                    current.failureMessage
                        ?? "第一张写真没有通过质量检查，项目仍保留在你的写真集中。"
                )
            default:
                try await Task.sleep(for: .seconds(2))
            }
        }
        throw APIError.server(504, "生成时间比预期更长，项目会继续保留在你的写真集中。")
    }

    private func pollDelivery(projectID: String, token: String) async throws {
        for _ in 0..<360 {
            try Task.checkCancellation()
            let current = try await api.project(projectID, token: token)
            project = current
            switch current.status {
            case .delivered:
                guard let setID = current.photoSetId else { throw APIError.invalidResponse }
                deliveredSet = try await api.photoSet(
                    projectID: projectID, setID: setID, token: token
                )
                stage = .delivered
                return
            case .failed:
                throw APIError.server(
                    422,
                    current.failureMessage
                        ?? "有一张写真没有通过最终检查。你的购买权益已保留，可稍后重试或联系支持。"
                )
            default:
                try await Task.sleep(for: .seconds(2))
            }
        }
        throw APIError.server(504, "整套写真仍在创作中，你可以放心返回写真集。")
    }

    private static func jpegData(_ items: [PhotosPickerItem]) async -> [Data] {
        var output: [Data] = []
        for item in items.prefix(6) {
            guard !Task.isCancelled,
                  let data = try? await item.loadTransferable(type: Data.self),
                  let jpeg = await PortraitImagePreparer.prepare(data) else { continue }
            output.append(jpeg)
        }
        return output
    }

    private static func jpegData(_ sources: [Data]) async -> [Data] {
        var output: [Data] = []
        for data in sources.prefix(6) {
            guard !Task.isCancelled,
                  let jpeg = await PortraitImagePreparer.prepare(data) else { continue }
            output.append(jpeg)
        }
        return output
    }

    private static func importMessage(failedCount: Int, subject: String) -> String? {
        guard failedCount > 0 else { return nil }
        return "有 \(failedCount) 张\(subject)无法读取，请选择 JPEG、HEIC 或 PNG 图片。"
    }

    private static func preflightCues(for images: [Data]) async -> [ReferencePhotoCue] {
        await withTaskGroup(of: (Int, ReferencePhotoCue).self) { group in
            for (index, image) in images.enumerated() {
                group.addTask {
                    (index, await ReferencePhotoPreflight.analyze(image, position: index))
                }
            }
            var output = Array(repeating: ReferencePhotoCue.reading, count: images.count)
            for await (index, cue) in group {
                output[index] = cue
            }
            return output
        }
    }
}

struct CreateFlowView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var session: AppSession
    @EnvironmentObject private var store: StoreKitManager
    @StateObject private var model: CreateFlowViewModel
    @State private var referenceItems: [PhotosPickerItem] = []
    @State private var inspirationItem: PhotosPickerItem?
    @State private var developedHero = false
    @State private var previewConfirmed = false
    @State private var showingFullSet = false
    @State private var showingPreviewFullScreen = false
    @State private var showingLikenessComparison = false
    @State private var showingRetryChoice = false
    @State private var showingGuidedCapture = false

    init(
        theme: PortraitTheme?,
        source: ProjectSource,
        sharedRecipeID: String? = nil,
        intent: PortraitIntent? = nil
    ) {
        _model = StateObject(wrappedValue: CreateFlowViewModel(
            theme: theme,
            source: source,
            sharedRecipeID: sharedRecipeID,
            intent: intent
        ))
    }

    var body: some View {
        Group {
            switch model.stage {
            case .references: referencesView
            case .uploading, .previewGenerating: progressView
            case .ready: readyView
            case .previewReady: previewView
            case .purchasing: purchaseProgressView
            case .setGenerating: progressView
            case .delivered: deliveredView
            case .failed: failureView
            }
        }
        .id(model.stage)
        .transition(FlashShotMotion.stageTransition(reduceMotion: reduceMotion))
        .animation(FlashShotMotion.gentle(reduceMotion: reduceMotion), value: model.stage)
        .navigationTitle(navigationTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("关闭") { dismiss() }
            }
        }
        .fullScreenCover(isPresented: $showingPreviewFullScreen) {
            ZStack {
                Color.black.ignoresSafeArea()
                if let image = model.heroImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                }
            }
            .overlay(alignment: .topTrailing) {
                Button("关闭大图", systemImage: "xmark.circle.fill") {
                    showingPreviewFullScreen = false
                }
                .labelStyle(.iconOnly)
                .font(.largeTitle)
                .foregroundStyle(.white)
                .padding()
            }
        }
        .fullScreenCover(isPresented: $showingGuidedCapture) {
            GuidedPortraitCapture { photos in
                Task { await model.loadGuidedReferences(photos) }
            }
        }
        .confirmationDialog(
            "你觉得哪里还不够像？",
            isPresented: $showingRetryChoice,
            titleVisibility: .visible
        ) {
            Button("五官长相") {
                Task { await model.retryPreview(session: session, reason: "identity") }
            }
            Button("神情状态") {
                Task { await model.retryPreview(session: session, reason: "expression") }
            }
            Button("整体感觉") {
                Task { await model.retryPreview(session: session, reason: "overall") }
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text("这次免费重试会围绕你的反馈重新创作。")
        }
        .onChange(of: model.stage) { _, stage in
            if stage == .previewGenerating {
                developedHero = false
                previewConfirmed = false
                showingFullSet = false
                showingLikenessComparison = false
            }
        }
    }

    private var navigationTitle: String {
        model.theme?.title ?? "自由写真"
    }

    private var referencesView: some View {
        let inspirationSelected = model.inspirationImage != nil
        let referenceCount = model.referenceImages.count
        return ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("一段很短的拍摄")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(FlashShotStyle.accent)
                    Text("给写真四个真实的你作为起点")
                        .font(.title2.bold())
                    Text("跟随引导拍下你此刻的角度和神情，也可以直接选择已有的清晰照片。")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .flashShotEntrance(order: 0)

                if model.source == .privateInspiration {
                    sectionTitle("1", "你想进入的世界")
                    PhotosPicker(selection: $inspirationItem, matching: .images) {
                        photoPickerLabel(
                            title: inspirationSelected ? "已选择灵感图" : "选择一张灵感图",
                            symbol: inspirationSelected ? "checkmark.circle.fill" : "photo.badge.plus"
                        )
                    }
                    .buttonStyle(PortraitCardButtonStyle())
                    if model.isPreparingInspiration {
                        ProgressView("正在准备灵感图")
                            .font(.footnote)
                    } else if let message = model.inspirationImportMessage {
                        importMessage(message)
                    } else if inspirationSelected {
                        Label(
                            "我们只会读取这张图的氛围，你的脸始终只来自你自己的照片。",
                            systemImage: "checkmark.circle.fill"
                        )
                        .font(.footnote)
                        .foregroundStyle(FlashShotStyle.jade)
                    }
                    Toggle("我有权将这张图片用于私人创作", isOn: $model.rightsConfirmed)
                }

                sectionTitle(model.source == .privateInspiration ? "2" : "1", "留下真实的你")
                Button {
                    showingGuidedCapture = true
                } label: {
                    HStack(spacing: 14) {
                        Image(systemName: "camera.viewfinder")
                            .font(.title2.weight(.semibold))
                            .frame(width: 36, height: 36)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(referenceCount == 0 ? "开始引导拍摄" : "重新进行引导拍摄")
                                .font(.headline)
                            Text("四个角度 · 大约三分钟")
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.78))
                        }
                        Spacer()
                        Image(systemName: "arrow.right")
                            .font(.headline)
                    }
                    .foregroundStyle(.white)
                    .padding(16)
                    .background(FlashShotStyle.ink, in: RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(PortraitCardButtonStyle())

                HStack(spacing: 12) {
                    Rectangle().fill(Color.secondary.opacity(0.22)).frame(height: 1)
                    Text("或者")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.secondary)
                    Rectangle().fill(Color.secondary.opacity(0.22)).frame(height: 1)
                }

                PhotosPicker(selection: $referenceItems, maxSelectionCount: 6, matching: .images) {
                    photoPickerLabel(
                        title: referenceCount == 0 ? "从相册选择" : "更换相册照片",
                        symbol: "person.crop.rectangle.stack"
                    )
                }
                .buttonStyle(PortraitCardButtonStyle())
                if model.isPreparingReferences {
                    ProgressView("正在准备所选照片")
                        .font(.footnote)
                } else if let message = model.referenceImportMessage {
                    importMessage(message)
                }
                if !model.referenceImages.isEmpty {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 142), spacing: 10)], spacing: 14) {
                        ForEach(Array(model.referenceImages.enumerated()), id: \.offset) { index, data in
                            if let image = UIImage(data: data) {
                                ReferencePhotoThumbnail(
                                    image: image,
                                    cue: model.localPhotoCues.indices.contains(index)
                                        ? model.localPhotoCues[index]
                                        : nil,
                                    serverFeedback: model.qualityFeedback.indices.contains(index)
                                        ? model.qualityFeedback[index]
                                        : nil
                                )
                                .transition(.scale(scale: 0.92).combined(with: .opacity))
                            }
                        }
                    }
                    .animation(
                        FlashShotMotion.quick(reduceMotion: reduceMotion),
                        value: referenceCount
                    )
                }

                let failedFeedback = model.qualityFeedback.filter { !$0.pass }
                if !failedFeedback.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Label(
                            "有几张照片需要重新选择",
                            systemImage: "exclamationmark.triangle.fill"
                        )
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(FlashShotStyle.accent)

                        ForEach(failedFeedback, id: \.role) { feedback in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(feedback.title)
                                    .font(.footnote.weight(.semibold))
                                Text(feedback.problemTitle)
                                    .font(.footnote)
                                Text(feedback.nextStep)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(14)
                    .background(
                        FlashShotStyle.accent.opacity(0.07),
                        in: RoundedRectangle(cornerRadius: 7)
                    )
                }

                if referenceCount > 0 && !model.isPreparingReferences {
                    referenceEncouragement(count: referenceCount)
                }

                if showsPresentationPicker {
                    sectionTitle(model.source == .privateInspiration ? "3" : "2", "造型方向")
                    Picker("造型方向", selection: $model.gender) {
                        Text("女性造型").tag("female")
                        Text("男性造型").tag("male")
                    }
                    .pickerStyle(.segmented)
                }

                if referenceCount >= 4 && !model.isPreparingReferences {
                    sectionTitle(privacySectionNumber, "开始写真之前")
                    VStack(spacing: 0) {
                        privacyPromiseToggle(
                            title: "仅将这些照片用于本次创作",
                            detail: "照片会保持私密，并在七天内删除。",
                            symbol: "lock.fill",
                            isOn: $model.consent
                        )
                        Divider().padding(.leading, 54)
                        privacyPromiseToggle(
                            title: "照片中是我本人，且我已成年",
                            detail: "我确认照片中的人物是本人，并且已满 18 周岁。",
                            symbol: "person.crop.circle.badge.checkmark",
                            isOn: $model.adultConfirmed
                        )
                    }
                    .background(FlashShotStyle.secondaryPaper, in: RoundedRectangle(cornerRadius: 7))
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                if let error = model.errorMessage {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .font(.footnote)
                        .foregroundStyle(FlashShotStyle.accent)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(18)
            .padding(.bottom, 106)
            .flashShotContentWidth()
        }
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 7) {
                Label(
                    creationReadinessMessage,
                    systemImage: model.canSubmit ? "checkmark.circle.fill" : "circle.dotted"
                )
                .font(.footnote.weight(.medium))
                .foregroundStyle(model.canSubmit ? FlashShotStyle.jade : .secondary)
                .contentTransition(.opacity)

                Button("检查我的照片") {
                    Task { await model.checkReferences(session: session) }
                }
                .buttonStyle(PrimaryActionButtonStyle())
                .disabled(!model.canSubmit)
                .opacity(model.canSubmit ? 1 : 0.48)
            }
            .padding(.horizontal, 18)
            .padding(.top, 10)
            .padding(.bottom, 6)
            .background(.bar)
            .animation(
                FlashShotMotion.quick(reduceMotion: reduceMotion),
                value: model.canSubmit
            )
        }
        .onChange(of: referenceItems) { _, items in
            Task { await model.loadReferenceItems(items) }
        }
        .onChange(of: inspirationItem) { _, item in
            Task { await model.loadInspirationItem(item) }
        }
        .sensoryFeedback(.selection, trigger: referenceCount)
        .sensoryFeedback(.selection, trigger: model.gender)
        .sensoryFeedback(.success, trigger: model.canSubmit) { oldValue, newValue in
            !oldValue && newValue
        }
    }

    private var readyView: some View {
        VStack(spacing: 18) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(FlashShotStyle.jade)
            Text("你的照片准备好了")
                .font(.largeTitle.bold())
                .multilineTextAlignment(.center)
            Text("我们找到了四个有效角度。现在还没有生成任何图片。")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("免费创作第一张写真") {
                Task { await model.startPreview(session: session) }
            }
            .buttonStyle(PrimaryActionButtonStyle())
            Button("更换照片") {
                Task { await model.changeReferences(session: session) }
            }
            .buttonStyle(.bordered)
            if let error = model.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(FlashShotStyle.accent)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(28)
        .flashShotContentWidth(520)
        .flashShotEntrance()
    }

    private var progressView: some View {
        DarkroomDevelopingView(
            image: model.heroImage ?? model.referenceImages.first.flatMap(UIImage.init(data:)),
            mode: developingMode
        )
        .flashShotEntrance()
    }

    private var previewView: some View {
        ScrollView {
            VStack(spacing: 18) {
                Text("你的第一张写真完成了")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(FlashShotStyle.accent)
                VStack(spacing: 7) {
                    Text(revealTitle).font(.largeTitle.bold())
                    if developedHero {
                        Text(model.intent.previewLine)
                            .font(.title3)
                            .multilineTextAlignment(.center)
                        Label("已检查相似度与画面质量", systemImage: "checkmark.seal.fill")
                            .font(.footnote)
                            .foregroundStyle(FlashShotStyle.jade)
                        Text("先好好看看这张照片。只有当你确认它真的像自己，再继续。")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                }
                .flashShotEntrance(order: 1)

                if let image = model.heroImage {
                    if !developedHero {
                        PortraitHeroReveal(image: image, onDeveloped: completeReveal)
                            .transition(.opacity)
                    } else {
                        Button { showingPreviewFullScreen = true } label: {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                        }
                        .buttonStyle(PortraitCardButtonStyle())
                        .accessibilityLabel("全屏查看第一张写真")
                        .transition(.opacity)

                        HStack(spacing: 10) {
                            Button { showingPreviewFullScreen = true } label: {
                                Label("全屏查看", systemImage: "arrow.up.left.and.arrow.down.right")
                            }
                            .buttonStyle(.bordered)

                            if model.referenceImages.first.flatMap(UIImage.init(data:)) != nil {
                                Button {
                                    withAnimation(FlashShotMotion.quick(reduceMotion: reduceMotion)) {
                                        showingLikenessComparison.toggle()
                                    }
                                } label: {
                                    Label(
                                        showingLikenessComparison ? "收起对比" : "对比看看像不像",
                                        systemImage: "arrow.left.and.right"
                                    )
                                }
                                .buttonStyle(.bordered)
                            }
                        }

                        if showingLikenessComparison,
                           let sourceData = model.referenceImages.first,
                           let source = UIImage(data: sourceData) {
                            VStack(spacing: 8) {
                                PortraitComparisonView(source: source, portrait: image)
                                Label(
                                    "左右拖动，对比你熟悉的五官、神情和细节",
                                    systemImage: "arrow.left.and.right"
                                )
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            }
                            .transition(.move(edge: .top).combined(with: .opacity))
                        }
                    }
                }

                if developedHero {
                    if !previewConfirmed {
                        VStack(spacing: 12) {
                            Text("这张写真里的人，一眼看上去就是你吗？")
                                .font(.title3.bold())
                                .multilineTextAlignment(.center)
                            Text("先忽略造型，只看五官、神情，以及那些你熟悉的小细节。")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)

                            Button {
                                confirmPreview()
                            } label: {
                                Label("是的，这就是我", systemImage: "checkmark.circle.fill")
                            }
                            .buttonStyle(PrimaryActionButtonStyle())

                            if model.previewRetriesRemaining > 0 {
                                Button {
                                    showingRetryChoice = true
                                } label: {
                                    Label("还不太像", systemImage: "arrow.clockwise")
                                }
                                .buttonStyle(.bordered)
                            } else {
                                Text("这张预览仍然不够像你。先换一组更合适的照片重新开始，不要急着购买。")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                    .multilineTextAlignment(.center)
                                Button {
                                    Task { await model.changeReferences(session: session) }
                                } label: {
                                    Label("换一组照片重新拍", systemImage: "camera.rotate")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        .padding(.vertical, 6)
                    } else {
                        Label("你已确认这张照片像本人", systemImage: "checkmark.seal.fill")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(FlashShotStyle.jade)

                        if let image = model.heroImage {
                            PortraitKeepsakeButton(image: image)
                        }
                    }

                }

                if showingFullSet {
                    if let image = model.heroImage {
                        PortraitBundlePeek(
                            portrait: image,
                            shotLabels: model.theme?.shotLabels
                        )
                    }
                    if session.isAppleAccount {
                        if let product = store.product {
                            Button {
                                Task { await model.purchase(session: session, store: store) }
                            } label: {
                                HStack {
                                    Text("收藏完整写真故事")
                                    Spacer()
                                    Text(product.displayPrice)
                                }
                            }
                            .buttonStyle(PrimaryActionButtonStyle())
                            .disabled(store.isPurchasing)
                        } else {
                            storeProductUnavailableView
                        }
                    } else {
                        VStack(spacing: 10) {
                            Text("购买前请先通过 Apple 登录，以便在其他设备上恢复这套写真。")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                            AppleAccountButton()
                        }
                    }
                }
                if let error = model.errorMessage {
                    Text(error).font(.footnote).foregroundStyle(FlashShotStyle.accent)
                }
            }
            .padding(18)
            .flashShotContentWidth()
        }
    }

    private var revealTitle: String {
        "看见了，\(model.theme?.title ?? model.intent.title)里的你。"
    }

    private func completeReveal() {
        guard !developedHero else { return }
        withAnimation(FlashShotMotion.gentle(reduceMotion: reduceMotion)) {
            developedHero = true
        }
        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
    }

    private func confirmPreview() {
        withAnimation(FlashShotMotion.gentle(reduceMotion: reduceMotion)) {
            previewConfirmed = true
            showingFullSet = true
        }
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        Task { await model.recordPreviewConfirmation(session: session) }
    }

    private var purchaseProgressView: some View {
        VStack(spacing: 18) {
            ProgressView().controlSize(.large)
            Text("正在通过 Apple 验证购买").font(.headline)
        }
    }

    private var storeProductUnavailableView: some View {
        VStack(spacing: 10) {
            if store.isLoadingProduct {
                ProgressView("正在连接 App Store")
            } else {
                Text(store.productLoadError ?? "这套写真暂时无法购买。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Button { Task { await store.loadProduct() } } label: {
                    Label("重新尝试", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private var deliveredView: some View {
        VStack(spacing: 18) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 54))
                .foregroundStyle(FlashShotStyle.jade)
            Text("你的写真故事完成了").font(.title.bold())
            Text("六个瞬间，同一个你。")
                .foregroundStyle(.secondary)
            if let project = model.project {
                NavigationLink("打开我的写真", value: project)
                    .buttonStyle(PrimaryActionButtonStyle())
            }
        }
        .padding(22)
        .navigationDestination(for: PortraitProject.self) { ProjectDetailView(project: $0) }
    }

    private var failureView: some View {
        ContentUnavailableView {
            Label("这套写真需要处理", systemImage: "exclamationmark.triangle")
        } description: {
            Text(model.errorMessage ?? "项目仍然保留在你的写真集中。")
        }
    }

    private func sectionTitle(_ number: String, _ title: String) -> some View {
        HStack {
            Text(number)
                .font(.caption.bold())
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(FlashShotStyle.ink, in: Circle())
            Text(title).font(.title2.bold())
        }
    }

    private var developingMode: DarkroomDevelopingView.Mode {
        switch model.stage {
        case .uploading:
            return .readingPhotos
        case .previewGenerating:
            return model.heroImage == nil ? .firstPortrait : .closerMatch
        case .setGenerating:
            return .fullSet
        default:
            return .firstPortrait
        }
    }

    private var showsPresentationPicker: Bool {
        let presentation = model.theme?.presentation
        return model.source == .privateInspiration
            || (presentation != "female" && presentation != "male")
    }

    private var privacySectionNumber: String {
        if model.source == .privateInspiration { return "4" }
        return showsPresentationPicker ? "3" : "2"
    }

    private func privacyPromiseToggle(
        title: String,
        detail: String,
        symbol: String,
        isOn: Binding<Bool>
    ) -> some View {
        Toggle(isOn: isOn) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: symbol)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(FlashShotStyle.jade)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .toggleStyle(.switch)
        .padding(14)
    }

    private func photoPickerLabel(title: String, symbol: String) -> some View {
        HStack {
            Image(systemName: symbol).font(.title2)
            Text(title).font(.headline)
            Spacer()
            Image(systemName: "chevron.right").foregroundStyle(.secondary)
        }
        .foregroundStyle(.primary)
        .padding(15)
        .background(FlashShotStyle.secondaryPaper, in: RoundedRectangle(cornerRadius: 7))
    }

    private var creationReadinessMessage: String {
        if model.source == .privateInspiration && model.inspirationImage == nil {
            return "先选择一张灵感图"
        }
        if model.referenceImages.count < 4 {
            let remaining = 4 - model.referenceImages.count
            return "还需要添加 \(remaining) 张照片"
        }
        if model.source == .privateInspiration && !model.rightsConfirmed {
            return "请确认你有权使用这张灵感图"
        }
        if !model.consent || !model.adultConfirmed {
            return "开始前还需要最后一项确认"
        }
        return "可以开始检查照片了"
    }

    private func referenceEncouragement(count: Int) -> some View {
        let ready = (4...6).contains(count)
        return Label(
            ready
                ? "这些角度已经足够让我们清楚认识你。"
                : "还需要添加 \(4 - count) 张照片，才能更好地保留你的五官。",
            systemImage: ready ? "checkmark.circle.fill" : "person.crop.circle.badge.plus"
        )
        .font(.footnote.weight(.medium))
        .foregroundStyle(ready ? FlashShotStyle.jade : .secondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func importMessage(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.footnote)
            .foregroundStyle(FlashShotStyle.accent)
            .fixedSize(horizontal: false, vertical: true)
    }
}
