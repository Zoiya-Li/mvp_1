import Photos
import SwiftUI
import UIKit

@MainActor
final class ProjectDetailViewModel: ObservableObject {
    @Published var project: PortraitProject
    @Published var images: [UIImage] = []
    @Published var assetIDs: [String] = []
    @Published var hero: UIImage?
    @Published var shareURL: URL?
    @Published var isLoading = false
    @Published var isCompletingSet = false
    @Published var isRetryingPreview = false
    @Published var isSaving = false
    @Published var isDeleted = false
    @Published var message: String?

    init(project: PortraitProject) { self.project = project }

    func load(session: AppSession) async {
        guard let token = session.token else { return }
        isLoading = true
        defer { isLoading = false }
        do {
            project = try await APIClient.shared.project(project.projectId, token: token)
            if project.status == .delivered, let setID = project.photoSetId {
                let set = try await APIClient.shared.photoSet(projectID: project.projectId, setID: setID, token: token)
                var loaded: [UIImage] = []
                let orderedAssets = set.assets.sorted(by: { $0.position < $1.position })
                for asset in orderedAssets {
                    let data = try await APIClient.shared.assetData(
                        projectID: project.projectId, assetID: asset.assetId, token: token
                    )
                    if let image = UIImage(data: data) { loaded.append(image) }
                }
                images = loaded
                assetIDs = orderedAssets.map(\.assetId)
            } else if project.status == .previewReady {
                hero = UIImage(data: try await APIClient.shared.heroData(projectID: project.projectId, token: token))
            }
        } catch { message = error.localizedDescription }
    }

    func monitor(session: AppSession) async {
        await load(session: session)
        while project.status.isWorking && !Task.isCancelled {
            try? await Task.sleep(for: .seconds(2))
            await load(session: session)
        }
    }

    func completeSet(session: AppSession, store: StoreKitManager) async {
        guard let token = session.token else { return }
        isCompletingSet = true
        message = nil
        defer { isCompletingSet = false }
        do {
            do {
                _ = try await APIClient.shared.unlock(projectID: project.projectId, token: token)
            } catch APIError.server(402, _) {
                try await store.purchase(projectID: project.projectId, session: session)
            }
            await monitor(session: session)
        } catch {
            message = error.localizedDescription
            await load(session: session)
        }
    }

    func confirmPreview(session: AppSession) async {
        guard let token = session.token else { return }
        message = nil
        do {
            _ = try await APIClient.shared.confirmPreview(
                projectID: project.projectId,
                token: token
            )
            project = try await APIClient.shared.project(project.projectId, token: token)
        } catch {
            message = "刚才没能保存你的选择，请检查网络后重试。"
        }
    }

    func retryPreview(session: AppSession, reason: String) async {
        guard let token = session.token,
              project.status == .previewReady,
              (project.previewRetriesRemaining ?? 0) > 0 else { return }
        isRetryingPreview = true
        message = nil
        defer { isRetryingPreview = false }
        do {
            _ = try await APIClient.shared.retryPreview(
                projectID: project.projectId,
                token: token,
                reason: reason
            )
            hero = nil
            await monitor(session: session)
        } catch {
            message = error.localizedDescription
            await load(session: session)
        }
    }

    func saveCleanOriginals(session: AppSession) async {
        guard let token = session.token else { return }
        guard !assetIDs.isEmpty else {
            message = "写真完成后，就可以在这里保存。"
            return
        }
        isSaving = true
        defer { isSaving = false }

        let authorization = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard authorization == .authorized || authorization == .limited else {
            message = "请在系统设置中允许访问相册，才能保存写真。"
            return
        }
        do {
            var portraits: [UIImage] = []
            for assetID in assetIDs {
                let data = try await APIClient.shared.cleanAssetData(
                    projectID: project.projectId,
                    assetID: assetID,
                    token: token
                )
                guard let image = UIImage(data: data) else {
                    throw APIError.invalidResponse
                }
                portraits.append(image)
            }
            try await PHPhotoLibrary.shared().performChanges {
                for image in portraits {
                    PHAssetChangeRequest.creationRequestForAsset(from: image)
                }
            }
            message = "\(portraits.count) 张无水印原图已保存到相册。"
        } catch {
            message = "写真未能保存到相册：\(error.localizedDescription)"
        }
    }

    func createShare(session: AppSession) async {
        guard let token = session.token else { return }
        do {
            let recipe = try await APIClient.shared.shareRecipe(projectID: project.projectId, token: token)
            shareURL = URL(string: "https://flashshot.top/s/\(recipe.shareToken)")
        } catch { message = error.localizedDescription }
    }

    func delete(session: AppSession) async {
        guard let token = session.token else { return }
        do {
            try await APIClient.shared.deleteProject(project.projectId, token: token)
            isDeleted = true
        } catch { message = error.localizedDescription }
    }
}

struct ProjectDetailView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var session: AppSession
    @EnvironmentObject private var store: StoreKitManager
    @StateObject private var model: ProjectDetailViewModel
    @State private var selectedImageIndex: Int?
    @State private var confirmDelete = false
    @State private var confirmCleanExport = false
    @State private var showingRetryChoice = false

    init(project: PortraitProject) {
        _model = StateObject(wrappedValue: ProjectDetailViewModel(project: project))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text(model.project.status == .delivered ? "你的写真故事" : "你的第一张写真")
                        .font(.title.bold())
                    Spacer()
                    StatusPill(status: model.project.status)
                }
                .flashShotEntrance(order: 0)

                if model.isLoading && model.images.isEmpty && model.hero == nil {
                    ProgressView().frame(maxWidth: .infinity, minHeight: 260)
                } else if !model.images.isEmpty {
                    PortraitStorySequence(images: model.images) { index in
                        selectedImageIndex = index
                    }
                    HStack(spacing: 10) {
                        Button { confirmCleanExport = true } label: {
                            if model.isSaving {
                                ProgressView()
                            } else {
                                Label("保存无水印原图", systemImage: "square.and.arrow.down")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isSaving)
                        Button { Task { await model.createShare(session: session) } } label: {
                            Label("分享这套写真配方", systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(.bordered)
                    }
                } else if let hero = model.hero {
                    Image(uiImage: hero)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .flashShotEntrance(order: 1)
                    if model.project.previewConfirmed == true {
                        VStack(alignment: .leading, spacing: 5) {
                            Label("已确认像本人", systemImage: "checkmark.seal.fill")
                                .font(.title2.bold())
                                .foregroundStyle(FlashShotStyle.jade)
                            Text("余下五个构图会以这张已经确认的身份为基础继续创作。")
                                .foregroundStyle(.secondary)
                        }
                        if session.isAppleAccount {
                            if let product = store.product {
                                Button {
                                    Task { await model.completeSet(session: session, store: store) }
                                } label: {
                                    HStack {
                                        if model.isCompletingSet {
                                            ProgressView().tint(.white)
                                        } else {
                                            Text("完成整套写真")
                                            Spacer()
                                            Text(product.displayPrice)
                                        }
                                    }
                                }
                                .buttonStyle(PrimaryActionButtonStyle())
                                .disabled(model.isCompletingSet)
                            } else {
                                storeProductUnavailableView
                            }
                        } else {
                            VStack(alignment: .leading, spacing: 10) {
                                Text("购买前请先通过 Apple 登录，以便在不同设备上保留这套写真。")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                AppleAccountButton()
                            }
                        }
                    } else {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("这张写真里的人，一眼看上去就是你吗？")
                                .font(.title2.bold())
                            Text("先确认是否像本人，再决定要不要完成整套写真。")
                                .foregroundStyle(.secondary)
                            Button {
                                Task { await model.confirmPreview(session: session) }
                            } label: {
                                Label("是的，这就是我", systemImage: "checkmark.circle.fill")
                            }
                            .buttonStyle(PrimaryActionButtonStyle())

                            if (model.project.previewRetriesRemaining ?? 0) > 0 {
                                Button {
                                    showingRetryChoice = true
                                } label: {
                                    if model.isRetryingPreview {
                                        ProgressView()
                                    } else {
                                        Label("还不太像", systemImage: "arrow.clockwise")
                                    }
                                }
                                .buttonStyle(.bordered)
                                .disabled(model.isRetryingPreview)
                            } else {
                                Text("这张预览仍然不够像你。先删除它并重新拍摄，不要急着购买。")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                Button(role: .destructive) {
                                    confirmDelete = true
                                } label: {
                                    Label("删除这张预览", systemImage: "trash")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                } else if model.project.status == .failed {
                    VStack(spacing: 18) {
                        ContentUnavailableView {
                            Label("生成已暂停", systemImage: "exclamationmark.triangle")
                        } description: {
                            Text(
                                model.project.failureMessage
                                    ?? "已完成的写真仍然安全。准备好后可以继续生成剩余照片。"
                            )
                        }
                        Button {
                            Task { await model.completeSet(session: session, store: store) }
                        } label: {
                            if model.isCompletingSet {
                                ProgressView().tint(.white)
                            } else {
                                Label("继续生成剩余写真", systemImage: "arrow.clockwise")
                            }
                        }
                        .buttonStyle(PrimaryActionButtonStyle())
                        .disabled(model.isCompletingSet)
                    }
                } else {
                    ProgressView("你的写真正在成形")
                        .frame(maxWidth: .infinity, minHeight: 260)
                }

                if let shareURL = model.shareURL {
                    ShareLink(item: shareURL) {
                        Label("打开分享面板", systemImage: "paperplane")
                    }
                }
                if let message = model.message {
                    Text(message).font(.footnote).foregroundStyle(.secondary)
                }
            }
            .padding(16)
            .flashShotContentWidth(920)
        }
        .navigationTitle("我的写真集")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(role: .destructive) {
                    if store.hasPendingPurchase(for: model.project.projectId) {
                        model.message = "这个项目有一笔购买正在等待 Apple 批准，暂时不能删除。"
                    } else {
                        confirmDelete = true
                    }
                } label: {
                    Image(systemName: "trash")
                }
                .accessibilityLabel("删除写真项目")
            }
        }
        .task { await model.monitor(session: session) }
        .onChange(of: model.isDeleted) { _, deleted in
            if deleted { dismiss() }
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
        .confirmationDialog(
            "保存不带可见 AI 标识的原图？",
            isPresented: $confirmCleanExport,
            titleVisibility: .visible
        ) {
            Button("同意并保存 \(model.assetIDs.count) 张照片") {
                Task { await model.saveCleanOriginals(session: session) }
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text(
                "这些写真由 AI 创作。分享无水印版本时，你同意在法律或平台要求时如实说明，"
                + "并且不将其用于误导他人。FlashShot 会将本次请求记录至少保留六个月。"
            )
        }
        .confirmationDialog(
            "删除这个项目及其中的写真？",
            isPresented: $confirmDelete,
            titleVisibility: .visible
        ) {
            Button("确认删除", role: .destructive) {
                Task { await model.delete(session: session) }
            }
        }
        .fullScreenCover(isPresented: Binding(
            get: { selectedImageIndex != nil },
            set: { if !$0 { selectedImageIndex = nil } }
        )) {
            ZStack {
                Color.black.ignoresSafeArea()
                if !model.images.isEmpty {
                    TabView(selection: Binding(
                        get: { selectedImageIndex ?? 0 },
                        set: { selectedImageIndex = $0 }
                    )) {
                        ForEach(Array(model.images.enumerated()), id: \.offset) { index, image in
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .tag(index)
                                .accessibilityLabel("第 \(index + 1) 张写真，共 \(model.images.count) 张")
                        }
                    }
                    .tabViewStyle(.page(indexDisplayMode: .always))
                }
            }
            .overlay(alignment: .top) {
                if let selectedImageIndex {
                    Text("第 \(selectedImageIndex + 1) 张，共 \(model.images.count) 张")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.top, 18)
                }
            }
            .overlay(alignment: .topTrailing) {
                Button("关闭大图", systemImage: "xmark.circle.fill") {
                    selectedImageIndex = nil
                }
                .labelStyle(.iconOnly)
                .font(.largeTitle)
                .foregroundStyle(.white)
                .padding()
            }
        }
        .sensoryFeedback(.selection, trigger: selectedImageIndex ?? -1)
    }

    private var storeProductUnavailableView: some View {
        VStack(alignment: .leading, spacing: 10) {
            if store.isLoadingProduct {
                ProgressView("正在连接 App Store")
            } else {
                Text(store.productLoadError ?? "这套写真暂时无法购买。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Button { Task { await store.loadProduct() } } label: {
                    Label("重新尝试", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
            }
        }
    }
}

private struct PortraitStorySequence: View {
    let images: [UIImage]
    let onSelect: (Int) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("一场写真 · \(images.count) 个瞬间")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(FlashShotStyle.accent)
                Text("按时间展开的这一段光景")
                    .font(.title3.bold())
                Text("第一束光，一点距离，一次停顿，然后是最后一个镜头。")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            if !images.isEmpty {
                storyImage(at: 0)
            }

            storyPair(indices: [1, 2])

            if images.indices.contains(3) {
                storyImage(at: 3)
            }

            storyPair(indices: [4, 5])
        }
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private func storyPair(indices: [Int]) -> some View {
        let available = indices.filter { images.indices.contains($0) }
        if !available.isEmpty {
            HStack(alignment: .top, spacing: 10) {
                ForEach(available, id: \.self) { index in
                    storyImage(at: index)
                }
            }
        }
    }

    private func storyImage(at index: Int) -> some View {
        Button { onSelect(index) } label: {
            Image(uiImage: images[index])
                .resizable()
                .scaledToFill()
                .aspectRatio(3 / 4, contentMode: .fit)
                .clipped()
                .clipShape(RoundedRectangle(cornerRadius: 5))
                .overlay(alignment: .bottomLeading) {
                    Text(String(format: "%02d", index + 1))
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(.black.opacity(0.52), in: RoundedRectangle(cornerRadius: 3))
                        .padding(9)
                }
        }
        .buttonStyle(PortraitCardButtonStyle())
        .accessibilityLabel("第 \(index + 1) 张写真，共 \(images.count) 张")
        .frame(maxWidth: .infinity)
        .flashShotEntrance(order: min(index + 1, 6))
    }
}
