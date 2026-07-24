import AVFoundation
import SwiftUI
import UIKit

struct GuidedPortraitCapture: View {
    private enum Phase {
        case capture
        case review
    }

    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @StateObject private var camera = GuidedCameraController()
    @State private var phase: Phase = .capture
    @State private var currentIndex = 0
    @State private var photos = Array<Data?>(repeating: nil, count: GuidedCapturePlan.identitySession.count)
    @State private var countdown: Int?
    @State private var countdownTask: Task<Void, Never>?
    @State private var validationTask: Task<Void, Never>?
    @State private var isValidatingCapture = false
    @State private var captureCue: ReferencePhotoCue?
    @State private var captureError: String?

    let onComplete: ([Data]) -> Void

    private var prompts: [GuidedCapturePrompt] { GuidedCapturePlan.identitySession }
    private var completedCount: Int { photos.compactMap { $0 }.count }

    var body: some View {
        NavigationStack {
            Group {
                switch phase {
                case .capture:
                    captureView
                case .review:
                    reviewView
                }
            }
            .animation(FlashShotMotion.gentle(reduceMotion: reduceMotion), value: phase)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") { dismiss() }
                }
            }
        }
        .onAppear { camera.prepare() }
        .onDisappear {
            countdownTask?.cancel()
            validationTask?.cancel()
            camera.stop()
        }
    }

    private var captureView: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch camera.state {
            case .ready:
                cameraExperience
            case .denied:
                cameraMessage(
                    title: "相机权限未开启",
                    detail: "请前往系统设置允许相机访问，或者返回选择相册中的照片。",
                    symbol: "camera.fill"
                )
            case .unavailable:
                cameraMessage(
                    title: "相机暂时不可用",
                    detail: "这台设备无法启动前置相机，你仍然可以使用相册中的照片。",
                    symbol: "camera.badge.ellipsis"
                )
            case .failed(let message):
                cameraMessage(title: "相机需要重试", detail: message, symbol: "exclamationmark.triangle")
            default:
                ProgressView("正在准备相机")
                    .tint(.white)
                    .foregroundStyle(.white)
            }
        }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbarColorScheme(.dark, for: .navigationBar)
    }

    private var cameraExperience: some View {
        GeometryReader { proxy in
            ZStack {
                GuidedCameraPreview(session: camera.session)
                    .ignoresSafeArea()

                LinearGradient(
                    colors: [.black.opacity(0.68), .clear, .black.opacity(0.82)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                faceGuide(in: proxy.size)

                VStack(spacing: 0) {
                    captureHeader
                    Spacer()
                    captureControls
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 22)

                if let countdown {
                    Text(String(countdown))
                        .font(.system(size: 78, weight: .medium, design: .rounded))
                        .foregroundStyle(.white)
                        .contentTransition(.numericText())
                        .shadow(color: .black.opacity(0.35), radius: 10)
                        .accessibilityLabel("将在 \(countdown) 秒后拍照")
                }
            }
        }
    }

    private var captureHeader: some View {
        let prompt = prompts[currentIndex]
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("第 \(currentIndex + 1) 张，共 \(prompts.count) 张")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white.opacity(0.78))
                Spacer()
                Label("私人拍摄", systemImage: "lock.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.78))
            }

            ProgressView(value: Double(completedCount), total: Double(prompts.count))
                .tint(.white)

            Label(prompt.title, systemImage: prompt.systemImage)
                .font(.title2.bold())
                .foregroundStyle(.white)
            Text(prompt.instruction)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.84))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 12)
    }

    private var captureControls: some View {
        VStack(spacing: 14) {
            if isValidatingCapture {
                Label("正在检查光线、清晰度和角度", systemImage: "viewfinder")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.white)
            } else if let captureCue {
                Label(captureCue.message, systemImage: captureCue.systemImage)
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
            } else if let captureError {
                Text(captureError)
                    .font(.footnote)
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
            } else {
                Text("轻点一次，然后保持姿势三秒。")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.74))
            }

            Button(action: beginCountdown) {
                ZStack {
                    Circle()
                        .stroke(.white, lineWidth: 4)
                        .frame(width: 76, height: 76)
                    Circle()
                        .fill(.white)
                        .frame(width: 62, height: 62)
                    if countdown != nil || camera.isCapturing || isValidatingCapture {
                        ProgressView()
                            .tint(FlashShotStyle.ink)
                    }
                }
            }
            .disabled(countdown != nil || camera.isCapturing || isValidatingCapture)
            .accessibilityLabel("开始三秒倒计时拍照")

            HStack(spacing: 8) {
                ForEach(prompts.indices, id: \.self) { index in
                    Circle()
                        .fill(photos[index] == nil ? .white.opacity(0.32) : .white)
                        .frame(width: 7, height: 7)
                }
            }
            .accessibilityLabel("已完成 \(completedCount) 张，共 \(prompts.count) 张")
        }
    }

    private var reviewView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 7) {
                    Text("你的拍摄素材")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(FlashShotStyle.accent)
                    Text("四个真实瞬间，足够认识你")
                        .font(.largeTitle.bold())
                    Text("留下最像你的神情。开始写真创作前，每一张都可以重拍。")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 14) {
                    ForEach(prompts.indices, id: \.self) { index in
                        reviewTile(index: index)
                    }
                }

                Label(
                    "这些照片会保持私密，只用于创作本次写真。",
                    systemImage: "lock.shield.fill"
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            }
            .padding(18)
            .padding(.bottom, 104)
            .flashShotContentWidth(620)
        }
        .safeAreaInset(edge: .bottom) {
            Button("使用这四张照片") {
                onComplete(photos.compactMap { $0 })
                dismiss()
            }
            .buttonStyle(PrimaryActionButtonStyle())
            .disabled(completedCount != prompts.count)
            .padding(.horizontal, 18)
            .padding(.vertical, 10)
            .background(.bar)
        }
        .navigationTitle("确认照片")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func reviewTile(index: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let data = photos[index], let image = UIImage(data: data) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .aspectRatio(3 / 4, contentMode: .fit)
                    .clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                Rectangle()
                    .fill(FlashShotStyle.secondaryPaper)
                    .aspectRatio(3 / 4, contentMode: .fit)
                    .overlay { Image(systemName: "camera").foregroundStyle(.secondary) }
            }

            Text(prompts[index].title)
                .font(.subheadline.weight(.semibold))
                .lineLimit(2)
            Button("重拍", systemImage: "arrow.clockwise") {
                currentIndex = index
                captureCue = nil
                captureError = nil
                phase = .capture
                camera.prepare()
            }
            .font(.footnote.weight(.semibold))
        }
    }

    private func faceGuide(in size: CGSize) -> some View {
        let width = min(size.width * 0.63, 280)
        return RoundedRectangle(cornerRadius: width * 0.43)
            .stroke(.white.opacity(0.72), style: StrokeStyle(lineWidth: 1.5, dash: [7, 7]))
            .frame(width: width, height: width * 1.28)
            .offset(y: -10)
            .accessibilityHidden(true)
    }

    private func cameraMessage(title: String, detail: String, symbol: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: symbol)
                .font(.system(size: 44))
            Text(title).font(.title.bold())
            Text(detail)
                .font(.body)
                .foregroundStyle(.white.opacity(0.78))
                .multilineTextAlignment(.center)
            Button("从相册选择") { dismiss() }
                .buttonStyle(.borderedProminent)
                .tint(.white)
                .foregroundStyle(FlashShotStyle.ink)
        }
        .foregroundStyle(.white)
        .padding(28)
    }

    private func beginCountdown() {
        guard countdown == nil, !camera.isCapturing, !isValidatingCapture else { return }
        captureError = nil
        captureCue = nil
        countdownTask?.cancel()
        countdownTask = Task { @MainActor in
            for value in stride(from: 3, through: 1, by: -1) {
                countdown = value
                UIImpactFeedbackGenerator(style: .soft).impactOccurred()
                try? await Task.sleep(for: .seconds(1))
                guard !Task.isCancelled else { return }
            }
            countdown = nil
            camera.capture { result in
                switch result {
                case .success(let data):
                    validateCapturedPhoto(data)
                case .failure(let error):
                    captureError = error.localizedDescription
                }
            }
        }
    }

    private func validateCapturedPhoto(_ data: Data) {
        let index = currentIndex
        let prompt = prompts[index]
        isValidatingCapture = true
        validationTask?.cancel()
        validationTask = Task { @MainActor in
            let cue = await ReferencePhotoPreflight.analyze(data, for: prompt)
            guard !Task.isCancelled else { return }
            isValidatingCapture = false
            captureCue = cue

            guard cue.tone != .gentleFix else {
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
                return
            }

            photos[index] = data
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            try? await Task.sleep(for: .milliseconds(650))
            guard !Task.isCancelled else { return }
            captureCue = nil
            advanceAfterCapture()
        }
    }

    private func advanceAfterCapture() {
        if let next = prompts.indices.first(where: { photos[$0] == nil }) {
            currentIndex = next
        } else {
            phase = .review
            camera.stop()
        }
    }
}

private struct GuidedCameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> GuidedCameraPreviewView {
        let view = GuidedCameraPreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        if let connection = view.previewLayer.connection,
           connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = true
        }
        return view
    }

    func updateUIView(_ uiView: GuidedCameraPreviewView, context: Context) {
        uiView.previewLayer.session = session
        if let connection = uiView.previewLayer.connection,
           connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = true
        }
    }
}

private final class GuidedCameraPreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}
