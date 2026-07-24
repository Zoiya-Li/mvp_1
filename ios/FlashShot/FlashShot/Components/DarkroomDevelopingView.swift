import SwiftUI
import UIKit

struct DarkroomDevelopingView: View {
    enum Mode {
        case readingPhotos
        case firstPortrait
        case closerMatch
        case fullSet

        var messages: [String] {
            switch self {
            case .readingPhotos:
                [
                    "正在为你打开一间私人影棚",
                    "正在读懂光线落在五官上的样子",
                    "正在挑选最能保留你神态的角度",
                ]
            case .firstPortrait:
                [
                    "正在让光线靠近你的神情",
                    "正在构图，同时把你留在画面里",
                    "揭晓前，正在再次确认是否像你",
                ]
            case .closerMatch:
                [
                    "正在认真理解刚才哪里不够像",
                    "正在把你熟悉的五官细节带回来",
                    "替换之前，正在重新审阅这张照片",
                ]
            case .fullSet:
                [
                    "正在为你的故事安排下一个镜头",
                    "正在保持光线、服装和人物的一致",
                    "每一张，都在经过同样严格的终审",
                ]
            }
        }

        var eyebrow: String {
            switch self {
            case .readingPhotos: "正在认识你"
            case .firstPortrait: "第一张写真"
            case .closerMatch: "再靠近真实的你"
            case .fullSet: "完整写真创作中"
            }
        }
    }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let image: UIImage?
    let mode: Mode
    @State private var messageIndex = 0
    @State private var developing = false

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(red: 0.10, green: 0.085, blue: 0.075))
                        .frame(width: 280, height: 360)
                    if let image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 244, height: 304)
                            .clipped()
                            .saturation(developing || reduceMotion ? 0.88 : 0.25)
                            .blur(radius: developing || reduceMotion ? 5 : 16)
                            .opacity(developing || reduceMotion ? 0.78 : 0.42)
                    } else {
                        Rectangle()
                            .fill(FlashShotStyle.secondaryPaper)
                            .frame(width: 244, height: 304)
                    }
                    LinearGradient(
                        colors: [.clear, FlashShotStyle.accent.opacity(0.20), .clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                    .frame(width: 244, height: 90)
                    .offset(y: developing && !reduceMotion ? 108 : -108)
                    .blendMode(.screen)
                }
                .shadow(color: FlashShotStyle.ink.opacity(0.18), radius: 20, y: 12)
                .animation(
                    reduceMotion ? nil : .easeInOut(duration: 3.8).repeatForever(autoreverses: true),
                    value: developing
                )
                .accessibilityHidden(true)

                VStack(spacing: 9) {
                    Text(mode.eyebrow)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(FlashShotStyle.accent)
                    Text(mode.messages[messageIndex])
                        .font(.title2.bold())
                        .multilineTextAlignment(.center)
                        .contentTransition(.opacity)
                    Text("在它仍然像你之前，没有一张照片会离开影棚。")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: 540)

                HStack(spacing: 7) {
                    ForEach(mode.messages.indices, id: \.self) { index in
                        Capsule()
                            .fill(index == messageIndex ? FlashShotStyle.accent : Color.secondary.opacity(0.22))
                            .frame(width: index == messageIndex ? 26 : 8, height: 4)
                    }
                }
                .animation(FlashShotMotion.quick(reduceMotion: reduceMotion), value: messageIndex)

                Text("你可以随时离开，我们会继续创作，完成后会出现在“相册”里。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(28)
            .frame(maxWidth: .infinity)
        }
        .task(id: mode.eyebrow) {
            developing = true
            guard !reduceMotion else { return }
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard !Task.isCancelled else { return }
                withAnimation(FlashShotMotion.gentle(reduceMotion: false)) {
                    messageIndex = (messageIndex + 1) % mode.messages.count
                }
            }
        }
    }
}
