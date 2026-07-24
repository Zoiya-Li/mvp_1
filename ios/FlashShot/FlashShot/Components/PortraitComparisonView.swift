import SwiftUI
import UIKit

struct PortraitComparisonView: View {
    let source: UIImage
    let portrait: UIImage
    @State private var reveal: CGFloat = 0.62

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .leading) {
                Image(uiImage: source)
                    .resizable()
                    .scaledToFill()
                    .frame(width: width, height: geometry.size.height)
                    .clipped()

                Image(uiImage: portrait)
                    .resizable()
                    .scaledToFill()
                    .frame(width: width, height: geometry.size.height)
                    .clipped()
                    .mask(alignment: .leading) {
                        Rectangle().frame(width: max(0, width * reveal))
                    }

                Rectangle()
                    .fill(.white)
                    .frame(width: 2)
                    .offset(x: max(0, min(width - 2, width * reveal - 1)))
                    .shadow(color: .black.opacity(0.3), radius: 4)

                Image(systemName: "arrow.left.and.right.circle.fill")
                    .font(.title)
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(.white, FlashShotStyle.ink.opacity(0.72))
                    .offset(x: max(0, min(width - 34, width * reveal - 17)))

                VStack {
                    HStack {
                        comparisonLabel("写真")
                        Spacer()
                        comparisonLabel("原图")
                    }
                    Spacer()
                }
                .padding(12)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        reveal = min(0.94, max(0.06, value.location.x / max(width, 1)))
                    }
            )
        }
        .aspectRatio(3 / 4, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("对比原始照片与写真成片")
        .accessibilityValue("写真显示比例 \(Int(reveal * 100))%")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: reveal = min(0.94, reveal + 0.1)
            case .decrement: reveal = max(0.06, reveal - 0.1)
            @unknown default: break
            }
        }
    }

    private func comparisonLabel(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 4))
    }
}

struct PortraitBundlePeek: View {
    let portrait: UIImage
    let shotLabels: [String]?
    private let fallbackFrames = [
        ("开场肖像", "checkmark.seal.fill"),
        ("半身镜头", "person.crop.rectangle"),
        ("进入场景", "photo.on.rectangle.angled"),
        ("坐下片刻", "figure.seated.side"),
        ("侧面肖像", "person.crop.circle"),
        ("收尾瞬间", "figure.walk"),
    ]
    private let columns = Array(repeating: GridItem(.flexible(), spacing: 8), count: 3)

    private var frames: [(String, String)] {
        let labels = shotLabels ?? []
        return fallbackFrames.enumerated().map { index, fallback in
            guard labels.indices.contains(index), !labels[index].isEmpty else { return fallback }
            return (labels[index], fallback.1)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("这套写真的六个镜头")
                    .font(.headline)
                Text("首张已经确认像你。购买后才会创作另外五个全新构图。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: columns, spacing: 10) {
                ForEach(Array(frames.enumerated()), id: \.offset) { index, frame in
                    VStack(spacing: 5) {
                        Group {
                            if index == 0 {
                                Image(uiImage: portrait)
                                    .resizable()
                                    .scaledToFill()
                                    .overlay(alignment: .topTrailing) {
                                        Image(systemName: frame.1)
                                            .font(.caption.weight(.bold))
                                            .foregroundStyle(.white)
                                            .padding(7)
                                            .background(FlashShotStyle.jade, in: Circle())
                                            .padding(7)
                                    }
                            } else {
                                Rectangle()
                                    .fill(FlashShotStyle.secondaryPaper)
                                    .overlay {
                                        VStack(spacing: 8) {
                                            Image(systemName: frame.1)
                                                .font(.title2)
                                            Image(systemName: "lock.fill")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                        .foregroundStyle(FlashShotStyle.ink)
                                    }
                            }
                        }
                        .aspectRatio(3 / 4, contentMode: .fit)
                        .clipped()
                        .clipShape(RoundedRectangle(cornerRadius: 4))

                        Text(frame.0)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("完整写真包含一张已确认肖像和五张全新构图")
    }
}
