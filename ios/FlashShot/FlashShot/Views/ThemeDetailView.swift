import SwiftUI

struct ThemeDetailView: View {
    let theme: PortraitTheme
    let intent: PortraitIntent
    @State private var creating = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    init(theme: PortraitTheme, intent: PortraitIntent = .authentic) {
        self.theme = theme
        self.intent = intent
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                TabView {
                    ForEach(seriesImages, id: \.self) { image in
                        RemotePortraitImage(path: image)
                            .frame(maxWidth: 560)
                            .aspectRatio(3 / 4, contentMode: .fit)
                            .clipped()
                            .frame(maxWidth: .infinity)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: seriesImages.count > 1 ? .automatic : .never))
                .frame(height: 560)
                .flashShotEntrance(order: 0)

                VStack(alignment: .leading, spacing: 10) {
                    Text("整套写真预览")
                        .font(.caption.bold())
                        .foregroundStyle(FlashShotStyle.accent)
                    Text(theme.title).font(.largeTitle.bold())
                    Text(theme.tagline).font(.title3).foregroundStyle(.secondary)
                    Label("一个完整主题 · 六个精心设计的镜头", systemImage: "rectangle.stack")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
                    if let labels = theme.shotLabels, !labels.isEmpty {
                        LazyVGrid(
                            columns: [GridItem(.flexible()), GridItem(.flexible())],
                            alignment: .leading,
                            spacing: 10
                        ) {
                            ForEach(Array(labels.enumerated()), id: \.offset) { index, label in
                                HStack(alignment: .top, spacing: 8) {
                                    Text(String(index + 1))
                                        .font(.caption2.bold())
                                        .foregroundStyle(.white)
                                        .frame(width: 20, height: 20)
                                        .background(FlashShotStyle.ink, in: Circle())
                                    Text(label)
                                        .font(.caption.weight(.medium))
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                        .padding(.top, 6)
                    }
                    if let uses = theme.useCases, !uses.isEmpty {
                        LazyVGrid(
                            columns: [GridItem(.adaptive(minimum: 112), alignment: .leading)],
                            alignment: .leading,
                            spacing: 7
                        ) {
                            ForEach(uses, id: \.self) { use in
                                Label(use, systemImage: "checkmark")
                                    .font(.caption.weight(.medium))
                                    .lineLimit(2)
                            }
                        }
                    }
                }
                .padding(.horizontal, 16)
                .flashShotEntrance(order: 1)
            }
            .padding(.bottom, 112)
            .flashShotContentWidth(760)
        }
        .ignoresSafeArea(edges: .top)
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 5) {
                Button("开始拍这套写真") { creating = true }
                    .buttonStyle(PrimaryActionButtonStyle())
                Text("首张经过质检的写真免费。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.top, 10)
            .padding(.bottom, 6)
            .background(.bar)
        }
        .toolbar(.hidden, for: .tabBar)
        .sheet(isPresented: $creating) {
            NavigationStack {
                CreateFlowView(theme: theme, source: .officialTheme, intent: intent)
            }
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: creating) { _, newValue in
            newValue
        }
    }

    private var seriesImages: [String] {
        let images = theme.previewImages.isEmpty ? [theme.coverImage] : theme.previewImages
        return Array(images.reduce(into: [String]()) { result, image in
            if !image.isEmpty && !result.contains(image) { result.append(image) }
        }.prefix(3))
    }
}
