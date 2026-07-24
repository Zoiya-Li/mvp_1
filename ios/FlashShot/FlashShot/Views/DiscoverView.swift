import SwiftUI

enum ThemeCatalogStore {
    private static let cacheKey = "portrait-theme-catalog-v1"

    static func initialThemes(
        bundle: Bundle = .main,
        defaults: UserDefaults = .standard
    ) -> [PortraitTheme] {
        if let data = defaults.data(forKey: cacheKey),
           let response = try? decoder.decode(ThemeListResponse.self, from: data),
           !response.themes.isEmpty {
            return response.themes
        }
        guard let url = bundle.url(forResource: "ThemeCatalog", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let response = try? decoder.decode(ThemeListResponse.self, from: data) else {
            return []
        }
        return response.themes
    }

    static func save(_ themes: [PortraitTheme], defaults: UserDefaults = .standard) {
        guard !themes.isEmpty,
              let data = try? encoder.encode(ThemeListResponse(themes: themes)) else { return }
        defaults.set(data, forKey: cacheKey)
    }

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()
}

@MainActor
final class DiscoverViewModel: ObservableObject {
    @Published var themes: [PortraitTheme] = []
    @Published var error: String?
    @Published var isLoading = false
    private let api: APIClient
    private let persistsCatalog: Bool
    private var loadedRemoteCatalog = false

    init(api: APIClient = .shared) {
        self.api = api
        self.persistsCatalog = api === APIClient.shared
        self.themes = persistsCatalog ? ThemeCatalogStore.initialThemes() : []
    }

    func load(force: Bool = false) async {
        guard !isLoading, force || !loadedRemoteCatalog else { return }
        isLoading = themes.isEmpty
        defer { isLoading = false }
        do {
            let freshThemes = try await api.fetchThemes()
            themes = freshThemes
            error = nil
            loadedRemoteCatalog = true
            if persistsCatalog { ThemeCatalogStore.save(freshThemes) }
        }
        catch { self.error = error.localizedDescription }
    }
}

struct DiscoverView: View {
    @StateObject private var model = DiscoverViewModel()

    private var orderedThemes: [PortraitTheme] {
        model.themes
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 7) {
                    Text("FLASHSHOT")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(FlashShotStyle.accent)
                    Text("今天，想拍怎样的自己？")
                        .font(.largeTitle.bold())
                        .fixedSize(horizontal: false, vertical: true)
                    Text("每个主题都包含完整的光线、服装、场景与六张连贯成片。")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
                .flashShotEntrance(order: 0)

                if model.isLoading && model.themes.isEmpty {
                    ProgressView().frame(maxWidth: .infinity, minHeight: 260)
                } else if let error = model.error, model.themes.isEmpty {
                    ContentUnavailableView {
                        Label("暂时无法加载写真主题", systemImage: "wifi.exclamationmark")
                    } description: { Text(error) } actions: {
                        Button("重新加载") { Task { await model.load() } }
                            .buttonStyle(.borderedProminent)
                    }
                } else {
                    LazyVStack(spacing: 26) {
                        ForEach(Array(orderedThemes.enumerated()), id: \.element.id) { index, theme in
                            NavigationLink(value: theme) { ThemeCard(theme: theme) }
                                .buttonStyle(PortraitCardButtonStyle())
                                .accessibilityIdentifier("theme-card-\(theme.slug)")
                                .flashShotEntrance(order: min(index + 1, 7))
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 18)
            .flashShotContentWidth(1180)
        }
        .navigationTitle("写真")
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(for: PortraitTheme.self) {
            ThemeDetailView(theme: $0, intent: .authentic)
        }
        .task { await model.load() }
        .refreshable {
            await model.load(force: true)
        }
    }
}

private struct ThemeCard: View {
    let theme: PortraitTheme

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            GeometryReader { proxy in
                RemotePortraitImage(path: preferredCover)
                    .frame(width: proxy.size.width, height: proxy.size.height)
                    .clipped()
            }
            .aspectRatio(4 / 3, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(theme.featured ? "本期精选" : theme.category)
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(FlashShotStyle.accent)
                    Text(theme.title)
                        .font(.title3.bold())
                        .foregroundStyle(FlashShotStyle.ink)
                        .lineLimit(2)
                    Text(theme.tagline)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer(minLength: 8)
                Image(systemName: "arrow.right")
                    .font(.headline)
                    .foregroundStyle(FlashShotStyle.ink)
                    .padding(.top, 4)
            }
        }
        .accessibilityElement(children: .combine)
        .scrollTransition(.interactive, axis: .vertical) { content, phase in
            content
                .opacity(phase.isIdentity ? 1 : 0.82)
        }
    }

    private var preferredCover: String {
        theme.coverImage
    }
}
