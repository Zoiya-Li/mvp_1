import ImageIO
import OSLog
import SwiftUI
import UIKit

@MainActor
final class RemotePortraitImageLoader: ObservableObject {
    @Published private(set) var image: UIImage?
    @Published private(set) var isLoading = false
    @Published private(set) var failed = false

    private static let logger = Logger(subsystem: "com.flashshot.app", category: "RemotePortraitImage")
    private static let cache: NSCache<NSURL, UIImage> = {
        let cache = NSCache<NSURL, UIImage>()
        cache.countLimit = 64
        cache.totalCostLimit = 96 * 1_024 * 1_024
        return cache
    }()
    private static let defaultSession = URLSession.shared

    private let session: URLSession
    private let retryDelaysNanoseconds: [UInt64]
    private var currentURL: URL?

    init(
        session: URLSession? = nil,
        retryDelaysNanoseconds: [UInt64] = [350_000_000, 1_000_000_000, 2_500_000_000]
    ) {
        self.session = session ?? Self.defaultSession
        self.retryDelaysNanoseconds = retryDelaysNanoseconds
    }

    func load(url: URL?) async {
        guard let url else {
            image = nil
            failed = true
            return
        }
        if currentURL == url, image != nil || isLoading { return }
        currentURL = url
        failed = false

        if let cached = Self.cache.object(forKey: url as NSURL) {
            image = cached
            return
        }

        image = nil
        isLoading = true
        defer { if currentURL == url { isLoading = false } }

        var lastError: Error?
        for attempt in 0...retryDelaysNanoseconds.count {
            guard !Task.isCancelled, currentURL == url else { return }
            do {
                var request = URLRequest(url: url)
                request.timeoutInterval = 30
                request.cachePolicy = .returnCacheDataElseLoad
                request.setValue("image/avif,image/webp,image/png,image/jpeg,*/*", forHTTPHeaderField: "Accept")

                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw RemotePortraitImageError.invalidResponse
                }
                guard (200..<300).contains(http.statusCode) else {
                    throw RemotePortraitImageError.httpStatus(http.statusCode)
                }
                if let mimeType = http.mimeType, !mimeType.hasPrefix("image/") {
                    throw RemotePortraitImageError.invalidContentType(mimeType)
                }
                guard let decodedImage = await Task.detached(priority: .utility, operation: {
                    Self.decode(data)
                }).value else {
                    throw RemotePortraitImageError.decodingFailed
                }
                let decoded = UIImage(
                    cgImage: decodedImage,
                    scale: UIScreen.main.scale,
                    orientation: .up
                )
                guard !Task.isCancelled, currentURL == url else { return }
                Self.cache.setObject(decoded, forKey: url as NSURL, cost: data.count)
                image = decoded
                failed = false
                return
            } catch {
                guard !Task.isCancelled else { return }
                lastError = error
                guard attempt < retryDelaysNanoseconds.count else { break }
                do {
                    try await Task.sleep(nanoseconds: retryDelaysNanoseconds[attempt])
                } catch {
                    return
                }
            }
        }

        guard currentURL == url else { return }
        failed = true
        let errorDescription = lastError?.localizedDescription ?? "unknown error"
        Self.logger.error(
            "Image load failed after retries: \(url.absoluteString, privacy: .public), \(errorDescription, privacy: .public)"
        )
    }

    nonisolated private static func decode(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: 1_600
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }
}

private enum RemotePortraitImageError: LocalizedError {
    case invalidResponse
    case httpStatus(Int)
    case invalidContentType(String)
    case decodingFailed

    var errorDescription: String? {
        switch self {
        case .invalidResponse: "图片服务返回了无法读取的数据。"
        case .httpStatus(let status): "图片服务返回错误状态 \(status)。"
        case .invalidContentType(let type): "图片服务返回了不支持的格式：\(type)。"
        case .decodingFailed: "下载的图片无法显示。"
        }
    }
}

struct RemotePortraitImage: View {
    let path: String
    var contentMode: ContentMode = .fill
    @StateObject private var loader = RemotePortraitImageLoader()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Group {
            if let image = loader.image ?? bundledFallback {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
                    .transition(.opacity)
            } else if loader.failed {
                placeholder(systemImage: "photo.badge.exclamationmark")
            } else {
                ZStack {
                    FlashShotStyle.secondaryPaper
                    ProgressView()
                }
            }
        }
        .animation(
            FlashShotMotion.quick(reduceMotion: reduceMotion),
            value: loader.image != nil
        )
        .task(id: resolvedURL) { await loader.load(url: resolvedURL) }
    }

    var resolvedURL: URL? {
        if path.hasPrefix("http") { return URL(string: path) }
        return URL(string: path, relativeTo: APIClient.shared.mediaBaseURL)?.absoluteURL
    }

    private var bundledFallback: UIImage? {
        guard let components = URLComponents(string: path),
              components.path.contains("/catalog-images/") else { return nil }
        let filename = (components.path as NSString).lastPathComponent
        let stem = (filename as NSString).deletingPathExtension
        return UIImage(named: "Catalog_\(stem)")
    }

    private func placeholder(systemImage: String) -> some View {
        ZStack {
            FlashShotStyle.secondaryPaper
            Image(systemName: systemImage)
                .font(.title2)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("写真图片加载失败")
        .accessibilityIdentifier("portrait-image-load-failure")
    }
}
