import Foundation
import ImageIO
import UniformTypeIdentifiers

enum PortraitImagePreparer {
    static let maximumPixelSize = 2_048
    static let jpegQuality = 0.88

    static func prepare(_ sourceData: Data) async -> Data? {
        await Task.detached(priority: .userInitiated) {
            normalizedJPEG(from: sourceData)
        }.value
    }

    private static func normalizedJPEG(from sourceData: Data) -> Data? {
        guard let source = CGImageSourceCreateWithData(sourceData as CFData, nil) else {
            return nil
        }
        guard CGImageSourceGetCount(source) > 0, CGImageSourceGetType(source) != nil else {
            return nil
        }
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
            kCGImageSourceShouldCacheImmediately: false
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            thumbnailOptions as CFDictionary
        ) else {
            return nil
        }

        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else {
            return nil
        }
        let properties: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: jpegQuality
        ]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return output as Data
    }
}
