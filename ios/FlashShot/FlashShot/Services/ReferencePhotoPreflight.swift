import CoreGraphics
import Foundation
import ImageIO
import Vision

struct ReferencePhotoCue: Equatable, Sendable {
    enum Tone: Equatable, Sendable {
        case reading
        case encouraging
        case gentleFix
    }

    let tone: Tone
    let message: String
    let systemImage: String

    static let reading = ReferencePhotoCue(
        tone: .reading,
        message: "正在看清光线和你的五官",
        systemImage: "viewfinder"
    )
}

enum ReferencePhotoPreflight {
    static func analyze(_ data: Data, position: Int) async -> ReferencePhotoCue {
        await Task.detached(priority: .userInitiated) {
            analyzeSynchronously(data, position: position, guidedPrompt: nil)
        }.value
    }

    static func analyze(_ data: Data, for prompt: GuidedCapturePrompt) async -> ReferencePhotoCue {
        await Task.detached(priority: .userInitiated) {
            analyzeSynchronously(data, position: 0, guidedPrompt: prompt)
        }.value
    }

    private static func analyzeSynchronously(
        _ data: Data,
        position: Int,
        guidedPrompt: GuidedCapturePrompt?
    ) -> ReferencePhotoCue {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return gentleFix("换一张照片吧，这张暂时无法清晰读取", "photo.badge.exclamationmark")
        }

        let orientation = imageOrientation(from: source)
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation)
        do {
            try handler.perform([request])
        } catch {
            return gentleFix("请选择一张面部清晰可见的照片", "viewfinder.circle")
        }

        let faces = request.results ?? []
        if faces.isEmpty {
            return gentleFix("靠近一些拍，能更好地保留你的五官", "person.crop.rectangle")
        }
        if faces.count > 1 {
            return gentleFix("请选择单人照片，让最终成片只属于你", "person.crop.rectangle.badge.plus")
        }

        let face = faces[0]
        let faceArea = face.boundingBox.width * face.boundingBox.height
        if faceArea < 0.035 {
            return gentleFix("这张请让脸再靠近一点", "arrow.up.left.and.arrow.down.right")
        }
        if let luminance = averageLuminance(image), luminance < 0.20 {
            return gentleFix("光线再亮一点，会更好地看清你的神情", "sun.max")
        }

        let yaw = abs(face.yaw?.doubleValue ?? 0)
        if let guidedPrompt {
            return guidedCue(for: guidedPrompt, yaw: yaw)
        }
        if yaw > 0.12 {
            return encouraging("这个角度很好，能让五官更有层次", "camera.aperture")
        }

        let messages = [
            "清晰又自然，这是很好的正面参考",
            "这张光线很好，能清楚看见你的神情",
            "这个角度让五官更有层次",
            "很好的辅助角度，能让整套写真更像你",
            "这样的自然细节，会让成片更有真实感",
            "这张让我们又多看见了一点真实的你",
        ]
        return encouraging(messages[position % messages.count], "checkmark.circle.fill")
    }

    private static func guidedCue(for prompt: GuidedCapturePrompt, yaw: Double) -> ReferencePhotoCue {
        switch prompt.id {
        case "front":
            guard yaw < 0.16 else {
                return gentleFix("再转回镜头一点，让我们看清正面五官", "person.crop.square")
            }
            return encouraging("正面清晰，这是一张很好的身份参考", "checkmark.circle.fill")
        case "turn_left", "turn_right":
            guard yaw > 0.06 else {
                return gentleFix("再多转一点，同时保持双眼可见", "arrow.triangle.turn.up.right.circle")
            }
            return encouraging("这个角度很好，五官层次很清楚", "camera.aperture")
        case "expression":
            return encouraging("自然又清晰，这个神情会让整套照片更生动", "checkmark.circle.fill")
        default:
            return encouraging("清晰可用，很适合这套写真", "checkmark.circle.fill")
        }
    }

    private static func imageOrientation(from source: CGImageSource) -> CGImagePropertyOrientation {
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let rawValue = properties[kCGImagePropertyOrientation] as? UInt32,
              let orientation = CGImagePropertyOrientation(rawValue: rawValue) else {
            return .up
        }
        return orientation
    }

    private static func encouraging(_ message: String, _ systemImage: String) -> ReferencePhotoCue {
        ReferencePhotoCue(tone: .encouraging, message: message, systemImage: systemImage)
    }

    private static func gentleFix(_ message: String, _ systemImage: String) -> ReferencePhotoCue {
        ReferencePhotoCue(tone: .gentleFix, message: message, systemImage: systemImage)
    }

    private static func averageLuminance(_ image: CGImage) -> Double? {
        let width = 24
        let height = 24
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        context.interpolationQuality = .low
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        var total = 0.0
        for index in stride(from: 0, to: pixels.count, by: 4) {
            total += 0.2126 * Double(pixels[index])
                + 0.7152 * Double(pixels[index + 1])
                + 0.0722 * Double(pixels[index + 2])
        }
        return total / Double(width * height) / 255.0
    }
}
