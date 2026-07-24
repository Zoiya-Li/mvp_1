import Photos
import SwiftUI
import UIKit

enum PortraitKeepsakeRenderer {
    static func watermarked(_ image: UIImage) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))

            let width = image.size.width
            let padding = width * 0.045
            let shadow: NSShadow = {
                let shadow = NSShadow()
                shadow.shadowColor = UIColor.black.withAlphaComponent(0.45)
                shadow.shadowBlurRadius = width * 0.008
                shadow.shadowOffset = .zero
                return shadow
            }()
            let brand = NSAttributedString(
                string: "FlashShot",
                attributes: [
                    .font: UIFont.systemFont(ofSize: width * 0.052, weight: .semibold),
                    .foregroundColor: UIColor.white.withAlphaComponent(0.72),
                    .shadow: shadow,
                ]
            )
            let note = NSAttributedString(
                string: "免费预览",
                attributes: [
                    .font: UIFont.systemFont(ofSize: width * 0.030, weight: .medium),
                    .foregroundColor: UIColor.white.withAlphaComponent(0.60),
                    .shadow: shadow,
                ]
            )

            let brandSize = brand.size()
            let noteSize = note.size()
            let noteRect = CGRect(
                x: width - padding - noteSize.width,
                y: image.size.height - padding - noteSize.height,
                width: noteSize.width,
                height: noteSize.height
            )
            let brandRect = CGRect(
                x: width - padding - brandSize.width,
                y: noteRect.minY - width * 0.010 - brandSize.height,
                width: brandSize.width,
                height: brandSize.height
            )
            brand.draw(in: brandRect)
            note.draw(in: noteRect)
        }
    }
}

struct PortraitKeepsakeButton: View {
    let image: UIImage
    @State private var isSaving = false
    @State private var message: String?
    @State private var saved = false

    var body: some View {
        VStack(spacing: 8) {
            Button {
                Task { await save() }
            } label: {
                if isSaving {
                    ProgressView()
                } else {
                    Label("保存纪念预览", systemImage: "square.and.arrow.down")
                }
            }
            .buttonStyle(.bordered)
            .disabled(isSaving)
            if let message {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(saved ? FlashShotStyle.jade : .secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .contain)
    }

    @MainActor
    private func save() async {
        isSaving = true
        defer { isSaving = false }
        let authorization = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard authorization == .authorized || authorization == .limited else {
            saved = false
            message = "请在系统设置中允许访问相册，才能保存这张纪念预览。"
            return
        }
        let keepsake = PortraitKeepsakeRenderer.watermarked(image)
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: keepsake)
            }
            saved = true
            message = "带水印的纪念预览已保存到相册。"
        } catch {
            saved = false
            message = "刚才没能保存到相册，请再试一次。"
        }
    }
}
