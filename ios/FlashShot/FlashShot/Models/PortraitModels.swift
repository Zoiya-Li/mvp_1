import Foundation

struct PortraitTheme: Codable, Identifiable, Hashable {
    let themeId: String
    let slug: String
    let title: String
    let titleEn: String
    let tagline: String
    let category: String
    let coverImage: String
    let previewImages: [String]
    let featured: Bool
    let sourceStyleKey: String
    let activeVersion: Int
    let presentation: String?
    let previewIntegrity: String?
    let shotLabels: [String]?
    let useCases: [String]?
    let shotCount: Int?
    let referenceMin: Int?
    let referenceMax: Int?

    var id: String { themeId }
}

struct ThemeListResponse: Codable {
    let themes: [PortraitTheme]
}

struct GuestIdentity: Codable {
    let userId: String
    let accessToken: String
    let createdAt: String
}

struct AuthenticatedIdentity: Codable {
    let userId: String
    let accessToken: String
    let accountType: String
    let mergedGuestWorkspace: Bool
    let createdAt: String
}

enum ProjectSource: String, Codable, CaseIterable {
    case officialTheme = "official_theme"
    case privateInspiration = "private_inspiration"
    case sharedRecipe = "shared_recipe"
}

enum ProjectStatus: String, Codable {
    case draft
    case awaitingReferences = "awaiting_references"
    case ready
    case previewGenerating = "preview_generating"
    case previewReady = "preview_ready"
    case setGenerating = "set_generating"
    case delivered
    case failed

    var isWorking: Bool {
        self == .previewGenerating || self == .setGenerating
    }
}

struct PortraitProject: Codable, Identifiable, Hashable {
    let projectId: String
    let userId: String
    let themeId: String?
    let source: ProjectSource
    let status: ProjectStatus
    let gender: String
    let inspirationAssetId: String?
    let heroAssetId: String?
    let photoSetId: String?
    let legacySessionId: String?
    let failureCode: String?
    let failureMessage: String?
    let previewRetriesUsed: Int?
    let previewRetriesRemaining: Int?
    let previewConfirmed: Bool?
    let createdAt: String?
    let updatedAt: String?

    var id: String { projectId }
}

struct ProjectListResponse: Codable {
    let projects: [PortraitProject]
}

struct ReferenceQuality: Codable {
    let pass: Bool?
    let issues: [String]?
    let roleCoverage: [ReferenceRoleFeedback]?
}

struct ReferenceRoleFeedback: Codable, Hashable {
    let role: String
    let filename: String?
    let pass: Bool
    let issues: [String]
    let headline: String?
    let guidance: String?

    var title: String {
        switch role {
        case "front_neutral": "正面自然照"
        case "front_smile": "正面微笑照"
        case "left_45": "左侧角度照"
        case "right_45": "右侧角度照"
        case "profile": "侧面照"
        default: "参考照片"
        }
    }

    var problemTitle: String {
        headline ?? "这个角度需要换一张照片"
    }

    var nextStep: String {
        guidance ?? "请选择一张清晰、光线充足的单人照片后重试。"
    }
}

struct ReferenceUploadResponse: Codable {
    let projectId: String
    let legacySessionId: String
    let referenceCount: Int
    let status: ProjectStatus
    let referenceQuality: ReferenceQuality
}

struct PreviewRetryResponse: Codable {
    let projectId: String
    let status: ProjectStatus
    let retriesRemaining: Int
}

struct PreviewConfirmationResponse: Codable {
    let feedbackId: String
    let sessionId: String
    let imageId: String
    let event: String
}

struct InspirationUploadResponse: Codable {
    let assetId: String
    let projectId: String
    let analysisStatus: String
    let message: String
}

struct JobResponse: Codable, Identifiable {
    let jobId: String
    let sessionId: String
    let jobType: String
    let status: String
    let progress: Double
    let error: String?

    var id: String { jobId }
}

struct PortraitSetAsset: Codable, Identifiable, Hashable {
    let assetId: String
    let mimeType: String
    let position: Int

    var id: String { assetId }
}

struct PortraitPhotoSet: Codable, Identifiable {
    let photoSetId: String
    let projectId: String
    let title: String
    let status: String
    let coverAssetId: String?
    let assets: [PortraitSetAsset]
    let createdAt: String
    let deliveredAt: String

    var id: String { photoSetId }
}

struct ApplePurchaseClaim: Codable {
    let orderId: String
    let projectId: String
    let productId: String
    let transactionId: String
    let status: String
    let newlyClaimed: Bool
}

struct SharedRecipe: Codable {
    let shareToken: String
    let title: String
    let themeId: String?
    let themeSlug: String?
    let source: ProjectSource
    let portraitAvailable: Bool
}
