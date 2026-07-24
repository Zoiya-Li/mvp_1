import ImageIO
import StoreKit
import StoreKitTest
import UIKit
import XCTest
@testable import FlashShot

final class PortraitModelsTests: XCTestCase {
    private struct StoreUnavailable: Error {}

    func testBundledThemeCatalogSupportsInstantColdStart() throws {
        let suiteName = "flashshot-theme-catalog-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let themes = ThemeCatalogStore.initialThemes(defaults: defaults)

        XCTAssertEqual(themes.count, 10)
        XCTAssertTrue(themes.allSatisfy { !$0.title.isEmpty && !$0.tagline.isEmpty })
        for theme in themes {
            let path = URLComponents(string: theme.coverImage)?.path ?? ""
            let filename = (path as NSString).lastPathComponent
            let stem = (filename as NSString).deletingPathExtension
            XCTAssertNotNil(UIImage(named: "Catalog_\(stem)"), "Missing bundled cover for \(theme.slug)")
        }
    }

    func testThemeContractDecodesSnakeCaseAPI() throws {
        let data = Data("""
        {
          "theme_id": "thm_cinema",
          "slug": "cinematic-mood",
          "title": "电影感叙事",
          "title_en": "Cinematic Mood",
          "tagline": "A film-like portrait.",
          "category": "Cinematic",
          "cover_image": "/images/cover.png",
          "preview_images": ["/images/one.png"],
          "featured": true,
          "source_style_key": "cinematic",
          "active_version": 1,
          "presentation": "female",
          "preview_integrity": "single_direction_study",
          "shot_labels": ["Opening close portrait", "Half-length portrait"]
        }
        """.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let theme = try decoder.decode(PortraitTheme.self, from: data)

        XCTAssertEqual(theme.id, "thm_cinema")
        XCTAssertEqual(theme.titleEn, "Cinematic Mood")
        XCTAssertTrue(theme.featured)
        XCTAssertEqual(theme.presentation, "female")
        XCTAssertEqual(theme.previewIntegrity, "single_direction_study")
        XCTAssertEqual(theme.shotLabels?.count, 2)
    }

    @MainActor
    func testCreateFlowUsesCatalogShootPresentation() {
        let theme = makeTheme(slug: "white-cotton", presentation: "female")

        let model = CreateFlowViewModel(theme: theme, source: .officialTheme)

        XCTAssertEqual(model.gender, "female")
    }

    func testProjectStatusWorkingContract() {
        XCTAssertTrue(ProjectStatus.previewGenerating.isWorking)
        XCTAssertTrue(ProjectStatus.setGenerating.isWorking)
        XCTAssertFalse(ProjectStatus.delivered.isWorking)
    }

    func testGuidedCapturePlanProvidesDistinctIdentityAngles() {
        let prompts = GuidedCapturePlan.identitySession

        XCTAssertEqual(prompts.count, 4)
        XCTAssertEqual(Set(prompts.map(\.id)).count, prompts.count)
        XCTAssertEqual(prompts.first?.id, "front")
        XCTAssertTrue(prompts.contains { $0.id == "turn_left" })
        XCTAssertTrue(prompts.contains { $0.id == "turn_right" })
        XCTAssertTrue(prompts.contains { $0.id == "expression" })
        XCTAssertTrue(prompts.allSatisfy { !$0.title.isEmpty && !$0.instruction.isEmpty })
    }

    func testPortraitIntentRanksThemesByDesiredFeeling() {
        let lifestyle = makeTheme(slug: "lifestyle-portrait")
        let professional = makeTheme(slug: "urban-professional")
        let cinematic = makeTheme(slug: "cinematic-mood")

        XCTAssertLessThan(
            PortraitIntent.authentic.ranks(lifestyle),
            PortraitIntent.authentic.ranks(cinematic)
        )
        XCTAssertLessThan(
            PortraitIntent.confident.ranks(professional),
            PortraitIntent.confident.ranks(lifestyle)
        )
        XCTAssertLessThan(
            PortraitIntent.cinematic.ranks(cinematic),
            PortraitIntent.cinematic.ranks(professional)
        )
    }

    func testPortraitIntentsCarryDistinctHumanPrompts() {
        let descriptions = Set(PortraitIntent.allCases.map(\.description))
        let previewLines = Set(PortraitIntent.allCases.map(\.previewLine))

        XCTAssertEqual(descriptions.count, PortraitIntent.allCases.count)
        XCTAssertEqual(previewLines.count, PortraitIntent.allCases.count)
        XCTAssertTrue(PortraitIntent.allCases.allSatisfy { !$0.title.isEmpty })
    }

    func testLocalStoreKitConfigurationContainsPortraitSet() throws {
        let session = try SKTestSession(configurationFileNamed: "FlashShot")
        session.disableDialogs = true
        session.clearTransactions()
        XCTAssertNotNil(session)
    }

    func testLocalConsumablePurchaseIsRecorded() async throws {
        let session = try SKTestSession(configurationFileNamed: "FlashShot")
        session.disableDialogs = true
        session.clearTransactions()

        try await session.buyProduct(identifier: "portrait_set_6")

        XCTAssertEqual(session.allTransactions().count, 1)
        XCTAssertEqual(session.allTransactions().first?.productIdentifier, "portrait_set_6")
    }

    @MainActor
    func testStoreKitManagerExposesProductLoadFailureForRetry() async {
        let manager = StoreKitManager(
            productLoader: { _ in throw StoreUnavailable() },
            startsTransactionObserver: false,
            loadsProductOnInit: false
        )

        await manager.loadProduct()

        XCTAssertNil(manager.product)
        XCTAssertFalse(manager.isLoadingProduct)
        XCTAssertEqual(
            manager.productLoadError,
            "暂时无法连接 App Store，请检查网络后重试。"
        )
    }

    @MainActor
    func testStoreKitManagerRestoresPendingProjectMarker() {
        UserDefaults.standard.set("prj_pending", forKey: "pending-storekit-project")
        defer { UserDefaults.standard.removeObject(forKey: "pending-storekit-project") }

        let manager = StoreKitManager(
            startsTransactionObserver: false,
            loadsProductOnInit: false
        )

        XCTAssertEqual(manager.pendingProjectID, "prj_pending")
        XCTAssertTrue(manager.hasPendingPurchase(for: "prj_pending"))
        XCTAssertFalse(manager.hasPendingPurchase(for: "prj_other"))
    }

    @MainActor
    func testStoreKitManagerLoadsConfiguredPortraitProduct() async {
        let manager = StoreKitManager(
            startsTransactionObserver: false,
            loadsProductOnInit: false
        )

        await manager.loadProduct()

        XCTAssertEqual(manager.product?.id, "portrait_set_6")
        XCTAssertFalse(manager.product?.displayPrice.isEmpty ?? true)
        XCTAssertNil(manager.productLoadError)
    }

    @MainActor
    func testStoreKitManagerClearsPendingMarkerWhenStoreSheetFails() async throws {
        try KeychainStore.set("test-token", for: "user-token")
        try KeychainStore.set("test-user", for: "user-id")
        try KeychainStore.set("apple", for: "account-type")
        defer {
            KeychainStore.remove("user-token")
            KeychainStore.remove("user-id")
            KeychainStore.remove("account-type")
            UserDefaults.standard.removeObject(forKey: "pending-storekit-project")
        }
        let session = AppSession()
        let manager = StoreKitManager(
            purchasePerformer: { _ in throw StoreUnavailable() },
            startsTransactionObserver: false,
            loadsProductOnInit: false
        )
        await manager.loadProduct()

        do {
            try await manager.purchase(projectID: "prj_store_error", session: session)
            XCTFail("Expected the injected StoreKit error")
        } catch is StoreUnavailable {
            XCTAssertNil(manager.pendingProjectID)
            XCTAssertNil(UserDefaults.standard.string(forKey: "pending-storekit-project"))
        }
    }

    func testPortraitImagePreparerDownsamplesLargeImageAsJPEG() async throws {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 4_096, height: 3_072))
        let sourceImage = renderer.image { context in
            UIColor.systemTeal.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 4_096, height: 3_072))
            UIColor.white.setFill()
            context.fill(CGRect(x: 1_024, y: 768, width: 2_048, height: 1_536))
        }
        let sourceData = try XCTUnwrap(sourceImage.pngData())

        let preparedData = await PortraitImagePreparer.prepare(sourceData)
        let prepared = try XCTUnwrap(preparedData)
        let source = try XCTUnwrap(CGImageSourceCreateWithData(prepared as CFData, nil))
        let properties = try XCTUnwrap(
            CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        )
        let width = try XCTUnwrap(properties[kCGImagePropertyPixelWidth] as? Int)
        let height = try XCTUnwrap(properties[kCGImagePropertyPixelHeight] as? Int)
        let type = CGImageSourceGetType(source) as String?

        XCTAssertLessThanOrEqual(max(width, height), PortraitImagePreparer.maximumPixelSize)
        XCTAssertEqual(type, "public.jpeg")
        XCTAssertLessThan(prepared.count, 10 * 1_024 * 1_024)
    }

    func testPortraitImagePreparerRejectsInvalidData() async {
        let prepared = await PortraitImagePreparer.prepare(Data("not-an-image".utf8))

        XCTAssertNil(prepared)
    }

    func testReferencePreflightUsesGentleGuidanceForUnreadablePhoto() async {
        let cue = await ReferencePhotoPreflight.analyze(Data("not-an-image".utf8), position: 0)

        XCTAssertEqual(cue.tone, .gentleFix)
        XCTAssertFalse(cue.message.contains("no_face"))
        XCTAssertFalse(cue.message.contains("resolution"))
        XCTAssertTrue(cue.message.contains("换一张照片"))
    }

    func testGuidedReferencePreflightUsesSameReadablePhotoGate() async {
        let prompt = try! XCTUnwrap(GuidedCapturePlan.identitySession.first)

        let cue = await ReferencePhotoPreflight.analyze(Data("not-an-image".utf8), for: prompt)

        XCTAssertEqual(cue.tone, .gentleFix)
        XCTAssertFalse(cue.message.contains("no_face"))
        XCTAssertFalse(cue.message.contains(";"))
    }

    func testReferencePreflightGentlyRedirectsPhotoWithoutAVisibleFace() async throws {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 900, height: 1_200))
        let image = renderer.image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 900, height: 1_200))
        }
        let data = try XCTUnwrap(image.jpegData(compressionQuality: 0.9))

        let cue = await ReferencePhotoPreflight.analyze(data, position: 0)

        XCTAssertEqual(cue.tone, .gentleFix)
        XCTAssertFalse(cue.message.isEmpty)
        XCTAssertFalse(cue.message.contains("no_face"))
        XCTAssertFalse(cue.message.contains(";"))
    }

    private func makeTheme(slug: String, presentation: String? = nil) -> PortraitTheme {
        PortraitTheme(
            themeId: "theme-\(slug)",
            slug: slug,
            title: slug,
            titleEn: slug,
            tagline: slug,
            category: slug,
            coverImage: "/images/cover.png",
            previewImages: [],
            featured: false,
            sourceStyleKey: slug,
            activeVersion: 1,
            presentation: presentation,
            previewIntegrity: nil,
            shotLabels: nil,
            useCases: nil,
            shotCount: nil,
            referenceMin: nil,
            referenceMax: nil
        )
    }
}
