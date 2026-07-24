import Foundation
import UIKit
import XCTest
@testable import FlashShot

private final class MockURLProtocol: URLProtocol {
    static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: APIError.invalidResponse)
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private func requestBodyData(_ request: URLRequest) throws -> Data {
    if let body = request.httpBody { return body }
    guard let stream = request.httpBodyStream else { throw APIError.invalidResponse }
    stream.open()
    defer { stream.close() }
    var data = Data()
    var buffer = [UInt8](repeating: 0, count: 4_096)
    while true {
        let count = stream.read(&buffer, maxLength: buffer.count)
        if count < 0 { throw stream.streamError ?? APIError.invalidResponse }
        if count == 0 { break }
        data.append(contentsOf: buffer.prefix(count))
    }
    return data
}

@MainActor
final class APIClientTests: XCTestCase {
    override func tearDown() {
        MockURLProtocol.handler = nil
        super.tearDown()
    }

    func testDiscoverViewModelLoadsRealV2ThemeContract() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let api = APIClient(session: session, baseURL: URL(string: "https://example.test/api/")!)
        MockURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/v2/themes")
            XCTAssertEqual(request.httpMethod, "GET")
            XCTAssertNotNil(request.value(forHTTPHeaderField: "X-Device-ID"))
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = Data("""
            {"themes":[{
              "theme_id":"thm_test","slug":"cinema","title":"电影感",
              "title_en":"Cinema","tagline":"Directed light.",
              "category":"Editorial","cover_image":"/cover.png",
              "preview_images":[],"featured":true,
              "source_style_key":"cinematic","active_version":1
            }]}
            """.utf8)
            return (response, data)
        }
        let model = DiscoverViewModel(api: api)

        await model.load()

        XCTAssertNil(model.error)
        XCTAssertEqual(model.themes.map(\.id), ["thm_test"])
        XCTAssertFalse(model.isLoading)
    }

    func testRemotePortraitImageResolvesRootRelativeProductionURL() {
        let image = RemotePortraitImage(path: "/images/cover.png")

        XCTAssertEqual(image.resolvedURL?.absoluteString, "https://flashshot.top/images/cover.png")
    }

    func testAPIClientDerivesMediaOriginFromConfiguredAPIBase() {
        let api = APIClient(baseURL: URL(string: "http://127.0.0.1:8001/api/")!)

        XCTAssertEqual(api.mediaBaseURL.absoluteString, "http://127.0.0.1:8001/")
    }

    func testRemotePortraitImageLoaderRetriesTransientFailure() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let imageData = try XCTUnwrap(
            UIGraphicsImageRenderer(size: CGSize(width: 4, height: 4))
                .image { context in
                    UIColor.systemRed.setFill()
                    context.fill(CGRect(x: 0, y: 0, width: 4, height: 4))
                }
                .pngData()
        )
        var requestCount = 0
        MockURLProtocol.handler = { request in
            requestCount += 1
            if requestCount == 1 { throw URLError(.networkConnectionLost) }
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: "HTTP/2",
                headerFields: ["Content-Type": "image/png"]
            )!
            return (response, imageData)
        }
        let loader = RemotePortraitImageLoader(
            session: session,
            retryDelaysNanoseconds: [0]
        )

        await loader.load(url: URL(string: "https://example.test/images/retry.png"))

        XCTAssertEqual(requestCount, 2)
        XCTAssertNotNil(loader.image)
        XCTAssertFalse(loader.failed)
        XCTAssertFalse(loader.isLoading)
    }

    func testRetryPreviewUsesFeedbackConditionedV2RecoveryContract() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let api = APIClient(session: session, baseURL: URL(string: "https://example.test/api/")!)
        MockURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/v2/projects/prj_retry/preview/retry")
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-User-Token"), "token")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = Data("""
            {
              "project_id":"prj_retry","status":"preview_generating",
              "retries_remaining":0,"jobs":[]
            }
            """.utf8)
            return (response, data)
        }

        let response = try await api.retryPreview(
            projectID: "prj_retry",
            token: "token",
            reason: "identity"
        )

        XCTAssertEqual(response.projectId, "prj_retry")
        XCTAssertEqual(response.status, .previewGenerating)
        XCTAssertEqual(response.retriesRemaining, 0)
    }

    func testConfirmPreviewRecordsExplicitIdentityDecision() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let api = APIClient(session: session, baseURL: URL(string: "https://example.test/api/")!)
        MockURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/v2/projects/prj_confirm/preview/confirm")
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-User-Token"), "token")
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = Data("""
            {
              "feedback_id":"fb_confirm","session_id":"s_confirm",
              "image_id":"img_confirm","event":"looks_like_me",
              "reason":"preview_confirmed","score":2,"created_at":"2026-07-20T12:00:00Z"
            }
            """.utf8)
            return (response, data)
        }

        let response = try await api.confirmPreview(
            projectID: "prj_confirm",
            token: "token"
        )

        XCTAssertEqual(response.feedbackId, "fb_confirm")
        XCTAssertEqual(response.imageId, "img_confirm")
        XCTAssertEqual(response.event, "looks_like_me")
    }

    func testCleanExportSendsExplicitDisclosureConsent() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let api = APIClient(session: session, baseURL: URL(string: "https://example.test/api/")!)
        let expected = Data("clean portrait".utf8)
        MockURLProtocol.handler = { request in
            XCTAssertEqual(
                request.url?.path,
                "/api/v2/projects/prj_clean/assets/ast_clean/clean-export"
            )
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-User-Token"), "token")
            let body = try requestBodyData(request)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: body) as? [String: Any]
            )
            XCTAssertEqual(json["terms_version"] as? String, "cn-ai-label-2025-09-v1")
            XCTAssertEqual(json["ai_generated_acknowledged"] as? Bool, true)
            XCTAssertEqual(json["redistribution_responsibility_accepted"] as? Bool, true)
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil,
                headerFields: ["Content-Type": "image/png"]
            )!
            return (response, expected)
        }

        let data = try await api.cleanAssetData(
            projectID: "prj_clean",
            assetID: "ast_clean",
            token: "token"
        )

        XCTAssertEqual(data, expected)
    }

    func testCreateFlowDeletesDraftWhenReferenceGateFails() async throws {
        KeychainStore.remove("user-token")
        KeychainStore.remove("user-id")
        KeychainStore.remove("account-type")
        defer {
            KeychainStore.remove("user-token")
            KeychainStore.remove("user-id")
            KeychainStore.remove("account-type")
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let api = APIClient(session: session, baseURL: URL(string: "https://example.test/api/")!)
        var deletedDraft = false
        MockURLProtocol.handler = { request in
            let path = request.url!.path
            let status: Int
            let data: Data
            switch (request.httpMethod, path) {
            case ("POST", "/api/v2/users/guest"):
                status = 200
                data = Data("""
                {"user_id":"usr_test","access_token":"token_test","created_at":"now"}
                """.utf8)
            case ("POST", "/api/v2/projects"):
                status = 200
                data = Data("""
                {
                  "project_id":"prj_draft","user_id":"usr_test","theme_id":"thm_test",
                  "source":"official_theme","status":"draft","gender":"female",
                  "inspiration_asset_id":null,"hero_asset_id":null,"photo_set_id":null,
                  "legacy_session_id":null,"failure_code":null,"failure_message":null,
                  "created_at":"now","updated_at":"now"
                }
                """.utf8)
            case ("POST", "/api/v2/projects/prj_draft/references"):
                status = 200
                data = Data("""
                {
                  "project_id":"prj_draft","legacy_session_id":"session_test",
                  "reference_count":4,"status":"awaiting_references",
                  "reference_quality":{
                    "pass":false,"issues":["front:needs_quality_reference"],
                    "role_coverage":[{
                      "role":"front","filename":"reference-1.jpg","pass":false,
                      "issues":["face_too_small"]
                    }]
                  }
                }
                """.utf8)
            case ("DELETE", "/api/v2/projects/prj_draft"):
                deletedDraft = true
                status = 204
                data = Data()
            default:
                XCTFail("Unexpected request: \(request.httpMethod ?? "nil") \(path)")
                status = 500
                data = Data()
            }
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, data)
        }
        let appSession = AppSession(api: api)
        let model = CreateFlowViewModel(theme: nil, source: .officialTheme, api: api)
        model.referenceImages = Array(repeating: Data("image".utf8), count: 4)
        model.consent = true
        model.adultConfirmed = true

        await model.checkReferences(session: appSession)

        XCTAssertTrue(deletedDraft)
        XCTAssertNil(model.project)
        XCTAssertEqual(model.stage, .references)
        XCTAssertEqual(model.qualityFeedback.first?.role, "front")
        XCTAssertEqual(model.errorMessage, "请替换标记出的照片，然后重新检查。")
    }
}
