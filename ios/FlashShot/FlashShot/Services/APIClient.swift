import Foundation

enum APIError: LocalizedError {
    case invalidURL
    case invalidResponse
    case server(Int, String)
    case missingToken

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "服务地址无效。"
        case .invalidResponse: return "FlashShot 返回了无法读取的数据，请稍后重试。"
        case .server(_, let message): return message
        case .missingToken: return "暂时无法恢复你的私人创作空间。"
        }
    }
}

final class APIClient: @unchecked Sendable {
    static let shared = APIClient()

    private let session: URLSession
    private let baseURL: URL
    private let decoder: JSONDecoder
    private let encoder = JSONEncoder()

    var mediaBaseURL: URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: true)
        components?.path = "/"
        components?.query = nil
        components?.fragment = nil
        return components?.url ?? URL(string: "https://flashshot.top/")!
    }

    init(session: URLSession = .shared, baseURL: URL? = nil) {
        self.session = session
        let configured = ProcessInfo.processInfo.environment["FLASHSHOT_API_BASE_URL"]
            ?? Bundle.main.object(forInfoDictionaryKey: "API_BASE_URL") as? String
        let rawBase = configured ?? "https://flashshot.top/api"
        self.baseURL = baseURL ?? URL(string: rawBase.hasSuffix("/") ? rawBase : rawBase + "/")!
        self.decoder = JSONDecoder()
        self.decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.encoder.keyEncodingStrategy = .convertToSnakeCase
    }

    private func url(_ path: String) throws -> URL {
        guard let value = URL(string: path, relativeTo: baseURL) else { throw APIError.invalidURL }
        return value
    }

    private func request(
        _ path: String,
        method: String = "GET",
        token: String? = nil,
        json: [String: Any]? = nil
    ) throws -> URLRequest {
        var request = URLRequest(url: try url(path))
        request.httpMethod = method
        request.timeoutInterval = 60
        request.setValue(KeychainStore.stableDeviceID(), forHTTPHeaderField: "X-Device-ID")
        if let token { request.setValue(token, forHTTPHeaderField: "X-User-Token") }
        if let json {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: json)
        }
        return request
    }

    private func perform<T: Decodable>(_ request: URLRequest, as type: T.Type = T.self) async throws -> T {
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        do { return try decoder.decode(T.self, from: data) }
        catch { throw APIError.invalidResponse }
    }

    private func validate(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            var message = HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let detail = object["detail"] {
                if let text = detail as? String { message = text }
                else if let object = detail as? [String: Any], let code = object["code"] as? String {
                    message = Self.userMessage(for: code)
                }
            }
            throw APIError.server(http.statusCode, message)
        }
    }

    private static func userMessage(for code: String) -> String {
        switch code {
        case "reference_quality_failed":
            return "有些照片还不够适合生成，请按提示更换后再试。"
        case "reference_identity_mismatch":
            return "这些照片看起来不像同一个人，请重新选择。"
        case "inspiration_required":
            return "请先选择一张灵感照片。"
        case "payment_required":
            return "请先解锁这套写真。"
        case "project_not_found":
            return "没有找到这次创作，它可能已被删除。"
        case "generation_in_progress":
            return "这套写真仍在制作中，请稍后查看。"
        case "delivery_gate_failed":
            return "这次成片没有达到交付标准，我们会为你保留重试机会。"
        case "rate_limited":
            return "操作有些频繁，请稍后再试。"
        default:
            return "操作暂时无法完成，请稍后重试。"
        }
    }

    func fetchThemes() async throws -> [PortraitTheme] {
        let response: ThemeListResponse = try await perform(request("v2/themes"))
        return response.themes
    }

    func createGuest() async throws -> GuestIdentity {
        try await perform(request("v2/users/guest", method: "POST"))
    }

    func authenticateApple(
        token: String, identityToken: String, rawNonce: String, displayName: String?
    ) async throws -> AuthenticatedIdentity {
        var body: [String: Any] = [
            "identity_token": identityToken,
            "raw_nonce": rawNonce
        ]
        if let displayName { body["display_name"] = displayName }
        return try await perform(request(
            "v2/auth/apple",
            method: "POST",
            token: token,
            json: body
        ))
    }

    func createProject(
        token: String, source: ProjectSource, themeID: String?, gender: String = "unspecified",
        sharedRecipeID: String? = nil
    ) async throws -> PortraitProject {
        var body: [String: Any] = ["source": source.rawValue, "gender": gender]
        if let themeID { body["theme_id"] = themeID }
        if let sharedRecipeID { body["shared_recipe_id"] = sharedRecipeID }
        return try await perform(request("v2/projects", method: "POST", token: token, json: body))
    }

    func projects(token: String) async throws -> [PortraitProject] {
        let response: ProjectListResponse = try await perform(request("v2/projects", token: token))
        return response.projects
    }

    func project(_ id: String, token: String) async throws -> PortraitProject {
        try await perform(request("v2/projects/\(id)", token: token))
    }

    func uploadInspiration(projectID: String, token: String, image: Data) async throws -> InspirationUploadResponse {
        let fields = ["rights_confirmed": "true", "private_style_reference_only": "true"]
        let files = [(name: "file", filename: "inspiration.jpg", mime: "image/jpeg", data: image)]
        return try await multipart("v2/projects/\(projectID)/inspiration", token: token, fields: fields, files: files)
    }

    func uploadReferences(
        projectID: String, token: String, images: [Data], gender: String
    ) async throws -> ReferenceUploadResponse {
        let fields = [
            "gender": gender,
            "face_processing_consent": "true",
            "adult_subject_confirmed": "true"
        ]
        let files = images.enumerated().map {
            (name: "files", filename: "reference-\($0.offset + 1).jpg", mime: "image/jpeg", data: $0.element)
        }
        return try await multipart("v2/projects/\(projectID)/references", token: token, fields: fields, files: files)
    }

    func startPreview(projectID: String, token: String) async throws -> [JobResponse] {
        try await perform(request("v2/projects/\(projectID)/preview", method: "POST", token: token))
    }

    func retryPreview(
        projectID: String,
        token: String,
        reason: String
    ) async throws -> PreviewRetryResponse {
        try await perform(request(
            "v2/projects/\(projectID)/preview/retry",
            method: "POST",
            token: token,
            json: ["reason": reason]
        ))
    }

    func confirmPreview(projectID: String, token: String) async throws -> PreviewConfirmationResponse {
        try await perform(request(
            "v2/projects/\(projectID)/preview/confirm",
            method: "POST",
            token: token
        ))
    }

    func claimApplePurchase(projectID: String, token: String, signedTransaction: String) async throws -> ApplePurchaseClaim {
        try await perform(request(
            "v2/projects/\(projectID)/apple-purchases/claim",
            method: "POST",
            token: token,
            json: ["signed_transaction": signedTransaction]
        ))
    }

    func unlock(projectID: String, token: String) async throws -> [JobResponse] {
        try await perform(request("v2/projects/\(projectID)/unlock", method: "POST", token: token))
    }

    func shareRecipe(projectID: String, token: String, includePortrait: Bool = false) async throws -> SharedRecipe {
        try await perform(request(
            "v2/projects/\(projectID)/share-recipe",
            method: "POST",
            token: token,
            json: ["include_portrait": includePortrait]
        ))
    }

    func sharedRecipe(_ token: String) async throws -> SharedRecipe {
        try await perform(request("v2/shares/\(token)"))
    }

    func heroData(projectID: String, token: String) async throws -> Data {
        let request = try request("v2/projects/\(projectID)/hero", token: token)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        return data
    }

    func photoSet(projectID: String, setID: String, token: String) async throws -> PortraitPhotoSet {
        try await perform(request("v2/projects/\(projectID)/sets/\(setID)", token: token))
    }

    func assetData(projectID: String, assetID: String, token: String) async throws -> Data {
        let request = try request("v2/projects/\(projectID)/assets/\(assetID)", token: token)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        return data
    }

    func cleanAssetData(projectID: String, assetID: String, token: String) async throws -> Data {
        let request = try request(
            "v2/projects/\(projectID)/assets/\(assetID)/clean-export",
            method: "POST",
            token: token,
            json: [
                "terms_version": "cn-ai-label-2025-09-v1",
                "ai_generated_acknowledged": true,
                "redistribution_responsibility_accepted": true
            ]
        )
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
        return data
    }

    func deleteProject(_ id: String, token: String) async throws {
        let request = try request("v2/projects/\(id)", method: "DELETE", token: token)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
    }

    func deleteAccount(token: String) async throws {
        let request = try request("v2/users/me", method: "DELETE", token: token)
        let (data, response) = try await session.data(for: request)
        try validate(response, data: data)
    }

    private func multipart<T: Decodable>(
        _ path: String,
        token: String,
        fields: [String: String],
        files: [(name: String, filename: String, mime: String, data: Data)]
    ) async throws -> T {
        let boundary = "FlashShot-\(UUID().uuidString)"
        var body = Data()
        for (name, value) in fields {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            body.append("\(value)\r\n")
        }
        for file in files {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"\(file.name)\"; filename=\"\(file.filename)\"\r\n")
            body.append("Content-Type: \(file.mime)\r\n\r\n")
            body.append(file.data)
            body.append("\r\n")
        }
        body.append("--\(boundary)--\r\n")
        var request = try request(path, method: "POST", token: token)
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 180
        request.httpBody = body
        return try await perform(request)
    }
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}
