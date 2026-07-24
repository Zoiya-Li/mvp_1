import Foundation

@MainActor
final class AppSession: ObservableObject {
    @Published private(set) var token: String?
    @Published private(set) var userID: String?
    @Published private(set) var isAppleAccount = false
    @Published var incomingShareToken: String?

    private let api: APIClient

    init(api: APIClient = .shared) {
        self.api = api
        self.token = KeychainStore.string(for: "user-token")
        self.userID = KeychainStore.string(for: "user-id")
        self.isAppleAccount = KeychainStore.string(for: "account-type") == "apple"
    }

    func ensureGuest() async throws -> String {
        if let token { return token }
        let guest = try await api.createGuest()
        try persist(token: guest.accessToken, userID: guest.userId, accountType: "guest")
        return guest.accessToken
    }

    func authenticate(with payload: AppleAuthorizationPayload) async throws {
        let currentToken = try await ensureGuest()
        let identity = try await api.authenticateApple(
            token: currentToken,
            identityToken: payload.identityToken,
            rawNonce: payload.rawNonce,
            displayName: payload.displayName
        )
        try persist(token: identity.accessToken, userID: identity.userId, accountType: "apple")
    }

    func deleteAccount() async throws {
        guard let token else { return }
        try await api.deleteAccount(token: token)
        KeychainStore.remove("user-token")
        KeychainStore.remove("user-id")
        KeychainStore.remove("account-type")
        self.token = nil
        self.userID = nil
        self.isAppleAccount = false
    }

    func handle(url: URL) {
        let parts = url.pathComponents
        if let index = parts.firstIndex(of: "s"), parts.indices.contains(index + 1) {
            incomingShareToken = parts[index + 1]
        }
    }

    private func persist(token: String, userID: String, accountType: String) throws {
        try KeychainStore.set(token, for: "user-token")
        try KeychainStore.set(userID, for: "user-id")
        try KeychainStore.set(accountType, for: "account-type")
        self.token = token
        self.userID = userID
        self.isAppleAccount = accountType == "apple"
    }
}
