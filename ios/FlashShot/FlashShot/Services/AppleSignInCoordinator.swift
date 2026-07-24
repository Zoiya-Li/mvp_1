import AuthenticationServices
import CryptoKit
import Foundation
import Security
import UIKit

struct AppleAuthorizationPayload {
    let identityToken: String
    let rawNonce: String
    let displayName: String?
}

enum AppleSignInError: LocalizedError {
    case unavailable
    case invalidCredential

    var errorDescription: String? {
        switch self {
        case .unavailable: return "暂时无法使用 Apple 登录。"
        case .invalidCredential: return "Apple 未返回有效的身份凭证。"
        }
    }
}

@MainActor
final class AppleSignInCoordinator: NSObject, ASAuthorizationControllerDelegate, ASAuthorizationControllerPresentationContextProviding {
    private var continuation: CheckedContinuation<AppleAuthorizationPayload, Error>?
    private var rawNonce: String?
    private var controller: ASAuthorizationController?

    func signIn() async throws -> AppleAuthorizationPayload {
        let nonce = try Self.randomNonce()
        rawNonce = nonce
        let request = ASAuthorizationAppleIDProvider().createRequest()
        request.requestedScopes = [.fullName, .email]
        request.nonce = Self.sha256(nonce)
        let controller = ASAuthorizationController(authorizationRequests: [request])
        controller.delegate = self
        controller.presentationContextProvider = self
        self.controller = controller
        return try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            controller.performRequests()
        }
    }

    func presentationAnchor(for controller: ASAuthorizationController) -> ASPresentationAnchor {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        return scenes.flatMap(\.windows).first(where: \.isKeyWindow) ?? ASPresentationAnchor()
    }

    func authorizationController(controller: ASAuthorizationController, didCompleteWithAuthorization authorization: ASAuthorization) {
        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential,
              let data = credential.identityToken,
              let token = String(data: data, encoding: .utf8),
              let nonce = rawNonce else {
            finish(.failure(AppleSignInError.invalidCredential))
            return
        }
        let formatter = PersonNameComponentsFormatter()
        let displayName = credential.fullName.map { formatter.string(from: $0) }
        finish(.success(AppleAuthorizationPayload(
            identityToken: token,
            rawNonce: nonce,
            displayName: displayName?.isEmpty == false ? displayName : nil
        )))
    }

    func authorizationController(controller: ASAuthorizationController, didCompleteWithError error: Error) {
        finish(.failure(error))
    }

    private func finish(_ result: Result<AppleAuthorizationPayload, Error>) {
        continuation?.resume(with: result)
        continuation = nil
        rawNonce = nil
        controller = nil
    }

    fileprivate static func sha256(_ input: String) -> String {
        SHA256.hash(data: Data(input.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    fileprivate static func randomNonce(length: Int = 32) throws -> String {
        let characters = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz-._")
        var result = ""
        var remaining = length
        while remaining > 0 {
            var bytes = [UInt8](repeating: 0, count: 16)
            guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
                throw AppleSignInError.unavailable
            }
            for byte in bytes where remaining > 0 && byte < characters.count {
                result.append(characters[Int(byte)])
                remaining -= 1
            }
        }
        return result
    }
}

@MainActor
final class AppleButtonSignInHandler: ObservableObject {
    private var rawNonce: String?

    func configure(_ request: ASAuthorizationAppleIDRequest) {
        guard let nonce = try? AppleSignInCoordinator.randomNonce() else { return }
        rawNonce = nonce
        request.requestedScopes = [.fullName, .email]
        request.nonce = AppleSignInCoordinator.sha256(nonce)
    }

    func payload(from result: Result<ASAuthorization, Error>) throws -> AppleAuthorizationPayload {
        let authorization = try result.get()
        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential,
              let data = credential.identityToken,
              let token = String(data: data, encoding: .utf8),
              let rawNonce else {
            throw AppleSignInError.invalidCredential
        }
        let formatter = PersonNameComponentsFormatter()
        let displayName = credential.fullName.map { formatter.string(from: $0) }
        self.rawNonce = nil
        return AppleAuthorizationPayload(
            identityToken: token,
            rawNonce: rawNonce,
            displayName: displayName?.isEmpty == false ? displayName : nil
        )
    }
}
