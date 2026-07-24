import AuthenticationServices
import SwiftUI

struct AppleAccountButton: View {
    @EnvironmentObject private var session: AppSession
    @StateObject private var handler = AppleButtonSignInHandler()
    @State private var errorMessage: String?
    @State private var isWorking = false

    var onAuthenticated: (() -> Void)?

    var body: some View {
        VStack(spacing: 10) {
            SignInWithAppleButton(.continue) { request in
                handler.configure(request)
            } onCompletion: { result in
                Task {
                    isWorking = true
                    defer { isWorking = false }
                    do {
                        try await session.authenticate(with: handler.payload(from: result))
                        onAuthenticated?()
                    } catch {
                        errorMessage = error.localizedDescription
                    }
                }
            }
            .signInWithAppleButtonStyle(.black)
            .frame(height: 50)
            .disabled(isWorking)

            if isWorking { ProgressView().controlSize(.small) }
            if let errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(FlashShotStyle.accent)
                    .multilineTextAlignment(.center)
            }
        }
    }
}
