import Foundation
import StoreKit

enum PurchaseError: LocalizedError {
    case productUnavailable
    case unverified
    case cancelled
    case pending
    case accountRequired
    case purchaseInProgress

    var errorDescription: String? {
        switch self {
        case .productUnavailable: return "这套写真暂时无法购买，请稍后重试。"
        case .unverified: return "Apple 暂时无法验证这笔购买。"
        case .cancelled: return "购买已取消。"
        case .pending: return "这笔购买正在等待批准。"
        case .accountRequired: return "购买前请先通过 Apple 登录，以便在其他设备上恢复写真。"
        case .purchaseInProgress: return "已有一笔购买正在处理中。"
        }
    }
}

@MainActor
final class StoreKitManager: ObservableObject {
    static let productID = "portrait_set_6"

    @Published private(set) var product: Product?
    @Published private(set) var isPurchasing = false
    @Published private(set) var isLoadingProduct = false
    @Published private(set) var productLoadError: String?
    @Published private(set) var pendingProjectID: String?

    private let api: APIClient
    private let productLoader: @Sendable ([String]) async throws -> [Product]
    private let purchasePerformer: @Sendable (Product) async throws -> Product.PurchaseResult
    private var updatesTask: Task<Void, Never>?

    init(
        api: APIClient = .shared,
        productLoader: @escaping @Sendable ([String]) async throws -> [Product] = {
            try await Product.products(for: $0)
        },
        purchasePerformer: @escaping @Sendable (Product) async throws -> Product.PurchaseResult = {
            try await $0.purchase()
        },
        startsTransactionObserver: Bool = true,
        loadsProductOnInit: Bool = true
    ) {
        self.api = api
        self.productLoader = productLoader
        self.purchasePerformer = purchasePerformer
        self.pendingProjectID = UserDefaults.standard.string(forKey: "pending-storekit-project")
        if startsTransactionObserver { updatesTask = observeTransactions() }
        if loadsProductOnInit { Task { await loadProduct() } }
    }

    deinit { updatesTask?.cancel() }

    func loadProduct() async {
        guard !isLoadingProduct else { return }
        isLoadingProduct = true
        productLoadError = nil
        defer { isLoadingProduct = false }
        do {
            product = try await productLoader([Self.productID]).first
            if product == nil {
                productLoadError = PurchaseError.productUnavailable.localizedDescription
            }
        } catch {
            product = nil
            productLoadError = "暂时无法连接 App Store，请检查网络后重试。"
        }
    }

    func purchase(projectID: String, session: AppSession) async throws {
        guard !isPurchasing else { throw PurchaseError.purchaseInProgress }
        guard session.isAppleAccount, let token = session.token else {
            throw PurchaseError.accountRequired
        }
        guard let product else { throw PurchaseError.productUnavailable }
        isPurchasing = true
        defer { isPurchasing = false }
        setPendingProject(projectID)
        let result: Product.PurchaseResult
        do {
            result = try await purchasePerformer(product)
        } catch {
            clearPendingProject()
            throw error
        }
        switch result {
        case .success(let verification):
            guard case .verified(let transaction) = verification else {
                clearPendingProject()
                throw PurchaseError.unverified
            }
            _ = try await api.claimApplePurchase(
                projectID: projectID,
                token: token,
                signedTransaction: verification.jwsRepresentation
            )
            _ = try await api.unlock(projectID: projectID, token: token)
            await transaction.finish()
            clearPendingProject()
        case .userCancelled:
            clearPendingProject()
            throw PurchaseError.cancelled
        case .pending:
            throw PurchaseError.pending
        @unknown default:
            clearPendingProject()
            throw PurchaseError.unverified
        }
    }

    func hasPendingPurchase(for projectID: String) -> Bool {
        pendingProjectID == projectID
    }

    private func setPendingProject(_ projectID: String) {
        UserDefaults.standard.set(projectID, forKey: "pending-storekit-project")
        pendingProjectID = projectID
    }

    private func clearPendingProject() {
        UserDefaults.standard.removeObject(forKey: "pending-storekit-project")
        pendingProjectID = nil
    }

    private func observeTransactions() -> Task<Void, Never> {
        Task { [weak self] in
            for await verification in Transaction.updates {
                guard !Task.isCancelled,
                      let self,
                      case .verified(let transaction) = verification,
                      transaction.productID == Self.productID,
                      let projectID = self.pendingProjectID
                else { continue }
                guard let token = KeychainStore.string(for: "user-token") else { continue }
                do {
                    _ = try await self.api.claimApplePurchase(
                        projectID: projectID,
                        token: token,
                        signedTransaction: verification.jwsRepresentation
                    )
                    _ = try await self.api.unlock(projectID: projectID, token: token)
                    await transaction.finish()
                    self.clearPendingProject()
                } catch {
                    // Keep the transaction unfinished so StoreKit redelivers it.
                }
            }
        }
    }
}
