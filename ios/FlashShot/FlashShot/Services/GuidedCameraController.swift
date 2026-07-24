import AVFoundation
import Foundation

final class GuidedCameraController: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    enum CameraState: Equatable {
        case idle
        case requestingPermission
        case preparing
        case ready
        case denied
        case unavailable
        case failed(String)
    }

    @Published private(set) var state: CameraState = .idle
    @Published private(set) var isCapturing = false

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "com.flashshot.guided-camera")
    private let output = AVCapturePhotoOutput()
    private var configured = false
    private var captureCompletion: ((Result<Data, Error>) -> Void)?

    func prepare() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            state = .requestingPermission
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if granted {
                        self.configureAndStart()
                    } else {
                        self.state = .denied
                    }
                }
            }
        case .denied, .restricted:
            state = .denied
        @unknown default:
            state = .unavailable
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    func capture(completion: @escaping (Result<Data, Error>) -> Void) {
        guard state == .ready, !isCapturing else { return }
        isCapturing = true
        captureCompletion = completion

        let settings = AVCapturePhotoSettings()
        settings.flashMode = .off
        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = false
            }
        }
        output.capturePhoto(with: settings, delegate: self)
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        let result: Result<Data, Error>
        if let error {
            result = .failure(error)
        } else if let data = photo.fileDataRepresentation() {
            result = .success(data)
        } else {
            result = .failure(GuidedCameraError.missingPhotoData)
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.isCapturing = false
            let completion = self.captureCompletion
            self.captureCompletion = nil
            completion?(result)
        }
    }

    private func configureAndStart() {
        state = .preparing
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                if !self.configured {
                    try self.configureSession()
                }
                guard !self.session.isRunning else {
                    self.publish(.ready)
                    return
                }
                self.session.startRunning()
                self.publish(.ready)
            } catch let error as GuidedCameraError {
                self.publish(error == .cameraUnavailable ? .unavailable : .failed(error.localizedDescription))
            } catch {
                self.publish(.failed(error.localizedDescription))
            }
        }
    }

    private func configureSession() throws {
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            throw GuidedCameraError.cameraUnavailable
        }

        let input = try AVCaptureDeviceInput(device: camera)
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .photo

        guard session.canAddInput(input), session.canAddOutput(output) else {
            throw GuidedCameraError.configurationFailed
        }
        session.addInput(input)
        session.addOutput(output)
        output.maxPhotoQualityPrioritization = .quality
        configured = true
    }

    private func publish(_ newState: CameraState) {
        DispatchQueue.main.async { [weak self] in
            self?.state = newState
        }
    }
}

private enum GuidedCameraError: LocalizedError, Equatable {
    case cameraUnavailable
    case configurationFailed
    case missingPhotoData

    var errorDescription: String? {
        switch self {
        case .cameraUnavailable:
            return "这台设备无法使用前置摄像头。"
        case .configurationFailed:
            return "FlashShot 暂时无法启动相机。"
        case .missingPhotoData:
            return "相机没有返回可读取的照片，请重拍一次。"
        }
    }
}
