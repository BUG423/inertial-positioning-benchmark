package com.bug423.inertiallab

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.bug423.inertiallab.ui.InertialLabApp
import com.bug423.inertiallab.ui.theme.InertialLabTheme
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Session
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException

class MainActivity : ComponentActivity() {
    private val viewModel: AppViewModel by viewModels()
    private var pendingArCoreCapture: CaptureRequest? = null
    private var arCoreInstallRequested = false
    private var arCoreCheckInFlight = false
    private val mainHandler = Handler(Looper.getMainLooper())
    private var arCoreAvailabilityTimeout: Runnable? = null

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            prepareArCoreCapture()
        } else {
            failPendingArCoreCapture("启用 VIO 参考轨迹需要相机权限")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            InertialLabTheme {
                val ui by viewModel.ui.collectAsStateWithLifecycle()
                val capture by viewModel.captureShell.collectAsStateWithLifecycle()
                var pendingExport by remember { mutableStateOf<String?>(null) }

                val importModels = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
                    viewModel.importModels(uris)
                }
                val importDatasets = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
                    viewModel.importDatasets(uris)
                }
                val chooseCaptureDirectory = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
                    uri?.let(::persistCaptureDirectory)
                }
                val exportDataset = rememberLauncherForActivityResult(
                    ActivityResultContracts.CreateDocument("application/zip"),
                ) { uri ->
                    val sequence = pendingExport
                    if (uri != null && sequence != null) viewModel.exportSession(sequence, uri)
                    pendingExport = null
                }
                val exportReport = rememberLauncherForActivityResult(
                    ActivityResultContracts.CreateDocument("application/json"),
                ) { uri -> uri?.let(viewModel::exportReport) }
                val exportBatchReport = rememberLauncherForActivityResult(
                    ActivityResultContracts.CreateDocument("application/json"),
                ) { uri -> uri?.let(viewModel::exportBatchReport) }

                DisposableEffect(capture.isRecording) {
                    if (capture.isRecording) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    onDispose { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
                }

                InertialLabApp(
                    ui = ui,
                    capture = capture,
                    imuTelemetry = viewModel.imuTelemetry,
                    captureTrajectory = viewModel.captureTrajectory,
                    onTab = viewModel::selectTab,
                    onStartCapture = ::requestCapture,
                    onStopCapture = viewModel::stopCapture,
                    onImportModel = { importModels.launch(PACKAGE_MIME_TYPES) },
                    onImportDataset = { importDatasets.launch(PACKAGE_MIME_TYPES) },
                    onExportDataset = { sequenceId, suggestedName ->
                        pendingExport = sequenceId
                        exportDataset.launch("$suggestedName.iplab")
                    },
                    onSelectSession = viewModel::selectSession,
                    onSelectModel = viewModel::selectModel,
                    onRunBenchmark = viewModel::runBenchmark,
                    onExportReport = {
                        val report = ui.report
                        val name = if (report == null) "benchmark-report.json"
                        else "${report.model_id}_${report.sequence_id}_report.json"
                        exportReport.launch(name)
                    },
                    onNoticeShown = viewModel::clearNotice,
                    captureSaveLocationLabel = ui.captureSettings.saveLocationLabel,
                    onChooseCaptureDirectory = { chooseCaptureDirectory.launch(null) },
                    onResetCaptureDirectory = viewModel::resetCaptureSaveDirectory,
                    onUseArCoreGroundTruthChange = viewModel::setUseArCoreGroundTruth,
                    onImportModels = { importModels.launch(PACKAGE_MIME_TYPES) },
                    onImportDatasets = { importDatasets.launch(PACKAGE_MIME_TYPES) },
                    onToggleSessionSelection = viewModel::toggleSessionSelection,
                    onToggleModelSelection = viewModel::toggleModelSelection,
                    onStartLiveBenchmark = ::requestLiveBenchmark,
                    onExportBatchReport = {
                        exportBatchReport.launch("benchmark-batch-${System.currentTimeMillis()}.json")
                    },
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (pendingArCoreCapture != null && hasCameraPermission() && arCoreInstallRequested) {
            prepareArCoreCapture()
        }
    }

    override fun onStop() {
        if (viewModel.capture.value.isRecording) viewModel.stopCapture()
        super.onStop()
    }

    private fun requestCapture(name: String, rate: Int, useArCoreReference: Boolean) {
        if (!useArCoreReference) {
            viewModel.startCapture(name, rate, false)
            return
        }
        viewModel.showNotice("正在准备相机与 ARCore…")
        Log.d(TAG, "capture requested: name=$name rate=$rate")
        pendingArCoreCapture = CaptureRequest(name, rate, CaptureMode.RECORD)
        requestCameraAndPrepareArCore()
    }

    private fun requestLiveBenchmark() {
        viewModel.showNotice("正在准备相机与 ARCore…")
        Log.d(TAG, "live benchmark requested")
        pendingArCoreCapture = CaptureRequest("live", 200, CaptureMode.LIVE_BENCHMARK)
        requestCameraAndPrepareArCore()
    }

    private fun requestCameraAndPrepareArCore() {
        if (!hasCameraPermission()) {
            cameraPermission.launch(Manifest.permission.CAMERA)
        } else {
            prepareArCoreCapture()
        }
    }

    private fun prepareArCoreCapture() {
        if (pendingArCoreCapture == null || arCoreCheckInFlight) return
        val immediateAvailability = runCatching {
            ArCoreApk.getInstance().checkAvailability(this)
        }.getOrElse { error ->
            Log.w(TAG, "ARCore availability lookup failed; probing local Session", error)
            probeLocalArCoreAndStart("兼容性查询异常")
            return
        }
        if (immediateAvailability != ArCoreApk.Availability.UNKNOWN_CHECKING) {
            handleArCoreAvailability(immediateAvailability)
            return
        }
        arCoreCheckInFlight = true
        val timeout = Runnable {
            if (!arCoreCheckInFlight || pendingArCoreCapture == null) return@Runnable
            arCoreCheckInFlight = false
            arCoreAvailabilityTimeout = null
            Log.w(TAG, "ARCore availability query timed out; probing local Session")
            viewModel.showNotice("ARCore 兼容性查询超时，正在进行本机运行时验证…")
            probeLocalArCoreAndStart("兼容性查询超时")
        }
        arCoreAvailabilityTimeout = timeout
        mainHandler.postDelayed(timeout, ARCORE_CHECK_TIMEOUT_MS)
        ArCoreApk.getInstance().checkAvailabilityAsync(this) { availability ->
            runOnUiThread {
                // The timeout path already falls back to a real local Session. Ignore a late
                // network callback so that the same pending capture cannot be started twice.
                if (!arCoreCheckInFlight || pendingArCoreCapture == null) return@runOnUiThread
                arCoreAvailabilityTimeout?.let(mainHandler::removeCallbacks)
                arCoreAvailabilityTimeout = null
                arCoreCheckInFlight = false
                Log.d(TAG, "ARCore availability result: ${availability.name}")
                handleArCoreAvailability(availability)
            }
        }
    }

    private fun handleArCoreAvailability(availability: ArCoreApk.Availability) {
        Log.d(TAG, "ARCore availability=$availability")
        when (arCoreSetupAction(availability)) {
            ArCoreSetupAction.REQUEST_INSTALL -> requestArCoreInstallAndStart()
            ArCoreSetupAction.PROBE_LOCAL_SESSION -> probeLocalArCoreAndStart(availability.name)
        }
    }

    private fun requestArCoreInstallAndStart() {
        try {
            Log.d(TAG, "requestInstall(forceRequest=${!arCoreInstallRequested})")
            when (ArCoreApk.getInstance().requestInstall(this, !arCoreInstallRequested)) {
                ArCoreApk.InstallStatus.INSTALLED -> {
                    Log.d(TAG, "ARCore runtime installed")
                    startPendingArCoreCapture()
                }
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> arCoreInstallRequested = true
            }
        } catch (_: UnavailableArcoreNotInstalledException) {
            failPendingArCoreCapture(
                "未安装 Google Play Services for AR（com.google.ar.core），或安装请求已被拒绝。" +
                    "中国大陆版手机请先从可信应用商店安装并启用“Google AR 服务/ARCore”",
            )
        } catch (_: UnavailableUserDeclinedInstallationException) {
            failPendingArCoreCapture("已取消 Google Play Services for AR 的安装或更新")
        } catch (_: UnavailableApkTooOldException) {
            failPendingArCoreCapture("请先更新 Google Play Services for AR")
        } catch (_: UnavailableSdkTooOldException) {
            failPendingArCoreCapture("当前应用的 ARCore SDK 版本过旧，请更新应用")
        } catch (_: UnavailableDeviceNotCompatibleException) {
            probeLocalArCoreAndStart("安装流程报告设备不兼容")
        } catch (error: Throwable) {
            failPendingArCoreCapture("ARCore 启动检查失败：${error.message ?: error.javaClass.simpleName}")
        }
    }

    /**
     * Availability can depend on remote compatibility data and is occasionally unreliable on
     * offline or non-Play Android builds. A real local Session construction is the authoritative
     * fallback: if it succeeds, the installed runtime can serve this app regardless of the lookup.
     */
    private fun probeLocalArCoreAndStart(lookupResult: String) {
        if (pendingArCoreCapture == null) return
        try {
            val probe = Session(this)
            runCatching { probe.close() }
            Log.w(TAG, "Local ARCore Session probe succeeded after: $lookupResult")
            viewModel.showNotice("联网兼容性结果不可靠，本机 ARCore 验证通过，正在启动…")
            startPendingArCoreCapture()
        } catch (_: UnavailableArcoreNotInstalledException) {
            failPendingArCoreCapture(
                "手机硬件可能支持 ARCore，但未检测到可用的 Google Play Services for AR。" +
                    "请从可信应用商店安装或启用 com.google.ar.core（Google AR 服务/ARCore）后重试" +
                    "（查询结果：$lookupResult）",
            )
        } catch (_: UnavailableApkTooOldException) {
            failPendingArCoreCapture("本机 Google Play Services for AR 版本过旧，请更新后重试")
        } catch (_: UnavailableSdkTooOldException) {
            failPendingArCoreCapture("当前应用的 ARCore SDK 版本过旧，请更新应用")
        } catch (_: UnavailableDeviceNotCompatibleException) {
            failPendingArCoreCapture(
                "联网查询为 $lookupResult，且本机 ARCore Session 也返回设备不兼容。" +
                    "请确认安装的是 Google Play Services for AR，而不是厂商自有 AR 引擎",
            )
        } catch (error: SecurityException) {
            failPendingArCoreCapture("ARCore 本机验证缺少相机权限：${error.message ?: "SecurityException"}")
        } catch (error: Throwable) {
            failPendingArCoreCapture(
                "ARCore 本机 Session 初始化失败（查询结果：$lookupResult）：" +
                    (error.message ?: error.javaClass.simpleName),
            )
        }
    }

    private fun startPendingArCoreCapture() {
        val request = pendingArCoreCapture ?: return
        arCoreAvailabilityTimeout?.let(mainHandler::removeCallbacks)
        arCoreAvailabilityTimeout = null
        pendingArCoreCapture = null
        arCoreInstallRequested = false
        when (request.mode) {
            CaptureMode.RECORD -> viewModel.startCapture(request.name, request.rate, true)
            CaptureMode.LIVE_BENCHMARK -> viewModel.startLiveBenchmarkCapture()
        }
    }

    private fun failPendingArCoreCapture(message: String) {
        arCoreAvailabilityTimeout?.let(mainHandler::removeCallbacks)
        arCoreAvailabilityTimeout = null
        pendingArCoreCapture = null
        arCoreInstallRequested = false
        viewModel.showNotice(message)
    }

    private fun hasCameraPermission(): Boolean = ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.CAMERA,
    ) == PackageManager.PERMISSION_GRANTED

    private fun persistCaptureDirectory(uri: android.net.Uri) {
        runCatching {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
        }.onFailure { error ->
            viewModel.showNotice("无法保留该目录权限：${error.message ?: error.javaClass.simpleName}")
            return
        }
        viewModel.setCaptureSaveDirectory(uri)
    }

    private data class CaptureRequest(
        val name: String,
        val rate: Int,
        val mode: CaptureMode,
    )

    private enum class CaptureMode { RECORD, LIVE_BENCHMARK }

    private companion object {
        const val TAG = "InertialLab.ARCore"
        const val ARCORE_CHECK_TIMEOUT_MS = 2_500L
        val PACKAGE_MIME_TYPES = arrayOf("application/zip", "application/octet-stream")
    }
}

internal enum class ArCoreSetupAction {
    REQUEST_INSTALL,
    PROBE_LOCAL_SESSION,
}

internal fun arCoreSetupAction(availability: ArCoreApk.Availability): ArCoreSetupAction = when {
    availability.isSupported -> ArCoreSetupAction.REQUEST_INSTALL
    else -> ArCoreSetupAction.PROBE_LOCAL_SESSION
}
