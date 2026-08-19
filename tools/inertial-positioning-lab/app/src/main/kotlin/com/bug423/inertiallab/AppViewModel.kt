package com.bug423.inertiallab

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.core.net.toUri
import com.bug423.inertiallab.data.CapturePreferences
import com.bug423.inertiallab.data.CaptureSettings
import com.bug423.inertiallab.data.CaptureState
import com.bug423.inertiallab.data.SessionStore
import com.bug423.inertiallab.data.SessionSummary
import com.bug423.inertiallab.data.Vec3
import com.bug423.inertiallab.model.BenchmarkEngine
import com.bug423.inertiallab.model.BenchmarkBatchReport
import com.bug423.inertiallab.model.BenchmarkRequest
import com.bug423.inertiallab.model.BenchmarkReport
import com.bug423.inertiallab.model.InstalledModel
import com.bug423.inertiallab.model.LiveBenchmarkEngine
import com.bug423.inertiallab.model.LiveBenchmarkSession
import com.bug423.inertiallab.model.LiveBenchmarkSnapshot
import com.bug423.inertiallab.model.ModelStore
import com.bug423.inertiallab.sensor.CANONICAL_SAMPLE_RATE_HZ
import com.bug423.inertiallab.sensor.SamplingPlan
import com.bug423.inertiallab.sensor.SensorRecorder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

enum class AppTab { OVERVIEW, CAPTURE, MODELS }

data class AppUiState(
    val tab: AppTab = AppTab.OVERVIEW,
    val sessions: List<SessionSummary> = emptyList(),
    val models: List<InstalledModel> = emptyList(),
    val selectedSequenceId: String? = null,
    val selectedModelDirectory: String? = null,
    val selectedSequenceIds: Set<String> = emptySet(),
    val selectedModelDirectories: Set<String> = emptySet(),
    val isBusy: Boolean = false,
    val benchmarkProgress: Float = 0f,
    val benchmarkCompletedRuns: Int = 0,
    val benchmarkTotalRuns: Int = 0,
    val benchmarkCurrentModelDirectory: String? = null,
    val benchmarkCurrentSequenceId: String? = null,
    val report: BenchmarkReport? = null,
    val reports: List<BenchmarkReport> = emptyList(),
    val batchReport: BenchmarkBatchReport? = null,
    val captureSettings: CaptureSettings = CaptureSettings(),
    val liveBenchmark: LiveBenchmarkSnapshot = LiveBenchmarkSnapshot(),
    val notice: String? = null,
)

data class CaptureShellState(
    val isRecording: Boolean = false,
    val elapsedSeconds: Long = 0L,
    val sampleRateHz: Int = CANONICAL_SAMPLE_RATE_HZ,
    val imuAvailable: Boolean = false,
    val orientationAvailable: Boolean = false,
    val referenceAvailable: Boolean = false,
    val referenceStatus: String? = null,
    val message: String? = null,
)

data class ImuTelemetryState(
    val accelerometer: Vec3 = Vec3(),
    val gyroscope: Vec3 = Vec3(),
)

data class CaptureTrajectoryState(
    val points: List<Vec3> = emptyList(),
    val distanceMeters: Float = 0f,
    val elapsedSeconds: Long = 0L,
)

private fun CaptureState.toShellState() = CaptureShellState(
    isRecording = isRecording,
    elapsedSeconds = elapsedSeconds.toLong(),
    sampleRateHz = sampleRateHz,
    imuAvailable = imuAvailable,
    orientationAvailable = orientationAvailable,
    referenceAvailable = referenceAvailable,
    referenceStatus = referenceStatus,
    message = message,
)

class AppViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionStore = SessionStore(application)
    private val modelStore = ModelStore(application)
    private val benchmarkEngine = BenchmarkEngine(application, sessionStore, modelStore)
    private val recorder = SensorRecorder(application, sessionStore)
    private val capturePreferences = CapturePreferences(application)
    private val liveBenchmarkEngine = LiveBenchmarkEngine(modelStore)
    private val initialSamplingPlan = runCatching { recorder.samplingPlan() }.getOrElse {
        SamplingPlan(
            targetRateHz = CANONICAL_SAMPLE_RATE_HZ,
            effectiveRateHz = CANONICAL_SAMPLE_RATE_HZ,
            hardwareMaximumRateHz = CANONICAL_SAMPLE_RATE_HZ,
        )
    }
    private val mutableUi = MutableStateFlow(
        AppUiState(
            captureSettings = capturePreferences.load(
                effectiveSampleRateHz = initialSamplingPlan.effectiveRateHz,
                hardwareMaximumRateHz = initialSamplingPlan.hardwareMaximumRateHz,
            ),
        ),
    )
    private var liveBenchmarkSession: LiveBenchmarkSession? = null
    private var liveFrameJob: Job? = null
    val ui: StateFlow<AppUiState> = mutableUi.asStateFlow()
    val capture: StateFlow<CaptureState> = recorder.state
    val captureShell: StateFlow<CaptureShellState> = capture
        .map(CaptureState::toShellState)
        .distinctUntilChanged()
        .stateIn(viewModelScope, SharingStarted.Eagerly, capture.value.toShellState())
    val imuTelemetry: StateFlow<ImuTelemetryState> = capture
        .map { ImuTelemetryState(it.accelerometer, it.gyroscope) }
        .distinctUntilChanged()
        .stateIn(viewModelScope, SharingStarted.Eagerly, ImuTelemetryState())
    val captureTrajectory: StateFlow<CaptureTrajectoryState> = capture
        .map {
            CaptureTrajectoryState(
                points = it.trajectory,
                distanceMeters = it.trajectoryDistanceMeters,
                elapsedSeconds = it.elapsedSeconds.toLong(),
            )
        }
        .distinctUntilChanged()
        .stateIn(viewModelScope, SharingStarted.Eagerly, CaptureTrajectoryState())

    init {
        refresh()
    }

    fun selectTab(tab: AppTab) {
        mutableUi.value = mutableUi.value.copy(tab = tab)
    }

    fun startCapture(name: String, rate: Int, useArCoreReference: Boolean) {
        if (mutableUi.value.isBusy) return notice("请先等待当前模型或数据任务完成")
        runCatching {
            recorder.start(name, CANONICAL_SAMPLE_RATE_HZ, useArCoreReference)
        }
            .onSuccess {
                mutableUi.value = mutableUi.value.copy(liveBenchmark = LiveBenchmarkSnapshot())
            }
            .onFailure { notice(it.userMessage()) }
    }

    fun stopCapture() {
        val result = runCatching { recorder.stop() }
        stopLiveBenchmark()
        result
            .onSuccess { summary -> summary?.let(::exportToConfiguredDirectory) }
            .onFailure { notice(it.userMessage()) }
        refresh()
    }

    fun startLiveBenchmarkCapture() {
        val state = mutableUi.value
        if (state.isBusy) return notice("当前仍有任务在运行")
        if (capture.value.isRecording) return notice("请先结束当前采集")
        val selected = state.selectedModelDirectories
            .ifEmpty { state.selectedModelDirectory?.let(::setOf).orEmpty() }
        val models = state.models.filter { it.directoryName in selected }
        if (models.isEmpty()) return notice("请先选择至少一个实时测试模型")
        val plan = runCatching { recorder.samplingPlan() }
            .getOrElse { return notice(it.userMessage()) }
        mutableUi.value = state.copy(
            isBusy = true,
            liveBenchmark = LiveBenchmarkSnapshot(
                sampleRateHz = plan.effectiveRateHz,
                message = "正在加载实时模型",
            ),
            notice = null,
        )
        viewModelScope.launch {
            runCatching {
                val session = liveBenchmarkEngine.createSession(models, plan.effectiveRateHz)
                liveBenchmarkSession = session
                liveFrameJob = launch(Dispatchers.Default) {
                    try {
                        recorder.frames.collect { frame ->
                            session.accept(frame)?.let { snapshot ->
                                mutableUi.value = mutableUi.value.copy(liveBenchmark = snapshot)
                            }
                        }
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (error: Throwable) {
                        if (liveBenchmarkSession === session) {
                            mutableUi.value = mutableUi.value.copy(
                                liveBenchmark = session.snapshot().copy(
                                    isActive = false,
                                    message = error.userMessage(),
                                ),
                                notice = "实时测试已停止：${error.userMessage()}",
                            )
                        }
                    }
                }
                recorder.start(
                    name = "live-${System.currentTimeMillis()}",
                    sampleRateHz = CANONICAL_SAMPLE_RATE_HZ,
                    useArCoreReference = true,
                )
                session.snapshot()
            }.onSuccess { snapshot ->
                mutableUi.value = mutableUi.value.copy(
                    tab = AppTab.CAPTURE,
                    isBusy = false,
                    liveBenchmark = snapshot,
                    notice = "实时联测已开始",
                )
            }.onFailure { error ->
                liveFrameJob?.cancel()
                liveFrameJob = null
                liveBenchmarkSession?.close()
                liveBenchmarkSession = null
                mutableUi.value = mutableUi.value.copy(
                    isBusy = false,
                    liveBenchmark = LiveBenchmarkSnapshot(message = error.userMessage()),
                    notice = error.userMessage(),
                )
            }
        }
    }

    fun setUseArCoreGroundTruth(enabled: Boolean) {
        capturePreferences.setUseArCoreGroundTruth(enabled)
        reloadCaptureSettings()
    }

    fun setCaptureSaveDirectory(uri: Uri) {
        capturePreferences.setSaveDirectory(uri)
        reloadCaptureSettings()
        notice("后续采集将自动保存到所选目录")
    }

    fun resetCaptureSaveDirectory() {
        capturePreferences.clearSaveDirectory()
        reloadCaptureSettings()
        notice("已恢复应用内部默认保存位置")
    }

    fun selectSession(sequenceId: String) {
        selectSessions(setOf(sequenceId))
    }

    fun selectModel(directoryName: String) {
        selectModels(setOf(directoryName))
    }

    fun toggleSessionSelection(sequenceId: String) {
        val state = mutableUi.value
        if (state.sessions.none { it.sequenceId == sequenceId }) return
        val selected = state.selectedSequenceIds.toMutableSet().apply {
            if (!add(sequenceId)) remove(sequenceId)
        }
        updateSessionSelection(selected)
    }

    fun toggleModelSelection(directoryName: String) {
        val state = mutableUi.value
        if (state.models.none { it.directoryName == directoryName }) return
        val selected = state.selectedModelDirectories.toMutableSet().apply {
            if (!add(directoryName)) remove(directoryName)
        }
        updateModelSelection(selected)
    }

    fun selectSessions(sequenceIds: Collection<String>) {
        val available = mutableUi.value.sessions.mapTo(hashSetOf(), SessionSummary::sequenceId)
        updateSessionSelection(sequenceIds.filterTo(linkedSetOf()) { it in available })
    }

    fun selectModels(directoryNames: Collection<String>) {
        val available = mutableUi.value.models.mapTo(hashSetOf(), InstalledModel::directoryName)
        updateModelSelection(directoryNames.filterTo(linkedSetOf()) { it in available })
    }

    fun importModel(uri: Uri) = launchBusy("模型导入成功") {
        val model = modelStore.import(uri)
        refresh(selectedModels = setOf(model.directoryName))
    }

    fun importModels(uris: Collection<Uri>) {
        if (uris.isEmpty()) return notice("没有选择模型包")
        launchBusy {
            val result = modelStore.importAll(uris)
            if (result.imported.isNotEmpty()) {
                refresh(selectedModels = result.imported.mapTo(linkedSetOf(), InstalledModel::directoryName))
            }
            when {
                result.failures.isEmpty() -> "已导入 ${result.successfulCount} 个模型"
                result.imported.isEmpty() -> "模型导入失败：${result.failures.first().reason}"
                else -> "已导入 ${result.successfulCount}/${result.requestedCount} 个模型；${result.failures.size} 个失败"
            }
        }
    }

    fun importDataset(uri: Uri) = launchBusy("数据集导入成功") {
        val session = sessionStore.importArchive(uri)
        refresh(selectedSequences = setOf(session.sequenceId))
    }

    fun importDatasets(uris: Collection<Uri>) {
        if (uris.isEmpty()) return notice("没有选择数据包")
        launchBusy {
            val sources = uris.distinctBy(Uri::toString)
            val imported = ArrayList<SessionSummary>(sources.size)
            val failures = ArrayList<String>()
            sources.forEach { uri ->
                try {
                    imported += sessionStore.importArchive(uri)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    failures += (error.message ?: error::class.java.simpleName)
                }
            }
            if (imported.isNotEmpty()) {
                refresh(selectedSequences = imported.mapTo(linkedSetOf(), SessionSummary::sequenceId))
            }
            when {
                failures.isEmpty() -> "已导入 ${imported.size} 个数据序列"
                imported.isEmpty() -> "数据包导入失败：${failures.first()}"
                else -> "已导入 ${imported.size}/${sources.size} 个数据序列；${failures.size} 个失败"
            }
        }
    }

    fun exportSession(sequenceId: String, uri: Uri) = launchBusy("数据集已导出") {
        sessionStore.export(sequenceId, uri)
    }

    fun exportReport(uri: Uri) {
        val report = mutableUi.value.report ?: return notice("当前没有可导出的报告")
        launchBusy("评测报告已导出") { benchmarkEngine.export(report, uri) }
    }

    fun exportBatchReport(uri: Uri) {
        val report = mutableUi.value.batchReport ?: return notice("当前没有可导出的批量报告")
        launchBusy("批量评测报告已导出") { benchmarkEngine.exportBatch(report, uri) }
    }

    fun runBenchmark() {
        val state = mutableUi.value
        if (capture.value.isRecording) return notice("请先结束当前采集或实时联测")
        val selectedModels = state.selectedModelDirectories
            .ifEmpty { state.selectedModelDirectory?.let(::setOf).orEmpty() }
        val selectedSequences = state.selectedSequenceIds
            .ifEmpty { state.selectedSequenceId?.let(::setOf).orEmpty() }
        val models = state.models.filter { it.directoryName in selectedModels }
        val sessions = state.sessions.filter { it.sequenceId in selectedSequences }
        if (models.isEmpty()) return notice("请先选择至少一个模型")
        if (sessions.isEmpty()) return notice("请先选择至少一个数据序列")
        val requests = models.flatMap { model ->
            sessions.map { session ->
                BenchmarkRequest(model, session.sequenceId, session.sampleRateHz)
            }
        }
        if (state.isBusy) return
        mutableUi.value = state.copy(
            isBusy = true,
            benchmarkProgress = 0f,
            benchmarkCompletedRuns = 0,
            benchmarkTotalRuns = requests.size,
            benchmarkCurrentModelDirectory = null,
            benchmarkCurrentSequenceId = null,
            report = null,
            reports = emptyList(),
            batchReport = null,
            notice = null,
        )
        viewModelScope.launch {
            runCatching {
                benchmarkEngine.runBatch(requests) { progress ->
                    mutableUi.value = mutableUi.value.copy(
                        benchmarkProgress = progress.overallFraction,
                        benchmarkCompletedRuns = progress.completedRuns,
                        benchmarkTotalRuns = progress.totalRuns,
                        benchmarkCurrentModelDirectory = progress.currentModelDirectory,
                        benchmarkCurrentSequenceId = progress.currentSequenceId,
                    )
                }
            }.onSuccess { batch ->
                mutableUi.value = mutableUi.value.copy(
                    isBusy = false,
                    benchmarkProgress = 1f,
                    benchmarkCompletedRuns = batch.requested_runs,
                    benchmarkTotalRuns = batch.requested_runs,
                    benchmarkCurrentModelDirectory = null,
                    benchmarkCurrentSequenceId = null,
                    report = batch.reports.lastOrNull(),
                    reports = batch.reports,
                    batchReport = batch,
                    notice = if (batch.failed_runs == 0) {
                        "评测完成：${batch.successful_runs} 项"
                    } else {
                        "评测完成：${batch.successful_runs}/${batch.requested_runs} 项成功，${batch.failed_runs} 项失败"
                    },
                )
            }.onFailure { error ->
                mutableUi.value = mutableUi.value.copy(
                    isBusy = false,
                    benchmarkProgress = 0f,
                    benchmarkCurrentModelDirectory = null,
                    benchmarkCurrentSequenceId = null,
                    notice = error.userMessage(),
                )
            }
        }
    }

    fun runBatchBenchmarks() = runBenchmark()

    fun clearNotice() {
        mutableUi.value = mutableUi.value.copy(notice = null)
    }

    fun showNotice(message: String) {
        notice(message)
    }

    private fun refresh(
        selectedModels: Set<String>? = null,
        selectedSequences: Set<String>? = null,
    ) {
        viewModelScope.launch {
            val sessions = sessionStore.list()
            val models = modelStore.list()
            val current = mutableUi.value
            val validSequenceIds = sessions.mapTo(hashSetOf(), SessionSummary::sequenceId)
            val validModelDirectories = models.mapTo(hashSetOf(), InstalledModel::directoryName)
            val reconciledSequences = (
                selectedSequences
                    ?: current.selectedSequenceIds.takeIf(Set<String>::isNotEmpty)
                    ?: current.selectedSequenceId?.let(::setOf)
                    ?: emptySet()
                ).filterTo(linkedSetOf()) { it in validSequenceIds }
                .ifEmpty { sessions.firstOrNull()?.sequenceId?.let(::setOf).orEmpty() }
            val reconciledModels = (
                selectedModels
                    ?: current.selectedModelDirectories.takeIf(Set<String>::isNotEmpty)
                    ?: current.selectedModelDirectory?.let(::setOf)
                    ?: emptySet()
                ).filterTo(linkedSetOf()) { it in validModelDirectories }
                .ifEmpty { models.firstOrNull()?.directoryName?.let(::setOf).orEmpty() }
            mutableUi.value = current.copy(
                sessions = sessions,
                models = models,
                selectedSequenceId = reconciledSequences.firstOrNull(),
                selectedModelDirectory = reconciledModels.firstOrNull(),
                selectedSequenceIds = reconciledSequences,
                selectedModelDirectories = reconciledModels,
            )
        }
    }

    private fun updateSessionSelection(selected: Set<String>) {
        mutableUi.value = mutableUi.value.copy(
            selectedSequenceId = selected.firstOrNull(),
            selectedSequenceIds = selected,
            report = null,
            reports = emptyList(),
            batchReport = null,
        )
    }

    private fun updateModelSelection(selected: Set<String>) {
        mutableUi.value = mutableUi.value.copy(
            selectedModelDirectory = selected.firstOrNull(),
            selectedModelDirectories = selected,
            report = null,
            reports = emptyList(),
            batchReport = null,
        )
    }

    private fun launchBusy(success: String, block: suspend () -> Unit) {
        launchBusy {
            block()
            success
        }
    }

    private fun launchBusy(block: suspend () -> String) {
        if (mutableUi.value.isBusy) return
        mutableUi.value = mutableUi.value.copy(isBusy = true, notice = null)
        viewModelScope.launch {
            runCatching { block() }
                .onSuccess { success -> mutableUi.value = mutableUi.value.copy(isBusy = false, notice = success) }
                .onFailure { mutableUi.value = mutableUi.value.copy(isBusy = false, notice = it.userMessage()) }
        }
    }

    private fun notice(message: String) {
        mutableUi.value = mutableUi.value.copy(notice = message)
    }

    private fun reloadCaptureSettings() {
        val plan = runCatching { recorder.samplingPlan() }.getOrElse { initialSamplingPlan }
        mutableUi.value = mutableUi.value.copy(
            captureSettings = capturePreferences.load(
                effectiveSampleRateHz = plan.effectiveRateHz,
                hardwareMaximumRateHz = plan.hardwareMaximumRateHz,
            ),
        )
    }

    private fun exportToConfiguredDirectory(summary: SessionSummary) {
        val directory = mutableUi.value.captureSettings.saveDirectoryUri ?: return
        viewModelScope.launch {
            runCatching { sessionStore.exportToDirectory(summary.sequenceId, directory.toUri()) }
                .onSuccess { notice("已保存 ${summary.name}，并复制到自定义目录") }
                .onFailure { error ->
                    notice("采集已安全保存在应用内；复制到自定义目录失败：${error.userMessage()}")
                }
        }
    }

    private fun stopLiveBenchmark() {
        liveFrameJob?.cancel()
        liveFrameJob = null
        val session = liveBenchmarkSession ?: return
        liveBenchmarkSession = null
        val finalSnapshot = session.snapshot(capture.value.elapsedSeconds).copy(
            isActive = false,
            message = "实时联测已结束",
        )
        session.close()
        mutableUi.value = mutableUi.value.copy(liveBenchmark = finalSnapshot)
    }

    override fun onCleared() {
        liveFrameJob?.cancel()
        liveBenchmarkSession?.close()
        recorder.close()
        super.onCleared()
    }
}

private fun Throwable.userMessage(): String = message ?: this::class.java.simpleName
