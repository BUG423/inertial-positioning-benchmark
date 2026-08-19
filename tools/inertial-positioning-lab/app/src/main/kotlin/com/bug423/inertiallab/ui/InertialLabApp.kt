package com.bug423.inertiallab.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.BatteryChargingFull
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.CloudUpload
import androidx.compose.material.icons.rounded.DataObject
import androidx.compose.material.icons.rounded.Download
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.FolderZip
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.LocationOn
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.ModelTraining
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Sensors
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material.icons.rounded.Timer
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalDensity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.bug423.inertiallab.AppTab
import com.bug423.inertiallab.AppUiState
import com.bug423.inertiallab.CaptureShellState
import com.bug423.inertiallab.CaptureTrajectoryState
import com.bug423.inertiallab.ImuTelemetryState
import com.bug423.inertiallab.data.CaptureSettings
import com.bug423.inertiallab.data.SessionSummary
import com.bug423.inertiallab.data.Vec3
import com.bug423.inertiallab.model.BenchmarkReport
import com.bug423.inertiallab.model.InstalledModel
import com.bug423.inertiallab.model.LiveModelResult
import com.bug423.inertiallab.ui.theme.Blue
import com.bug423.inertiallab.ui.theme.Coral
import com.bug423.inertiallab.ui.theme.Cyan
import com.bug423.inertiallab.ui.theme.Mint
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.abs
import kotlinx.coroutines.flow.StateFlow

private const val TARGET_SAMPLE_RATE_HZ = 200
private val TRAJECTORY_RANGES_METERS = floatArrayOf(.5f, 1f, 2f, 5f, 10f, 20f, 50f, 100f)

private enum class BenchmarkMode { LIVE, DATASET }

@Composable
fun InertialLabApp(
    ui: AppUiState,
    capture: CaptureShellState,
    imuTelemetry: StateFlow<ImuTelemetryState>,
    captureTrajectory: StateFlow<CaptureTrajectoryState>,
    onTab: (AppTab) -> Unit,
    onStartCapture: (String, Int, Boolean) -> Unit,
    onStopCapture: () -> Unit,
    onImportModel: () -> Unit,
    onImportDataset: () -> Unit,
    onExportDataset: (String, String) -> Unit,
    onSelectSession: (String) -> Unit,
    onSelectModel: (String) -> Unit,
    onRunBenchmark: () -> Unit,
    onExportReport: () -> Unit,
    onNoticeShown: () -> Unit,
    captureSaveLocationLabel: String = ui.captureSettings.saveLocationLabel,
    onChooseCaptureDirectory: () -> Unit = {},
    onResetCaptureDirectory: () -> Unit = {},
    onUseArCoreGroundTruthChange: (Boolean) -> Unit = {},
    onImportModels: () -> Unit = onImportModel,
    onImportDatasets: () -> Unit = onImportDataset,
    onToggleSessionSelection: (String) -> Unit = onSelectSession,
    onToggleModelSelection: (String) -> Unit = onSelectModel,
    onStartLiveBenchmark: () -> Unit = {},
    onExportBatchReport: () -> Unit = onExportReport,
) {
    val snackbar = remember { SnackbarHostState() }
    var settingsOpen by rememberSaveable { mutableStateOf(false) }
    var sequencePrefix by rememberSaveable { mutableStateOf("walk") }

    BackHandler(enabled = settingsOpen) { settingsOpen = false }

    LaunchedEffect(ui.notice) {
        ui.notice?.let {
            snackbar.showSnackbar(it)
            onNoticeShown()
        }
    }

    val navigate: (AppTab) -> Unit = { tab ->
        settingsOpen = false
        onTab(tab)
    }

    Box(
        Modifier.fillMaxSize().background(
            Brush.linearGradient(
                listOf(
                    MaterialTheme.colorScheme.background,
                    MaterialTheme.colorScheme.primary.copy(alpha = .055f),
                    MaterialTheme.colorScheme.background,
                ),
                start = Offset.Zero,
                end = Offset(1050f, 1750f),
            ),
        ),
    ) {
        Scaffold(
            modifier = Modifier.fillMaxSize(),
            containerColor = Color.Transparent,
            snackbarHost = { SnackbarHost(snackbar) },
            bottomBar = {
                if (!settingsOpen) BottomNavigation(ui.tab, navigate)
            },
        ) { padding ->
            if (settingsOpen) {
                SettingsScreen(
                    padding = padding,
                    sequencePrefix = sequencePrefix,
                    onSequencePrefixChange = { sequencePrefix = it },
                    captureSettings = ui.captureSettings,
                    onUseArCoreReferenceChange = onUseArCoreGroundTruthChange,
                    captureSaveLocationLabel = captureSaveLocationLabel,
                    onChooseCaptureDirectory = onChooseCaptureDirectory,
                    onResetCaptureDirectory = onResetCaptureDirectory,
                    onBack = { settingsOpen = false },
                )
            } else {
                when (ui.tab) {
                    AppTab.OVERVIEW -> OverviewScreen(
                        ui = ui,
                        capture = capture,
                        padding = padding,
                        onTab = navigate,
                        onOpenSettings = { settingsOpen = true },
                        onExport = onExportDataset,
                    )

                    AppTab.CAPTURE -> CaptureScreen(
                        ui = ui,
                        capture = capture,
                        imuTelemetry = imuTelemetry,
                        captureTrajectory = captureTrajectory,
                        padding = padding,
                        sequencePrefix = sequencePrefix,
                        useArCoreReference = ui.captureSettings.useArCoreGroundTruth,
                        onOpenSettings = { settingsOpen = true },
                        onStart = onStartCapture,
                        onStop = onStopCapture,
                    )

                    AppTab.MODELS -> ModelsScreen(
                        ui = ui,
                        captureActive = capture.isRecording,
                        padding = padding,
                        onImportModels = onImportModels,
                        onImportDatasets = onImportDatasets,
                        onToggleSession = onToggleSessionSelection,
                        onToggleModel = onToggleModelSelection,
                        onRun = onRunBenchmark,
                        onStartLive = onStartLiveBenchmark,
                        onOpenCapture = { navigate(AppTab.CAPTURE) },
                        onOpenSettings = { settingsOpen = true },
                        onExportReport = onExportReport,
                        onExportBatchReport = onExportBatchReport,
                    )
                }
            }
        }
    }
}

@Composable
private fun BottomNavigation(selected: AppTab, onTab: (AppTab) -> Unit) {
    val bottom = WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding()
    Row(
        Modifier.fillMaxWidth()
            .padding(horizontal = 22.dp)
            .padding(bottom = bottom + 10.dp, top = 7.dp)
            .clip(RoundedCornerShape(27.dp))
            .background(MaterialTheme.colorScheme.surface.copy(alpha = .88f))
            .padding(5.dp),
        horizontalArrangement = Arrangement.SpaceAround,
    ) {
        NavItem(AppTab.OVERVIEW, selected, Icons.Rounded.Home, "总览", onTab)
        NavItem(AppTab.CAPTURE, selected, Icons.Rounded.Sensors, "采集", onTab)
        NavItem(AppTab.MODELS, selected, Icons.Rounded.ModelTraining, "测试", onTab)
    }
}

@Composable
private fun RowScope.NavItem(
    tab: AppTab,
    selected: AppTab,
    icon: ImageVector,
    label: String,
    onTab: (AppTab) -> Unit,
) {
    val active = tab == selected
    Column(
        Modifier.weight(1f)
            .clip(RoundedCornerShape(21.dp))
            .background(if (active) MaterialTheme.colorScheme.primary.copy(alpha = .12f) else Color.Transparent)
            .clickable { onTab(tab) }
            .padding(vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Icon(
            icon,
            label,
            Modifier.size(20.dp),
            tint = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            label,
            color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = 11.sp,
            fontWeight = if (active) FontWeight.SemiBold else FontWeight.Normal,
        )
    }
}

@Composable
private fun OverviewScreen(
    ui: AppUiState,
    capture: CaptureShellState,
    padding: PaddingValues,
    onTab: (AppTab) -> Unit,
    onOpenSettings: () -> Unit,
    onExport: (String, String) -> Unit,
) {
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 20.dp),
        contentPadding = PaddingValues(
            top = padding.calculateTopPadding() + 20.dp,
            bottom = padding.calculateBottomPadding() + 20.dp,
        ),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            AppHeader("总览") {
                HeaderIcon(Icons.Rounded.Settings, "设置", onOpenSettings, enabled = !capture.isRecording)
            }
        }
        item {
            GlassPanel {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        StatusIcon(capture.isRecording)
                        Spacer(Modifier.width(13.dp))
                        Column(Modifier.weight(1f)) {
                            Text(
                                if (capture.isRecording) "采集进行中" else "设备已就绪",
                                style = MaterialTheme.typography.titleLarge,
                            )
                            Text(
                                if (capture.isRecording) {
                                    "${capture.sampleRateHz} Hz · ${formatClock(capture.elapsedSeconds.toDouble())}"
                                } else {
                                    if (ui.captureSettings.requiresResampling) {
                                        "目标 200 Hz · 本机 ${ui.captureSettings.effectiveSampleRateHz} Hz"
                                    } else {
                                        "默认 200 Hz · " +
                                            if (ui.captureSettings.useArCoreGroundTruth) "VIO 参考已启用" else "仅 IMU"
                                    }
                                },
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        PrimaryHomeAction(
                            title = if (capture.isRecording) "查看采集" else "采集数据",
                            subtitle = "IMU + ARCore",
                            icon = Icons.Rounded.Sensors,
                            color = Blue,
                            modifier = Modifier.weight(1f),
                        ) { onTab(AppTab.CAPTURE) }
                        PrimaryHomeAction(
                            title = "模型测试",
                            subtitle = "实时 / 数据包",
                            icon = Icons.Rounded.ModelTraining,
                            color = Cyan,
                            modifier = Modifier.weight(1f),
                        ) { onTab(AppTab.MODELS) }
                    }
                }
            }
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                MetricTile("序列", ui.sessions.size.toString(), Modifier.weight(1f), Blue)
                MetricTile("模型", ui.models.size.toString(), Modifier.weight(1f), Cyan)
                MetricTile("总时长", shortDuration(ui.sessions.sumOf(SessionSummary::durationSeconds)), Modifier.weight(1f), Mint)
            }
        }
        ui.sessions.firstOrNull()?.let { latest ->
            item { SectionTitle("最近一次", "完整数据管理在测试页") }
            item { SessionRow(latest) { onExport(latest.sequenceId, latest.sequenceId) } }
        }
    }
}

@Composable
private fun CaptureScreen(
    ui: AppUiState,
    capture: CaptureShellState,
    imuTelemetry: StateFlow<ImuTelemetryState>,
    captureTrajectory: StateFlow<CaptureTrajectoryState>,
    padding: PaddingValues,
    sequencePrefix: String,
    useArCoreReference: Boolean,
    onOpenSettings: () -> Unit,
    onStart: (String, Int, Boolean) -> Unit,
    onStop: () -> Unit,
) {
    val live = ui.liveBenchmark
    val liveActive = live.isActive
    val recording = capture.isRecording || liveActive
    val modelResults = if (liveActive) live.modelResults else emptyList()

    Column(
        Modifier.fillMaxSize()
            .padding(horizontal = 20.dp)
            .padding(
                top = padding.calculateTopPadding() + 20.dp,
                bottom = padding.calculateBottomPadding() + 10.dp,
            ),
    ) {
        AppHeader("采集") {
            HeaderIcon(Icons.Rounded.Settings, "设置", onOpenSettings, enabled = !recording)
        }
        Spacer(Modifier.height(14.dp))
        StreamedTrajectoryPanel(
            stream = captureTrajectory,
            imuStream = imuTelemetry,
            modelResults = modelResults,
            active = capture.isRecording || liveActive,
            referenceAvailable = capture.referenceAvailable,
            referenceStatus = capture.referenceStatus,
            liveActive = liveActive,
            modifier = Modifier.weight(1f),
        )
        Spacer(Modifier.height(12.dp))
        CaptureActionBar(
            recording = recording,
            liveActive = liveActive,
            onClick = {
                if (recording) {
                    onStop()
                } else {
                    val safePrefix = sequencePrefix.trim().ifEmpty { "walk" }
                    val suffix = DateTimeFormatter.ofPattern("MMdd-HHmmss").format(LocalDateTime.now())
                    onStart("$safePrefix-$suffix", TARGET_SAMPLE_RATE_HZ, useArCoreReference)
                }
            },
        )
    }
}

@Composable
private fun StreamedTrajectoryPanel(
    stream: StateFlow<CaptureTrajectoryState>,
    imuStream: StateFlow<ImuTelemetryState>,
    modelResults: List<LiveModelResult>,
    active: Boolean,
    referenceAvailable: Boolean,
    referenceStatus: String?,
    liveActive: Boolean,
    modifier: Modifier = Modifier,
) {
    val trajectory by stream.collectAsStateWithLifecycle()
    val telemetry = imuStream.collectAsStateWithLifecycle()
    TrajectoryPanel(
        referencePoints = trajectory.points,
        modelResults = modelResults,
        active = active,
        referenceAvailable = referenceAvailable,
        referenceStatus = referenceStatus,
        distanceMeters = trajectory.distanceMeters,
        elapsedSeconds = trajectory.elapsedSeconds,
        liveActive = liveActive,
        telemetry = telemetry,
        modifier = modifier,
    )
}

@Composable
private fun TrajectoryPanel(
    referencePoints: List<Vec3>,
    modelResults: List<LiveModelResult>,
    active: Boolean,
    referenceAvailable: Boolean,
    referenceStatus: String?,
    distanceMeters: Float,
    elapsedSeconds: Long,
    liveActive: Boolean,
    telemetry: State<ImuTelemetryState>,
    modifier: Modifier = Modifier,
) {
    val trackingLabel = when {
        referenceAvailable -> "定位稳定"
        active && referencePoints.isEmpty() -> "初始化"
        active && referenceStatus != null -> "恢复中"
        active -> "等待定位"
        else -> "未开始"
    }
    GlassPanel(modifier = modifier.fillMaxWidth()) {
        Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(if (liveActive) "实时轨迹" else "运动轨迹", style = MaterialTheme.typography.titleLarge)
                }
                SignalPill(trackingLabel, referenceAvailable)
            }
            TrajectoryMap(
                referencePoints,
                modelResults,
                active,
                telemetry,
                Modifier.weight(1f),
            )
            if (liveActive && modelResults.isNotEmpty()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    modelResults.take(3).forEachIndexed { index, result ->
                        LiveResultLine(result, trajectoryColor(index))
                    }
                    if (modelResults.size > 3) {
                        Text(
                            "另有 ${modelResults.size - 3} 个模型正在运行",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 12.sp,
                        )
                    }
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    CompactMetric("距离", "${distanceMeters.toDouble().pretty()} m", Modifier.weight(1f))
                    CompactMetric("轨迹点", referencePoints.size.toString(), Modifier.weight(1f))
                    CompactMetric("时间", formatClock(elapsedSeconds.toDouble()), Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun TrajectoryMap(
    referencePoints: List<Vec3>,
    modelResults: List<LiveModelResult>,
    active: Boolean,
    telemetry: State<ImuTelemetryState>,
    modifier: Modifier = Modifier,
) {
    val allPoints = buildList {
        addAll(referencePoints)
        modelResults.forEach { addAll(it.predictedTrajectory) }
    }
    val rangeMeters = stableTrajectoryRange(allPoints)
    val canvasBackground = Color(0xFF0B1730)
    val gridColor = Color.White.copy(alpha = .075f)
    val axisColor = Color.White.copy(alpha = .18f)
    val pathColor = if (active) Cyan else Color(0xFF7890B7)

    Box(
        modifier.fillMaxWidth()
            .clip(RoundedCornerShape(22.dp))
            .background(canvasBackground),
    ) {
        Canvas(Modifier.fillMaxSize().padding(16.dp)) {
            repeat(9) { index ->
                val fraction = index / 8f
                val x = size.width * fraction
                val y = size.height * fraction
                val verticalColor = if (index == 4) axisColor else gridColor
                val horizontalColor = if (index == 4) axisColor else gridColor
                drawLine(verticalColor, Offset(x, 0f), Offset(x, size.height), if (index == 4) 1.5f else 1f)
                drawLine(horizontalColor, Offset(0f, y), Offset(size.width, y), if (index == 4) 1.5f else 1f)
            }

            val drawableHalfWidth = size.width / 2f - 18.dp.toPx()
            val drawableHalfHeight = size.height / 2f - 18.dp.toPx()
            fun map(point: Vec3) = Offset(
                x = size.width / 2f + point.x / rangeMeters * drawableHalfWidth,
                y = size.height / 2f - point.y / rangeMeters * drawableHalfHeight,
            )
            fun drawSmoothSeries(points: List<Vec3>, color: Color, width: Float, glow: Boolean) {
                if (points.isEmpty()) return
                val mapped = points.map(::map)
                val path = Path().apply {
                    moveTo(mapped.first().x, mapped.first().y)
                    if (mapped.size == 2) {
                        lineTo(mapped.last().x, mapped.last().y)
                    } else {
                        for (index in 1 until mapped.lastIndex) {
                            val point = mapped[index]
                            val next = mapped[index + 1]
                            quadraticTo(
                                point.x,
                                point.y,
                                (point.x + next.x) / 2f,
                                (point.y + next.y) / 2f,
                            )
                        }
                        if (mapped.size > 1) lineTo(mapped.last().x, mapped.last().y)
                    }
                }
                if (glow) {
                    drawPath(
                        path,
                        color.copy(alpha = .18f),
                        style = Stroke(width = width * 2.8f, cap = StrokeCap.Round),
                    )
                }
                drawPath(path, color, style = Stroke(width = width, cap = StrokeCap.Round))
            }

            drawSmoothSeries(referencePoints, pathColor, 3.5.dp.toPx(), glow = true)
            modelResults.forEachIndexed { index, result ->
                drawSmoothSeries(result.predictedTrajectory, trajectoryColor(index), 2.5.dp.toPx(), glow = false)
            }
            val first = referencePoints.firstOrNull()?.let(::map)
            val current = referencePoints.lastOrNull()?.let(::map)
            if (first != null) {
                drawCircle(Mint.copy(alpha = .20f), 10.dp.toPx(), first)
                drawCircle(canvasBackground, 5.5.dp.toPx(), first)
                drawCircle(Mint, 5.5.dp.toPx(), first, style = Stroke(2.dp.toPx()))
            }
            if (current != null) {
                drawCircle(Coral.copy(alpha = .22f), 13.dp.toPx(), current)
                drawCircle(Coral, 6.dp.toPx(), current)
                drawCircle(Color.White, 2.dp.toPx(), current)
            }
        }

        TrajectoryImuOverlay(
            telemetry = telemetry,
            modifier = Modifier.align(Alignment.TopStart).padding(12.dp),
        )
        Text(
            "±${rangeMeters.toDouble().pretty()} m",
            modifier = Modifier.align(Alignment.BottomStart).padding(12.dp),
            color = Color.White.copy(alpha = .58f),
            fontSize = 10.sp,
        )
        Text(
            "Y ↑   X →",
            Modifier.align(Alignment.BottomEnd).padding(12.dp),
            color = Color.White.copy(alpha = .58f),
            fontSize = 10.sp,
        )
        if (allPoints.isEmpty()) {
            Column(
                Modifier.align(Alignment.Center).padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                Box(
                    Modifier.size(12.dp).clip(CircleShape)
                        .background(if (active) Coral else Color.White.copy(alpha = .38f)),
                )
                Text(
                    if (active) "正在建立 VIO 原点" else "等待开始采集",
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    if (active) "缓慢移动手机，让镜头看到有纹理的环境" else "开始后轨迹将从画布中心生成",
                    color = Color.White.copy(alpha = .58f),
                    fontSize = 11.sp,
                )
            }
        } else if (modelResults.isEmpty()) {
            Row(
                Modifier.align(Alignment.TopEnd).padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                MapLegend("起点", Mint)
                MapLegend("当前", Coral)
            }
        } else {
            Row(
                Modifier.align(Alignment.TopEnd).padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                modelResults.take(2).forEachIndexed { index, result ->
                    MapLegend(result.modelName, trajectoryColor(index))
                }
            }
        }
    }
}

@Composable
private fun MapLegend(label: String, color: Color) {
    Row(
        modifier = Modifier.clip(CircleShape).background(Color.Black.copy(alpha = .28f))
            .padding(horizontal = 8.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Box(Modifier.size(6.dp).background(color, CircleShape))
        Text(label, color = Color.White.copy(alpha = .78f), fontSize = 9.sp, maxLines = 1)
    }
}

@Composable
private fun TrajectoryImuOverlay(
    telemetry: State<ImuTelemetryState>,
    modifier: Modifier = Modifier,
) {
    val textColor = Color.White.toArgb()
    val mutedColor = Color.White.copy(alpha = .55f).toArgb()
    val density = LocalDensity.current
    val labelPaint = remember(textColor, density) {
        android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = textColor
            textSize = with(density) { 9.sp.toPx() }
            typeface = android.graphics.Typeface.create(android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
        }
    }
    val titlePaint = remember(mutedColor, density) {
        android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = mutedColor
            textSize = with(density) { 8.sp.toPx() }
        }
    }
    val valuePaint = remember(textColor, density) {
        android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = textColor
            textSize = with(density) { 9.sp.toPx() }
            textAlign = android.graphics.Paint.Align.CENTER
            typeface = android.graphics.Typeface.create(android.graphics.Typeface.MONOSPACE, android.graphics.Typeface.BOLD)
        }
    }
    Canvas(
        modifier.width(226.dp).height(82.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(Color(0xD91A2844)),
    ) {
        val current = telemetry.value
        val labelWidth = 46.dp.toPx()
        val valueWidth = (size.width - labelWidth) / 3f
        val canvas = drawContext.canvas.nativeCanvas

        canvas.drawText("IMU", 9.dp.toPx(), 15.dp.toPx(), labelPaint)
        canvas.drawText("m/s²  ·  rad/s", 9.dp.toPx(), 27.dp.toPx(), titlePaint)
        listOf("X" to Blue, "Y" to Mint, "Z" to Coral).forEachIndexed { index, (axis, color) ->
            valuePaint.color = color.toArgb()
            canvas.drawText(axis, labelWidth + valueWidth * (index + .5f), 16.dp.toPx(), valuePaint)
        }
        valuePaint.color = textColor

        fun drawRow(baseline: Float, label: String, value: Vec3, decimals: Int) {
            canvas.drawText(label, 9.dp.toPx(), baseline, labelPaint)
            val values = floatArrayOf(value.x, value.y, value.z)
            values.forEachIndexed { index, axisValue ->
                canvas.drawText(
                    "%+.${decimals}f".format(axisValue),
                    labelWidth + valueWidth * (index + .5f),
                    baseline,
                    valuePaint,
                )
            }
        }

        drawRow(49.dp.toPx(), "加速", current.accelerometer, 1)
        drawRow(70.dp.toPx(), "角速", current.gyroscope, 2)
    }
}

private fun stableTrajectoryRange(points: List<Vec3>): Float {
    val extent = points.maxOfOrNull { maxOf(abs(it.x), abs(it.y)) }?.times(1.18f) ?: 0f
    return TRAJECTORY_RANGES_METERS.firstOrNull { it >= extent } ?: extent.coerceAtLeast(100f)
}

@Composable
private fun LiveResultLine(result: LiveModelResult, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        StatusDot(color)
        Text(
            result.modelName,
            Modifier.padding(start = 8.dp).weight(1f),
            style = MaterialTheme.typography.labelLarge,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        Text(
            result.error ?: result.endpointErrorMeters?.let { "误差 ${it.toDouble().pretty()} m" } ?: "等待窗口",
            color = if (result.error == null) MaterialTheme.colorScheme.onSurfaceVariant else Coral,
            fontSize = 12.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        if (result.error == null) {
            Text(" · ${result.latestLatencyMs.pretty()} ms", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 12.sp)
        }
    }
}

@Composable
private fun CaptureActionBar(
    recording: Boolean,
    liveActive: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = .94f),
        shadowElevation = 16.dp,
    ) {
        Button(
            onClick = onClick,
            modifier = Modifier.padding(7.dp).fillMaxWidth().height(58.dp),
            shape = RoundedCornerShape(19.dp),
            colors = ButtonDefaults.buttonColors(containerColor = if (recording) Coral else Blue),
        ) {
            Icon(if (recording) Icons.Rounded.Stop else Icons.Rounded.PlayArrow, null)
            Spacer(Modifier.width(9.dp))
            Text(
                when {
                    liveActive -> "结束实时测试并保存"
                    recording -> "停止并保存"
                    else -> "开始采集"
                },
                style = MaterialTheme.typography.titleMedium,
            )
        }
    }
}

@Composable
private fun SettingsScreen(
    padding: PaddingValues,
    sequencePrefix: String,
    onSequencePrefixChange: (String) -> Unit,
    captureSettings: CaptureSettings,
    onUseArCoreReferenceChange: (Boolean) -> Unit,
    captureSaveLocationLabel: String,
    onChooseCaptureDirectory: () -> Unit,
    onResetCaptureDirectory: () -> Unit,
    onBack: () -> Unit,
) {
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 20.dp),
        contentPadding = PaddingValues(
            top = padding.calculateTopPadding() + 22.dp,
            bottom = WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding() + 24.dp,
        ),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                HeaderIcon(Icons.AutoMirrored.Rounded.ArrowBack, "返回", onBack)
                Spacer(Modifier.width(12.dp))
                AppHeader("采集设置", "常用设置集中在这里")
            }
        }
        item {
            GlassPanel {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    SettingLine(
                        icon = Icons.Rounded.Timer,
                        title = "目标采样率",
                        subtitle = if (captureSettings.requiresResampling) {
                            "本机最高 ${captureSettings.hardwareMaximumRateHz} Hz · 自动降级"
                        } else {
                            "规范固定为 200 Hz"
                        },
                        trailing = {
                            Text(
                                "${captureSettings.effectiveSampleRateHz} Hz",
                                color = if (captureSettings.requiresResampling) Coral else Blue,
                                fontWeight = FontWeight.Bold,
                            )
                        },
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = .12f))
                    Text(
                        "若设备 IMU 无法达到 200 Hz，采集器自动使用硬件可提供的最高频率；导出后建议重采样到 200 Hz。",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }
        item {
            GlassPanel {
                Column(verticalArrangement = Arrangement.spacedBy(13.dp)) {
                    SettingLine(
                        icon = Icons.Rounded.LocationOn,
                        title = "VIO 参考轨迹",
                        subtitle = "记录视觉—惯性里程计（VIO）局部位姿与轨迹",
                        trailing = {
                            Switch(captureSettings.useArCoreGroundTruth, onUseArCoreReferenceChange)
                        },
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = .12f))
                    Text(
                        "中国大陆版 Android 通常不会预装 Google Play Services for AR（包名 com.google.ar.core，" +
                            "部分应用商店显示“Google AR 服务”或“ARCore”）。手机硬件支持不代表运行时已经安装，" +
                            "请先从可信应用商店安装并启用。",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }
        item {
            GlassPanel {
                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    SettingLine(
                        icon = Icons.Rounded.FolderOpen,
                        title = "序列保存位置",
                        subtitle = captureSaveLocationLabel,
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Button(onChooseCaptureDirectory, Modifier.weight(1f), shape = RoundedCornerShape(16.dp)) {
                            Icon(Icons.Rounded.FolderOpen, null)
                            Spacer(Modifier.width(7.dp))
                            Text("选择文件夹")
                        }
                        OutlinedButton(onResetCaptureDirectory, shape = RoundedCornerShape(16.dp)) {
                            Icon(Icons.Rounded.Refresh, null)
                            Spacer(Modifier.width(7.dp))
                            Text("恢复默认")
                        }
                    }
                    Text(
                        "采集时先写入应用内的安全暂存；停止后会自动将 .iplab 复制到所选目录。",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontSize = 12.sp,
                    )
                }
            }
        }
        item {
            GlassPanel {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("序列命名", style = MaterialTheme.typography.titleMedium)
                    OutlinedTextField(
                        value = sequencePrefix,
                        onValueChange = { value ->
                            onSequencePrefixChange(value.filter { it.isLetterOrDigit() || it == '-' || it == '_' }.take(32))
                        },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        label = { Text("自动名称前缀") },
                        supportingText = { Text("示例：${sequencePrefix.ifEmpty { "walk" }}-0816-143025") },
                        shape = RoundedCornerShape(18.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            unfocusedContainerColor = MaterialTheme.colorScheme.surface.copy(alpha = .35f),
                            focusedContainerColor = MaterialTheme.colorScheme.surface.copy(alpha = .56f),
                        ),
                    )
                }
            }
        }
        item {
            Text(
                "采集期间屏幕保持唤醒。相机只由 ARCore 在内部用于 VIO，应用不读取或展示画面；数据包只保存位姿与 IMU。",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(horizontal = 5.dp),
            )
        }
    }
}

@Composable
private fun ModelsScreen(
    ui: AppUiState,
    captureActive: Boolean,
    padding: PaddingValues,
    onImportModels: () -> Unit,
    onImportDatasets: () -> Unit,
    onToggleSession: (String) -> Unit,
    onToggleModel: (String) -> Unit,
    onRun: () -> Unit,
    onStartLive: () -> Unit,
    onOpenCapture: () -> Unit,
    onOpenSettings: () -> Unit,
    onExportReport: () -> Unit,
    onExportBatchReport: () -> Unit,
) {
    var mode by rememberSaveable { mutableStateOf(BenchmarkMode.LIVE) }
    var showModels by rememberSaveable { mutableStateOf(true) }
    var showDatasets by rememberSaveable { mutableStateOf(true) }
    val selectedModels = ui.selectedModelDirectories
    val selectedDatasets = ui.selectedSequenceIds
    val liveActive = ui.liveBenchmark.isActive
    val runCount = selectedModels.size * selectedDatasets.size

    Box(Modifier.fillMaxSize()) {
        LazyColumn(
            Modifier.fillMaxSize().padding(horizontal = 20.dp),
            contentPadding = PaddingValues(
                top = padding.calculateTopPadding() + 20.dp,
                bottom = padding.calculateBottomPadding() + 112.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item {
                AppHeader("测试") {
                    HeaderIcon(
                        Icons.Rounded.Settings,
                        "设置",
                        onOpenSettings,
                        enabled = !captureActive && !liveActive && !ui.isBusy,
                    )
                }
            }
            item {
                ModeSelector(mode) { mode = it }
            }
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(
                        onImportModels,
                        Modifier.weight(1f),
                        enabled = !captureActive && !ui.isBusy,
                        shape = RoundedCornerShape(16.dp),
                    ) {
                        Icon(Icons.Rounded.CloudUpload, null)
                        Spacer(Modifier.width(7.dp))
                        Text("导入模型")
                    }
                    OutlinedButton(
                        onImportDatasets,
                        Modifier.weight(1f),
                        enabled = !captureActive && !ui.isBusy,
                        shape = RoundedCornerShape(16.dp),
                    ) {
                        Icon(Icons.Rounded.FolderZip, null)
                        Spacer(Modifier.width(7.dp))
                        Text("导入数据包")
                    }
                }
            }
            item {
                SelectionHeader(
                    title = "模型",
                    selectedCount = selectedModels.size,
                    totalCount = ui.models.size,
                    enabled = !captureActive && !ui.isBusy,
                    expanded = showModels,
                    onToggleExpanded = { showModels = !showModels },
                    onSelectAll = {
                        ui.models.filter { it.directoryName !in selectedModels }.forEach { onToggleModel(it.directoryName) }
                    },
                    onClear = {
                        ui.models.filter { it.directoryName in selectedModels }.forEach { onToggleModel(it.directoryName) }
                    },
                )
            }
            if (showModels) {
                if (ui.models.isEmpty()) {
                    item { EmptyPanel("还没有模型", "可一次选择多个 .iplmodel 模型包导入。", Icons.Rounded.ModelTraining) }
                } else {
                    items(ui.models, key = InstalledModel::directoryName) { model ->
                        MultiSelectModelRow(
                            model = model,
                            selected = model.directoryName in selectedModels,
                            enabled = !captureActive && !ui.isBusy,
                        ) {
                            onToggleModel(model.directoryName)
                        }
                    }
                }
            }
            if (mode == BenchmarkMode.DATASET) {
                item {
                    SelectionHeader(
                        title = "数据包",
                        selectedCount = selectedDatasets.size,
                        totalCount = ui.sessions.size,
                        enabled = !captureActive && !ui.isBusy,
                        expanded = showDatasets,
                        onToggleExpanded = { showDatasets = !showDatasets },
                        onSelectAll = {
                            ui.sessions.filter { it.sequenceId !in selectedDatasets }.forEach { onToggleSession(it.sequenceId) }
                        },
                        onClear = {
                            ui.sessions.filter { it.sequenceId in selectedDatasets }.forEach { onToggleSession(it.sequenceId) }
                        },
                    )
                }
                if (showDatasets) {
                    if (ui.sessions.isEmpty()) {
                        item { EmptyPanel("还没有数据", "先采集，或批量导入 .iplab 数据包。", Icons.Rounded.DataObject) }
                    } else {
                        items(ui.sessions, key = SessionSummary::sequenceId) { session ->
                            MultiSelectSessionRow(
                                session = session,
                                selected = session.sequenceId in selectedDatasets,
                                enabled = !captureActive && !ui.isBusy,
                            ) {
                                onToggleSession(session.sequenceId)
                            }
                        }
                    }
                }
            } else {
                item {
                    GlassPanel {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Rounded.Sensors, null, tint = Blue)
                            Spacer(Modifier.width(12.dp))
                            Column {
                                Text("实时联测", style = MaterialTheme.typography.titleMedium)
                                Text(
                                    "同步运行所选模型，并与 VIO 参考轨迹叠加。",
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    style = MaterialTheme.typography.bodyMedium,
                                )
                            }
                        }
                    }
                }
            }
            if (ui.isBusy) {
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                        LinearProgressIndicator(
                            progress = { ui.benchmarkProgress },
                            modifier = Modifier.fillMaxWidth().clip(CircleShape),
                        )
                        Text(
                            "已完成 ${ui.benchmarkCompletedRuns}/${ui.benchmarkTotalRuns}",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 12.sp,
                        )
                    }
                }
            }
            ui.batchReport?.let { batch ->
                item {
                    BatchSummaryPanel(
                        successful = batch.successful_runs,
                        requested = batch.requested_runs,
                        failed = batch.failed_runs,
                        onExport = onExportBatchReport,
                    )
                }
            }
            ui.report?.let { report -> item { ReportPanel(report, onExportReport) } }
        }

        ModelActionBar(
            mode = mode,
            liveActive = liveActive,
            captureActive = captureActive,
            selectedModels = selectedModels.size,
            runCount = runCount,
            busy = ui.isBusy,
            modifier = Modifier.align(Alignment.BottomCenter)
                .padding(horizontal = 20.dp)
                .padding(bottom = padding.calculateBottomPadding() + 10.dp),
            onClick = {
                when {
                    captureActive -> onOpenCapture()
                    mode == BenchmarkMode.LIVE -> onStartLive()
                    else -> onRun()
                }
            },
        )
    }
}

@Composable
private fun ModeSelector(selected: BenchmarkMode, onSelected: (BenchmarkMode) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surface.copy(alpha = .64f))
            .padding(5.dp),
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        ModeItem("实时联测", BenchmarkMode.LIVE, selected, Modifier.weight(1f), onSelected)
        ModeItem("数据包测试", BenchmarkMode.DATASET, selected, Modifier.weight(1f), onSelected)
    }
}

@Composable
private fun ModeItem(
    label: String,
    mode: BenchmarkMode,
    selected: BenchmarkMode,
    modifier: Modifier,
    onSelected: (BenchmarkMode) -> Unit,
) {
    val active = mode == selected
    Box(
        modifier.clip(RoundedCornerShape(16.dp))
            .background(if (active) MaterialTheme.colorScheme.primary else Color.Transparent)
            .clickable { onSelected(mode) }
            .padding(vertical = 11.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            label,
            color = if (active) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

@Composable
private fun SelectionHeader(
    title: String,
    selectedCount: Int,
    totalCount: Int,
    enabled: Boolean,
    expanded: Boolean,
    onToggleExpanded: () -> Unit,
    onSelectAll: () -> Unit,
    onClear: () -> Unit,
) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f).clickable(onClick = onToggleExpanded)) {
            Text(title, style = MaterialTheme.typography.titleLarge)
            Text(
                "已选 $selectedCount / $totalCount · ${if (expanded) "收起列表" else "展开列表"}",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 12.sp,
            )
        }
        TextButton(onSelectAll, enabled = enabled && totalCount > 0 && selectedCount < totalCount) { Text("全选") }
        TextButton(onClear, enabled = enabled && selectedCount > 0) { Text("清空") }
    }
}

@Composable
private fun MultiSelectModelRow(
    model: InstalledModel,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    MultiSelectRow(
        title = model.manifest.name,
        subtitle = "v${model.manifest.version} · ${model.manifest.input.sample_rate_hz} Hz · ${formatBytes(model.sizeBytes)}",
        icon = Icons.Rounded.ModelTraining,
        selected = selected,
        enabled = enabled,
        onClick = onClick,
    )
}

@Composable
private fun MultiSelectSessionRow(
    session: SessionSummary,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    MultiSelectRow(
        title = session.name,
        subtitle = "${session.sampleRateHz} Hz · ${formatDuration(session.durationSeconds)} · ${session.samples} samples",
        icon = Icons.Rounded.DataObject,
        selected = selected,
        enabled = enabled,
        onClick = onClick,
    )
}

@Composable
private fun MultiSelectRow(
    title: String,
    subtitle: String,
    icon: ImageVector,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(19.dp))
            .clickable(enabled = enabled, onClick = onClick),
        shape = RoundedCornerShape(19.dp),
        color = if (selected) MaterialTheme.colorScheme.primary.copy(alpha = .11f)
        else MaterialTheme.colorScheme.surface.copy(alpha = .54f),
    ) {
        Row(
            Modifier.padding(horizontal = 14.dp, vertical = 11.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(11.dp),
        ) {
            Icon(icon, null, Modifier.size(22.dp), tint = if (selected) Blue else MaterialTheme.colorScheme.onSurfaceVariant)
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(subtitle, color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 12.sp, maxLines = 1)
            }
            Checkbox(selected, enabled = enabled, onCheckedChange = { onClick() })
        }
    }
}

@Composable
private fun ModelActionBar(
    mode: BenchmarkMode,
    liveActive: Boolean,
    captureActive: Boolean,
    selectedModels: Int,
    runCount: Int,
    busy: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val enabled = when {
        captureActive -> true
        busy -> false
        mode == BenchmarkMode.LIVE -> selectedModels > 0
        else -> runCount > 0
    }
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = .94f),
        shadowElevation = 16.dp,
    ) {
        Button(
            onClick = onClick,
            enabled = enabled,
            modifier = Modifier.padding(7.dp).fillMaxWidth().height(58.dp),
            shape = RoundedCornerShape(19.dp),
        ) {
            if (busy) CircularProgressIndicator(Modifier.size(21.dp), color = Color.White, strokeWidth = 2.dp)
            else Icon(if (liveActive) Icons.Rounded.LocationOn else Icons.Rounded.PlayArrow, null)
            Spacer(Modifier.width(9.dp))
            Text(
                when {
                    liveActive -> "查看实时轨迹"
                    captureActive -> "查看当前采集"
                    busy -> "正在测试"
                    mode == BenchmarkMode.LIVE -> "开始实时对比 · $selectedModels 个模型"
                    else -> "开始批量测试 · $runCount 项"
                },
                style = MaterialTheme.typography.titleMedium,
            )
        }
    }
}

@Composable
private fun BatchSummaryPanel(
    successful: Int,
    requested: Int,
    failed: Int,
    onExport: () -> Unit,
) {
    GlassPanel {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.CheckCircle, null, tint = if (failed == 0) Mint else Coral)
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("批量测试完成", style = MaterialTheme.typography.titleMedium)
                Text("成功 $successful / $requested · 失败 $failed", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            IconButton(onExport) { Icon(Icons.Rounded.Download, "导出批量报告", tint = Blue) }
        }
    }
}

@Composable
private fun ReportPanel(report: BenchmarkReport, onExport: () -> Unit) {
    GlassPanel {
        Column(verticalArrangement = Arrangement.spacedBy(13.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.CheckCircle, null, tint = Mint)
                Spacer(Modifier.width(9.dp))
                Text("最近结果", style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.weight(1f))
                IconButton(onExport) { Icon(Icons.Rounded.Download, "导出", tint = Blue) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                MetricTile("平均延迟", "${report.latency.mean_ms.pretty()} ms", Modifier.weight(1f), Blue)
                MetricTile("P95", "${report.latency.p95_ms.pretty()} ms", Modifier.weight(1f), Cyan)
            }
            report.accuracy?.let { accuracy ->
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    MetricTile("速度 RMSE", "${accuracy.velocity_rmse_mps.pretty()} m/s", Modifier.weight(1f), Mint)
                    MetricTile("ATE RMSE", "${accuracy.ate_rmse_m.pretty()} m", Modifier.weight(1f), Coral)
                }
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = .12f))
            ResourceLine(Icons.Rounded.Memory, "峰值 PSS", "${report.resources.pss_peak_mb.pretty()} MB")
            ResourceLine(Icons.Rounded.BatteryChargingFull, "能量", report.resources.energy_used_mwh?.let { "${it.pretty()} mWh" } ?: "设备不支持")
            ResourceLine(Icons.Rounded.Timer, "耗时", "${report.resources.wall_time_seconds.pretty()} s")
        }
    }
}

@Composable
private fun ResourceLine(icon: ImageVector, label: String, value: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(label, Modifier.padding(start = 9.dp).weight(1f), color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun AppHeader(title: String, subtitle: String? = null, action: (@Composable () -> Unit)? = null) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.headlineMedium)
            subtitle?.let {
                Text(it, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
            }
        }
        action?.invoke()
    }
}

@Composable
private fun HeaderIcon(icon: ImageVector, label: String, onClick: () -> Unit, enabled: Boolean = true) {
    Surface(shape = CircleShape, color = MaterialTheme.colorScheme.surface.copy(alpha = .68f)) {
        IconButton(onClick, enabled = enabled) { Icon(icon, label) }
    }
}

@Composable
private fun PrimaryHomeAction(
    title: String,
    subtitle: String,
    icon: ImageVector,
    color: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    Column(
        modifier.clip(RoundedCornerShape(19.dp))
            .background(color.copy(alpha = .10f))
            .clickable(onClick = onClick)
            .padding(15.dp),
        verticalArrangement = Arrangement.spacedBy(9.dp),
    ) {
        Icon(icon, null, tint = color)
        Text(title, style = MaterialTheme.typography.titleMedium)
        Text(subtitle, color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 12.sp)
    }
}

@Composable
private fun StatusIcon(recording: Boolean) {
    Surface(
        Modifier.size(48.dp),
        shape = RoundedCornerShape(16.dp),
        color = if (recording) Coral.copy(alpha = .12f) else Mint.copy(alpha = .14f),
    ) {
        Icon(
            if (recording) Icons.Rounded.Sensors else Icons.Rounded.Check,
            null,
            Modifier.padding(13.dp),
            tint = if (recording) Coral else Mint,
        )
    }
}

@Composable
private fun SettingLine(
    icon: ImageVector,
    title: String,
    subtitle: String,
    trailing: (@Composable () -> Unit)? = null,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Surface(shape = RoundedCornerShape(14.dp), color = Blue.copy(alpha = .10f)) {
            Icon(icon, null, Modifier.padding(10.dp).size(21.dp), tint = Blue)
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(
                subtitle,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 12.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        trailing?.invoke()
    }
}

@Composable
private fun SessionRow(session: SessionSummary, onExport: () -> Unit) {
    Surface(
        Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = .60f),
    ) {
        Row(Modifier.padding(13.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.DataObject, null, tint = Blue)
            Spacer(Modifier.width(11.dp))
            Column(Modifier.weight(1f)) {
                Text(session.name, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(
                    "${formatDuration(session.durationSeconds)} · ${session.sampleRateHz} Hz · ${formatBytes(session.sizeBytes)}",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 12.sp,
                )
            }
            IconButton(onExport) { Icon(Icons.Rounded.Download, "导出", tint = Blue) }
        }
    }
}

@Composable
private fun EmptyPanel(title: String, description: String, icon: ImageVector) {
    GlassPanel {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, null, Modifier.size(30.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.width(13.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleMedium)
                Text(description, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

@Composable
private fun CompactMetric(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, style = MaterialTheme.typography.titleMedium)
        Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 11.sp)
    }
}

@Composable
private fun StatusDot(color: Color) {
    Box(Modifier.size(8.dp).background(color, CircleShape))
}

private fun trajectoryColor(index: Int): Color = when (index % 4) {
    0 -> Cyan
    1 -> Mint
    2 -> Coral
    else -> Color(0xFF9A78FF)
}

private fun formatClock(seconds: Double): String {
    val total = seconds.toLong().coerceAtLeast(0)
    return "%02d:%02d.%01d".format(total / 60, total % 60, ((seconds - total) * 10).toInt())
}

private fun formatDuration(seconds: Double): String {
    val total = seconds.toLong()
    return if (total < 60) "${total}s" else "${total / 60}m ${total % 60}s"
}

private fun shortDuration(seconds: Double): String {
    val minutes = (seconds / 60).toLong()
    return if (minutes < 60) "${minutes}m" else "${minutes / 60}h"
}

private fun formatBytes(bytes: Long): String = when {
    bytes >= 1024 * 1024 -> "%.1f MB".format(bytes / 1024.0 / 1024.0)
    bytes >= 1024 -> "%.1f KB".format(bytes / 1024.0)
    else -> "$bytes B"
}

private fun Double.pretty(): String = when {
    this >= 100 -> "%.0f".format(this)
    this >= 10 -> "%.1f".format(this)
    else -> "%.2f".format(this)
}
