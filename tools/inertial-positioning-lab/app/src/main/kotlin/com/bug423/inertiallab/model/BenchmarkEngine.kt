package com.bug423.inertiallab.model

import android.content.Context
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import com.bug423.inertiallab.data.QuaternionWxyz
import com.bug423.inertiallab.data.ARCORE_POSITION_SOURCE
import com.bug423.inertiallab.data.SensorFrame
import com.bug423.inertiallab.data.SessionStore
import com.bug423.inertiallab.data.Vec3
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.tensorflow.lite.Interpreter
import java.io.File
import java.time.Instant
import kotlin.coroutines.coroutineContext
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

@Serializable
data class AccuracyMetrics(
    val evaluated_windows: Int,
    val velocity_rmse_mps: Double,
    val ate_rmse_m: Double,
    val endpoint_drift_m: Double,
    val relative_drift_percent: Double,
)

@Serializable
data class LatencyMetrics(
    val mean_ms: Double,
    val p50_ms: Double,
    val p95_ms: Double,
    val p99_ms: Double,
    val throughput_hz: Double,
)

@Serializable
data class ResourceMetrics(
    val wall_time_seconds: Double,
    val process_cpu_percent: Double,
    val pss_before_mb: Double,
    val pss_peak_mb: Double,
    val java_heap_delta_mb: Double,
    val energy_used_mwh: Double? = null,
    val charge_used_mah: Double? = null,
    val average_power_mw: Double? = null,
    val current_before_ma: Double? = null,
    val current_after_ma: Double? = null,
    val thermal_status_before: Int? = null,
    val thermal_status_peak: Int? = null,
)

@Serializable
data class BenchmarkReport(
    val report_schema: Int = 1,
    val created_at_utc: String,
    val model_id: String,
    val model_name: String,
    val model_version: String,
    val sequence_id: String,
    val world_frame: String,
    val position_source: String,
    val orientation_source: String,
    val device: String,
    val android_version: String,
    val sample_rate_hz: Int,
    val window_size: Int,
    val stride: Int,
    val warmup_runs: Int,
    val measured_windows: Int,
    val accuracy: AccuracyMetrics? = null,
    val latency: LatencyMetrics,
    val resources: ResourceMetrics,
    val notes: List<String>,
)

data class BenchmarkRequest(
    val model: InstalledModel,
    val sequenceId: String,
    val sampleRateHz: Int,
)

data class BenchmarkBatchProgress(
    val completedRuns: Int,
    val totalRuns: Int,
    val currentModelDirectory: String?,
    val currentSequenceId: String?,
    val currentRunProgress: Float,
) {
    val overallFraction: Float
        get() = if (totalRuns <= 0) 0f else {
            ((completedRuns + currentRunProgress.coerceIn(0f, 1f)) / totalRuns)
                .coerceIn(0f, 1f)
        }
}

@Serializable
data class BenchmarkFailure(
    val model_id: String,
    val model_name: String,
    val model_version: String,
    val model_directory: String,
    val sequence_id: String,
    val reason: String,
)

@Serializable
data class BenchmarkBatchReport(
    val batch_schema: Int = 1,
    val created_at_utc: String,
    val requested_runs: Int,
    val successful_runs: Int,
    val failed_runs: Int,
    val reports: List<BenchmarkReport>,
    val failures: List<BenchmarkFailure>,
)

class BenchmarkEngine(
    private val context: Context,
    private val sessions: SessionStore,
    private val models: ModelStore,
) {
    private val json = Json { prettyPrint = true }
    private val reports = File(context.filesDir, "reports").apply { mkdirs() }
    private val battery = context.getSystemService(BatteryManager::class.java)
    private val power = context.getSystemService(PowerManager::class.java)

    suspend fun run(
        model: InstalledModel,
        sequenceId: String,
        sampleRateHz: Int,
        onProgress: (Float) -> Unit = {},
    ): BenchmarkReport = withContext(Dispatchers.Default) {
        val manifest = model.manifest
        require(sampleRateHz == manifest.input.sample_rate_hz) {
            "数据为 $sampleRateHz Hz，模型要求 ${manifest.input.sample_rate_hz} Hz；请按模型频率采集或在训练侧重采样"
        }
        val sequenceMetadata = sessions.metadata(sequenceId)
        val frames = sessions.frames(sequenceId)
        val window = manifest.input.shape[1]
        require(frames.size >= window) { "序列至少需要 $window 个样本" }
        val starts = (0..frames.size - window step manifest.benchmark.stride)
            .take(manifest.benchmark.max_windows)
        require(starts.isNotEmpty()) { "没有可评测窗口" }

        val mapped = models.modelFile(model).inputStream().channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
        val options = Interpreter.Options().setNumThreads(manifest.benchmark.threads)
        Interpreter(mapped, options).use { interpreter ->
            val input = Array(1) { Array(window) { FloatArray(6) } }
            val output = Array(1) { FloatArray(manifest.output.dimensions) }
            fillInput(input[0], frames, starts.first(), manifest)
            repeat(manifest.benchmark.warmup_runs) { interpreter.run(input, output) }

            val before = snapshot()
            val wallStart = System.nanoTime()
            val cpuStartMs = Process.getElapsedCpuTime()
            var peakPssKb = before.pssKb
            var peakThermal = before.thermal ?: 0
            val latencies = DoubleArray(starts.size)
            val predicted = ArrayList<Prediction>(starts.size)

            starts.forEachIndexed { index, start ->
                coroutineContext.ensureActive()
                fillInput(input[0], frames, start, manifest)
                val inferenceStart = System.nanoTime()
                interpreter.run(input, output)
                latencies[index] = (System.nanoTime() - inferenceStart) / 1_000_000.0
                val end = start + window - 1
                val raw = Vec3(
                    output[0].getOrElse(0) { 0f },
                    output[0].getOrElse(1) { 0f },
                    output[0].getOrElse(2) { 0f },
                )
                require(raw.x.isFinite() && raw.y.isFinite() && raw.z.isFinite()) {
                    "模型在第 ${index + 1} 个窗口输出 NaN 或 Inf"
                }
                val worldVelocity = if (manifest.output.coordinate_frame == "body") {
                    frames[end].orientation.rotate(raw)
                } else raw
                predicted += Prediction(start, end, worldVelocity)
                if (index % 25 == 0 || index == starts.lastIndex) {
                    peakPssKb = max(peakPssKb, Debug.getPss())
                    currentThermal()?.let { peakThermal = max(peakThermal, it) }
                    onProgress((index + 1f) / starts.size)
                }
            }
            val cpuEndMs = Process.getElapsedCpuTime()
            val wallEnd = System.nanoTime()
            val after = snapshot()
            val wallSeconds = (wallEnd - wallStart) / 1_000_000_000.0
            val accuracy = calculateAccuracy(
                frames = frames,
                predictions = predicted,
                dimensions = manifest.output.dimensions,
                requireOrientation = manifest.output.coordinate_frame == "body",
            )
            val energyNwh = positiveDelta(before.energyNwh, after.energyNwh)
            val chargeUah = positiveDelta(before.chargeUah, after.chargeUah)
            val latency = calculateLatency(latencies)
            val resources = ResourceMetrics(
                wall_time_seconds = wallSeconds,
                process_cpu_percent = (cpuEndMs - cpuStartMs) / (wallSeconds * 10.0),
                pss_before_mb = before.pssKb / 1024.0,
                pss_peak_mb = max(peakPssKb, after.pssKb) / 1024.0,
                java_heap_delta_mb = (after.heapBytes - before.heapBytes) / (1024.0 * 1024.0),
                energy_used_mwh = energyNwh?.div(1_000_000.0),
                charge_used_mah = chargeUah?.div(1_000.0),
                average_power_mw = energyNwh?.let { it / 1_000_000.0 / (wallSeconds / 3600.0) },
                current_before_ma = before.currentUa?.div(1_000.0),
                current_after_ma = after.currentUa?.div(1_000.0),
                thermal_status_before = before.thermal,
                thermal_status_peak = peakThermal.takeIf { Build.VERSION.SDK_INT >= 29 },
            )
            val notes = buildList {
                add("延迟为 TFLite 单次同步推理墙钟时间，含张量调用、不含窗口装载。")
                add("CPU 为评测期间整个应用进程 CPU 时间/墙钟时间，多线程时可超过 100%。")
                if (energyNwh == null) add("设备未提供 BatteryManager 能量计数器，能耗无法直接测量。")
                if (accuracy == null) add("序列缺少有效位置参考，未计算精度指标。")
                else if (sequenceMetadata.position_source == ARCORE_POSITION_SOURCE) {
                    add("参考轨迹沿用原版 IMUNet 的 ARCore VIO；它融合相机与 IMU，不是独立的 Vicon/RTK 测量。")
                } else {
                    add("位置来自采集清单声明的参考源，其精度取决于采集系统。")
                }
            }
            val report = BenchmarkReport(
                created_at_utc = Instant.now().toString(),
                model_id = manifest.id,
                model_name = manifest.name,
                model_version = manifest.version,
                sequence_id = sequenceId,
                world_frame = sequenceMetadata.world_frame,
                position_source = sequenceMetadata.position_source,
                orientation_source = sequenceMetadata.orientation_source,
                device = listOf(Build.MANUFACTURER, Build.MODEL).joinToString(" "),
                android_version = Build.VERSION.RELEASE,
                sample_rate_hz = sampleRateHz,
                window_size = window,
                stride = manifest.benchmark.stride,
                warmup_runs = manifest.benchmark.warmup_runs,
                measured_windows = starts.size,
                accuracy = accuracy,
                latency = latency,
                resources = resources,
                notes = notes,
            )
            File(reports, "${manifest.id}_${System.currentTimeMillis()}.json")
                .writeText(json.encodeToString(report))
            report
        }
    }

    /**
     * Runs a deterministic model-by-sequence queue. Runs are deliberately sequential so latency,
     * CPU and memory measurements from one model do not contaminate another. A bad package/data
     * pairing is retained as a failure and does not abort the rest of the queue.
     */
    suspend fun runBatch(
        requests: Collection<BenchmarkRequest>,
        onProgress: (BenchmarkBatchProgress) -> Unit = {},
    ): BenchmarkBatchReport = withContext(Dispatchers.Default) {
        val queue = requests.distinctBy { it.model.directoryName to it.sequenceId }
        require(queue.isNotEmpty()) { "批量评测至少需要一个模型和一个数据序列" }
        val completedReports = ArrayList<BenchmarkReport>(queue.size)
        val failures = ArrayList<BenchmarkFailure>()
        queue.forEachIndexed { index, request ->
            coroutineContext.ensureActive()
            val baseProgress = BenchmarkBatchProgress(
                completedRuns = index,
                totalRuns = queue.size,
                currentModelDirectory = request.model.directoryName,
                currentSequenceId = request.sequenceId,
                currentRunProgress = 0f,
            )
            onProgress(baseProgress)
            try {
                completedReports += run(
                    model = request.model,
                    sequenceId = request.sequenceId,
                    sampleRateHz = request.sampleRateHz,
                ) { currentProgress ->
                    onProgress(baseProgress.copy(currentRunProgress = currentProgress))
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                val manifest = request.model.manifest
                failures += BenchmarkFailure(
                    model_id = manifest.id,
                    model_name = manifest.name,
                    model_version = manifest.version,
                    model_directory = request.model.directoryName,
                    sequence_id = request.sequenceId,
                    reason = error.message ?: error::class.java.simpleName,
                )
            }
            onProgress(
                BenchmarkBatchProgress(
                    completedRuns = index + 1,
                    totalRuns = queue.size,
                    currentModelDirectory = null,
                    currentSequenceId = null,
                    currentRunProgress = 0f,
                ),
            )
        }
        val batch = BenchmarkBatchReport(
            created_at_utc = Instant.now().toString(),
            requested_runs = queue.size,
            successful_runs = completedReports.size,
            failed_runs = failures.size,
            reports = completedReports,
            failures = failures,
        )
        File(reports, "batch_${System.currentTimeMillis()}.json")
            .writeText(json.encodeToString(batch))
        batch
    }

    suspend fun export(report: BenchmarkReport, destination: Uri) = withContext(Dispatchers.IO) {
        context.contentResolver.openOutputStream(destination, "w")?.bufferedWriter()?.use { writer ->
            writer.write(json.encodeToString(report))
        } ?: error("无法打开报告导出位置")
    }

    suspend fun exportBatch(report: BenchmarkBatchReport, destination: Uri) = withContext(Dispatchers.IO) {
        context.contentResolver.openOutputStream(destination, "w")?.bufferedWriter()?.use { writer ->
            writer.write(json.encodeToString(report))
        } ?: error("无法打开批量报告导出位置")
    }

    private fun fillInput(
        destination: Array<FloatArray>,
        frames: List<SensorFrame>,
        start: Int,
        manifest: ModelManifest,
    ) {
        for (offset in destination.indices) {
            val frame = frames[start + offset]
            val values = floatArrayOf(
                frame.gyroscope.x, frame.gyroscope.y, frame.gyroscope.z,
                frame.accelerometer.x, frame.accelerometer.y, frame.accelerometer.z,
            )
            for (channel in 0..5) {
                destination[offset][channel] =
                    (values[channel] - manifest.input.mean[channel]) / manifest.input.std[channel]
            }
        }
    }

    private fun calculateLatency(values: DoubleArray): LatencyMetrics {
        val sorted = values.sorted()
        val mean = values.average()
        fun percentile(p: Double): Double {
            if (sorted.size == 1) return sorted.first()
            val index = p * (sorted.size - 1)
            val lower = index.toInt()
            val upper = min(lower + 1, sorted.lastIndex)
            return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower)
        }
        return LatencyMetrics(mean, percentile(.5), percentile(.95), percentile(.99), 1000.0 / mean)
    }

    private fun calculateAccuracy(
        frames: List<SensorFrame>,
        predictions: List<Prediction>,
        dimensions: Int,
        requireOrientation: Boolean,
    ): AccuracyMetrics? {
        val invalidImuPrefix = IntArray(frames.size + 1)
        frames.forEachIndexed { index, frame ->
            invalidImuPrefix[index + 1] = invalidImuPrefix[index] + if (frame.validImu) 0 else 1
        }
        val valid = predictions.filter {
            frames[it.start].validPosition && frames[it.end].validPosition &&
                invalidImuPrefix[it.end + 1] == invalidImuPrefix[it.start] &&
                (!requireOrientation || frames[it.end].validOrientation)
        }
        if (valid.size < 2) return null
        var velocitySquared = 0.0
        var ateSquared = 0.0
        var evaluated = 0
        var predictedPosition = Vec3()
        val origin = frames[valid.first().end].position
        var previousTime = frames[valid.first().end].timestampSeconds
        var lastGround = origin
        var pathLength = 0.0
        var endpointError = 0.0
        for (prediction in valid) {
            val startFrame = frames[prediction.start]
            val endFrame = frames[prediction.end]
            val duration = endFrame.timestampSeconds - startFrame.timestampSeconds
            if (duration <= 0.0) continue
            val target = ((endFrame.position - startFrame.position) / duration.toFloat()).dimensions(dimensions)
            velocitySquared += prediction.velocity.distanceSquared(target)
            val deltaTime = max(0.0, endFrame.timestampSeconds - previousTime).toFloat()
            predictedPosition += prediction.velocity * deltaTime
            val groundRelative = (endFrame.position - origin).dimensions(dimensions)
            endpointError = sqrt(predictedPosition.distanceSquared(groundRelative))
            ateSquared += endpointError.pow(2)
            pathLength += sqrt(endFrame.position.dimensions(dimensions).distanceSquared(lastGround.dimensions(dimensions)))
            lastGround = endFrame.position
            previousTime = endFrame.timestampSeconds
            evaluated += 1
        }
        if (evaluated < 2) return null
        return AccuracyMetrics(
            evaluated_windows = evaluated,
            velocity_rmse_mps = sqrt(velocitySquared / (evaluated * dimensions.toDouble())),
            ate_rmse_m = sqrt(ateSquared / evaluated),
            endpoint_drift_m = endpointError,
            relative_drift_percent = if (pathLength > 0.01) endpointError / pathLength * 100.0 else 0.0,
        )
    }

    private data class Prediction(val start: Int, val end: Int, val velocity: Vec3)
    private data class Snapshot(
        val energyNwh: Long?,
        val chargeUah: Long?,
        val currentUa: Long?,
        val pssKb: Long,
        val heapBytes: Long,
        val thermal: Int?,
    )

    private fun snapshot(): Snapshot {
        val runtime = Runtime.getRuntime()
        return Snapshot(
            energyNwh = property(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER),
            chargeUah = property(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            currentUa = property(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),
            pssKb = Debug.getPss(),
            heapBytes = runtime.totalMemory() - runtime.freeMemory(),
            thermal = currentThermal(),
        )
    }

    private fun property(id: Int): Long? = battery.getLongProperty(id).takeUnless { it == Long.MIN_VALUE }
    private fun currentThermal(): Int? = if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else null
    private fun positiveDelta(before: Long?, after: Long?): Long? =
        if (before != null && after != null && before >= after) before - after else null
}

private operator fun Vec3.plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
private operator fun Vec3.minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
private operator fun Vec3.times(scale: Float) = Vec3(x * scale, y * scale, z * scale)
private operator fun Vec3.div(scale: Float) = Vec3(x / scale, y / scale, z / scale)
private fun Vec3.distanceSquared(other: Vec3): Double =
    (x - other.x).toDouble().pow(2) + (y - other.y).toDouble().pow(2) + (z - other.z).toDouble().pow(2)
private fun Vec3.dimensions(count: Int) = if (count == 2) Vec3(x, y, 0f) else this

private fun QuaternionWxyz.rotate(vector: Vec3): Vec3 {
    val ix = w * vector.x + y * vector.z - z * vector.y
    val iy = w * vector.y + z * vector.x - x * vector.z
    val iz = w * vector.z + x * vector.y - y * vector.x
    val iw = -x * vector.x - y * vector.y - z * vector.z
    return Vec3(
        ix * w + iw * -x + iy * -z - iz * -y,
        iy * w + iw * -y + iz * -x - ix * -z,
        iz * w + iw * -z + ix * -y - iy * -x,
    )
}
