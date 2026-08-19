package com.bug423.inertiallab.model

import com.bug423.inertiallab.data.QuaternionWxyz
import com.bug423.inertiallab.data.SensorFrame
import com.bug423.inertiallab.data.Vec3
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.util.ArrayDeque
import kotlin.math.pow
import kotlin.math.sqrt

data class LiveModelResult(
    val modelDirectory: String,
    val modelName: String,
    val predictedPosition: Vec3 = Vec3(),
    val groundTruthPosition: Vec3? = null,
    val latestVelocity: Vec3 = Vec3(),
    val endpointErrorMeters: Float? = null,
    val velocityErrorMps: Float? = null,
    val latestLatencyMs: Double = 0.0,
    val meanLatencyMs: Double = 0.0,
    val inferenceCount: Long = 0,
    val predictedTrajectory: List<Vec3> = emptyList(),
    val error: String? = null,
)

data class LiveBenchmarkSnapshot(
    val isActive: Boolean = false,
    val sampleRateHz: Int = 200,
    val elapsedSeconds: Double = 0.0,
    val groundTruthTrajectory: List<Vec3> = emptyList(),
    val modelResults: List<LiveModelResult> = emptyList(),
    val message: String? = null,
)

/**
 * Executes one or more imported models against the same frames that are being persisted by
 * [com.bug423.inertiallab.sensor.SensorRecorder]. Inference runs off the sensor callback thread;
 * every prediction is integrated and compared with the current ARCore-relative trajectory.
 */
class LiveBenchmarkEngine(private val modelStore: ModelStore) {
    suspend fun createSession(
        selectedModels: List<InstalledModel>,
        sampleRateHz: Int,
    ): LiveBenchmarkSession = withContext(Dispatchers.Default) {
        require(selectedModels.isNotEmpty()) { "请至少选择一个实时测试模型" }
        val incompatible = selectedModels.filter { it.manifest.input.sample_rate_hz != sampleRateHz }
        require(incompatible.isEmpty()) {
            val names = incompatible.joinToString { it.manifest.name }
            "实时采集为 $sampleRateHz Hz，以下模型采样率不匹配：$names"
        }
        val runners = mutableListOf<LiveModelRunner>()
        try {
            selectedModels.forEach { model -> runners += LiveModelRunner(model, modelStore) }
            LiveBenchmarkSession(sampleRateHz, runners)
        } catch (error: Throwable) {
            runners.forEach(LiveModelRunner::close)
            throw error
        }
    }
}

class LiveBenchmarkSession internal constructor(
    private val sampleRateHz: Int,
    private val runners: List<LiveModelRunner>,
) : Closeable {
    private val groundTruthTrajectory = ArrayDeque<Vec3>(MAX_TRAJECTORY_POINTS)
    private var groundTruthOrigin: Vec3? = null
    private var lastGroundTruth: Vec3? = null
    private var lastUiTimestampSeconds = Double.NEGATIVE_INFINITY
    private var closed = false

    /** Returns a UI snapshot at most ten times per second, or null when no redraw is needed. */
    @Synchronized
    fun accept(frame: SensorFrame): LiveBenchmarkSnapshot? {
        check(!closed) { "实时测试已经结束" }
        if (frame.validPosition) appendGroundTruth(frame.position)
        runners.forEach { runner ->
            try {
                runner.accept(frame)
            } catch (error: Exception) {
                runner.fail(error.message ?: error::class.java.simpleName)
            }
        }
        if (frame.timestampSeconds - lastUiTimestampSeconds < UI_UPDATE_PERIOD_SECONDS) return null
        lastUiTimestampSeconds = frame.timestampSeconds
        return snapshot(frame.timestampSeconds)
    }

    @Synchronized
    fun snapshot(elapsedSeconds: Double = lastUiTimestampSeconds.coerceAtLeast(0.0)) =
        LiveBenchmarkSnapshot(
            isActive = !closed,
            sampleRateHz = sampleRateHz,
            elapsedSeconds = elapsedSeconds,
            groundTruthTrajectory = groundTruthTrajectory.toList(),
            modelResults = runners.map(LiveModelRunner::result),
            message = runners.mapNotNull(LiveModelRunner::failureReason)
                .takeIf { it.isNotEmpty() }
                ?.joinToString(prefix = "部分模型已停止：", separator = "；"),
        )

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runners.forEach(LiveModelRunner::close)
    }

    private fun appendGroundTruth(point: Vec3) {
        val origin = groundTruthOrigin ?: point.also { groundTruthOrigin = it }
        val relativePoint = point - origin
        val previous = lastGroundTruth
        if (previous != null && previous.distance(relativePoint) < MIN_TRAJECTORY_STEP_METERS) return
        if (groundTruthTrajectory.size == MAX_TRAJECTORY_POINTS) groundTruthTrajectory.removeFirst()
        groundTruthTrajectory.addLast(relativePoint)
        lastGroundTruth = relativePoint
    }

    companion object {
        private const val MAX_TRAJECTORY_POINTS = 1_500
        private const val MIN_TRAJECTORY_STEP_METERS = 0.02f
        private const val UI_UPDATE_PERIOD_SECONDS = 0.1
    }
}

internal class LiveModelRunner(
    private val model: InstalledModel,
    modelStore: ModelStore,
) : Closeable {
    private val manifest = model.manifest
    private val windowSize = manifest.input.shape[1]
    private val frames = ArrayDeque<SensorFrame>(windowSize)
    private val interpreter: Interpreter
    private val input = Array(1) { Array(windowSize) { FloatArray(6) } }
    private val output = Array(1) { FloatArray(manifest.output.dimensions) }
    private val predictedTrajectory = ArrayDeque<Vec3>(MAX_TRAJECTORY_POINTS)
    private var framesSinceInference = 0
    private var lastFrameTimestamp: Double? = null
    private var prediction = Vec3()
    private var latestVelocity = Vec3()
    private var groundOrigin: Vec3? = null
    private var predictionAtGroundOrigin = Vec3()
    private var lastPredictionTimestamp: Double? = null
    private var latestGroundRelative: Vec3? = null
    private var latestEndpointError: Float? = null
    private var latestVelocityError: Float? = null
    private var latestLatencyMs = 0.0
    private var latencyTotalMs = 0.0
    private var inferenceCount = 0L
    private var closed = false
    private var failedReason: String? = null

    init {
        val mapped = modelStore.modelFile(model).inputStream().channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
        interpreter = Interpreter(
            mapped,
            Interpreter.Options().setNumThreads(manifest.benchmark.threads),
        )
    }

    fun accept(frame: SensorFrame) {
        if (failedReason != null) return
        check(!closed) { "模型 ${manifest.name} 的实时测试已经结束" }
        lastFrameTimestamp?.let { previous ->
            val maximumGap = MAX_FRAME_GAP_PERIODS / manifest.input.sample_rate_hz
            if (frame.timestampSeconds - previous > maximumGap) {
                frames.clear()
                framesSinceInference = 0
            }
        }
        lastFrameTimestamp = frame.timestampSeconds
        if (frames.size == windowSize) frames.removeFirst()
        frames.addLast(frame)
        framesSinceInference += 1
        if (frames.size < windowSize || framesSinceInference < manifest.benchmark.stride) return
        framesSinceInference = 0
        if (frames.any { !it.validImu }) return
        if (manifest.output.coordinate_frame == "body" && !frame.validOrientation) return

        fillInput()
        val started = System.nanoTime()
        interpreter.run(input, output)
        latestLatencyMs = (System.nanoTime() - started) / 1_000_000.0
        latencyTotalMs += latestLatencyMs
        inferenceCount += 1

        val raw = Vec3(
            output[0].getOrElse(0) { 0f },
            output[0].getOrElse(1) { 0f },
            output[0].getOrElse(2) { 0f },
        )
        require(raw.isFinite()) { "模型 ${manifest.name} 实时输出 NaN 或 Inf" }
        latestVelocity = if (manifest.output.coordinate_frame == "body") {
            frame.orientation.rotate(raw)
        } else raw

        lastPredictionTimestamp?.let { previousTimestamp ->
            val deltaSeconds = (frame.timestampSeconds - previousTimestamp).coerceIn(0.0, MAX_INTEGRATION_STEP_SECONDS)
            prediction += latestVelocity * deltaSeconds.toFloat()
        }
        lastPredictionTimestamp = frame.timestampSeconds
        appendPrediction(prediction)

        if (frame.validPosition) {
            if (groundOrigin == null) {
                groundOrigin = frame.position
                predictionAtGroundOrigin = prediction
                predictedTrajectory.clear()
                predictedTrajectory.addLast(prediction)
            }
            val origin = checkNotNull(groundOrigin)
            val groundRelative = frame.position - origin
            val predictionRelative = prediction - predictionAtGroundOrigin
            latestGroundRelative = groundRelative
            latestEndpointError = predictionRelative.distance(groundRelative)
            latestVelocityError = latestVelocity.distance(frame.velocity)
        }
    }

    fun result() = LiveModelResult(
        modelDirectory = model.directoryName,
        modelName = manifest.name,
        predictedPosition = prediction - predictionAtGroundOrigin,
        groundTruthPosition = latestGroundRelative,
        latestVelocity = latestVelocity,
        endpointErrorMeters = latestEndpointError,
        velocityErrorMps = latestVelocityError,
        latestLatencyMs = latestLatencyMs,
        meanLatencyMs = if (inferenceCount == 0L) 0.0 else latencyTotalMs / inferenceCount,
        inferenceCount = inferenceCount,
        predictedTrajectory = predictedTrajectory.map { it - predictionAtGroundOrigin },
        error = failedReason,
    )

    fun failureReason(): String? = failedReason?.let { "${manifest.name}：$it" }

    fun fail(reason: String) {
        if (failedReason != null) return
        failedReason = reason
        close()
    }

    override fun close() {
        if (closed) return
        closed = true
        interpreter.close()
    }

    private fun fillInput() {
        frames.forEachIndexed { index, frame ->
            val values = floatArrayOf(
                frame.gyroscope.x,
                frame.gyroscope.y,
                frame.gyroscope.z,
                frame.accelerometer.x,
                frame.accelerometer.y,
                frame.accelerometer.z,
            )
            for (channel in 0..5) {
                input[0][index][channel] =
                    (values[channel] - manifest.input.mean[channel]) / manifest.input.std[channel]
            }
        }
    }

    private fun appendPrediction(point: Vec3) {
        val previous = predictedTrajectory.peekLast()
        if (previous != null && previous.distance(point) < MIN_TRAJECTORY_STEP_METERS) return
        if (predictedTrajectory.size == MAX_TRAJECTORY_POINTS) predictedTrajectory.removeFirst()
        predictedTrajectory.addLast(point)
    }

    companion object {
        private const val MAX_TRAJECTORY_POINTS = 1_500
        private const val MIN_TRAJECTORY_STEP_METERS = 0.02f
        private const val MAX_INTEGRATION_STEP_SECONDS = 1.0
        private const val MAX_FRAME_GAP_PERIODS = 5.0
    }
}

private operator fun Vec3.plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
private operator fun Vec3.minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
private operator fun Vec3.times(scale: Float) = Vec3(x * scale, y * scale, z * scale)
private fun Vec3.isFinite() = x.isFinite() && y.isFinite() && z.isFinite()
private fun Vec3.distance(other: Vec3): Float = sqrt(
    (x - other.x).pow(2) + (y - other.y).pow(2) + (z - other.z).pow(2),
)

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
