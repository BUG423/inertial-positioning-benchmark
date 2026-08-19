package com.bug423.inertiallab.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import com.bug423.inertiallab.data.CaptureState
import com.bug423.inertiallab.data.QuaternionWxyz
import com.bug423.inertiallab.data.SensorFrame
import com.bug423.inertiallab.data.SessionStore
import com.bug423.inertiallab.data.SessionSummary
import com.bug423.inertiallab.data.Vec3
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.channels.BufferOverflow
import kotlin.math.hypot
import java.util.ArrayDeque

class SensorRecorder(
    context: Context,
    private val sessionStore: SessionStore,
) : SensorEventListener {
    private val appContext = context.applicationContext
    private val sensorManager = appContext.getSystemService(SensorManager::class.java)
    private val worker = HandlerThread("InertialCapture").apply { start() }
    private val handler = Handler(worker.looper)
    private val arCoreTracker = ArCoreReferenceTracker(appContext)
    private val trajectoryStabilizer = TrajectoryStabilizer()
    private val mutableState = MutableStateFlow(CaptureState())
    val state: StateFlow<CaptureState> = mutableState.asStateFlow()
    private val mutableFrames = MutableSharedFlow<SensorFrame>(
        extraBufferCapacity = LIVE_FRAME_BUFFER_SIZE,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val frames: SharedFlow<SensorFrame> = mutableFrames.asSharedFlow()

    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val rotation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private var latestAcceleration = Vec3()
    private var latestGyroscope = Vec3()
    private var latestRotationVector = QuaternionWxyz()
    private var latestReference: ArCoreReferenceSample? = null
    private var accelerationValid = false
    private var rotationVectorValid = false
    private var useArCoreReference = false
    private var active: SessionStore.ActiveSession? = null
    private var samplePeriodNs = 10_000_000L
    private var firstTimestampNs = 0L
    private var lastWrittenTimestampNs = 0L
    private var lastUiTimestampNs = 0L
    private var sampleCount = 0L
    private val trajectory = ArrayDeque<Vec3>(MAX_TRAJECTORY_POINTS)
    private var trajectoryDistanceMeters = 0f
    private var lastTrajectoryPoint: Vec3? = null
    private var trajectoryDirty = false

    fun samplingPlan(targetRateHz: Int = CANONICAL_SAMPLE_RATE_HZ): SamplingPlan {
        val accelerationSensor = requireNotNull(accelerometer) { "设备缺少加速度计" }
        val gyroscopeSensor = requireNotNull(gyroscope) { "设备缺少陀螺仪" }
        return createSamplingPlan(
            accelerometerMinDelayUs = accelerationSensor.minDelay,
            gyroscopeMinDelayUs = gyroscopeSensor.minDelay,
            targetRateHz = targetRateHz,
        )
    }

    @Synchronized
    fun start(name: String, sampleRateHz: Int, useArCoreReference: Boolean) {
        check(active == null) { "采集已经开始" }
        val accelerationSensor = requireNotNull(accelerometer) { "设备缺少加速度计" }
        val gyroscopeSensor = requireNotNull(gyroscope) { "设备缺少陀螺仪" }
        val plan = samplingPlan(sampleRateHz)
        samplePeriodNs = 1_000_000_000L / plan.effectiveRateHz
        firstTimestampNs = 0L
        lastWrittenTimestampNs = 0L
        lastUiTimestampNs = 0L
        sampleCount = 0L
        latestAcceleration = Vec3()
        latestGyroscope = Vec3()
        latestRotationVector = QuaternionWxyz()
        latestReference = null
        accelerationValid = false
        rotationVectorValid = false
        this.useArCoreReference = useArCoreReference
        trajectory.clear()
        trajectoryDistanceMeters = 0f
        lastTrajectoryPoint = null
        trajectoryDirty = false
        trajectoryStabilizer.reset()
        val session = sessionStore.begin(name, plan.effectiveRateHz, useArCoreReference)
        active = session
        val periodUs = plan.requestedPeriodUs
        try {
            val accelerationRegistered = sensorManager.registerListener(this, accelerationSensor, periodUs, handler)
            val gyroscopeRegistered = sensorManager.registerListener(this, gyroscopeSensor, periodUs, handler)
            require(accelerationRegistered && gyroscopeRegistered) { "无法启动 IMU 传感器监听" }
            val rotationRegistered = !useArCoreReference && rotation?.let {
                sensorManager.registerListener(this, it, periodUs, handler)
            } == true
            mutableState.value = CaptureState(
                isRecording = true,
                sampleRateHz = plan.effectiveRateHz,
                targetSampleRateHz = plan.targetRateHz,
                hardwareMaximumRateHz = plan.hardwareMaximumRateHz,
                requiresResampling = plan.requiresResampling,
                imuAvailable = true,
                orientationAvailable = !useArCoreReference && rotationRegistered,
                referenceStatus = if (useArCoreReference) "正在初始化 ARCore" else "未启用",
                message = when {
                    plan.requiresResampling ->
                        "设备 IMU 最高共同采样率为 ${plan.effectiveRateHz} Hz；训练前请重采样到 200 Hz"
                    useArCoreReference -> "请保持相机朝向有纹理且光照充足的环境"
                    else -> null
                },
                activeSequenceId = session.initialMetadata.sequence_id,
            )
            if (useArCoreReference) {
                arCoreTracker.start(::onArCoreSample, ::onArCoreStatus)
            }
        } catch (error: Throwable) {
            sensorManager.unregisterListener(this)
            arCoreTracker.stop()
            active = null
            session.abort()
            throw error
        }
    }

    fun stop(): SessionSummary? {
        val session = synchronized(this) {
            val current = active ?: return null
            sensorManager.unregisterListener(this)
            active = null
            latestReference = null
            current
        }
        arCoreTracker.stop()
        val finished = runCatching { session.finish() }
        mutableState.value = mutableState.value.copy(
            isRecording = false,
            referenceAvailable = false,
            referenceStatus = null,
            activeSequenceId = null,
            message = finished.fold(
                onSuccess = { "已保存 ${it.samples} 个样本" },
                onFailure = { "保存失败：${it.message ?: it::class.java.simpleName}" },
            ),
        )
        return finished.getOrThrow()
    }

    fun close() {
        runCatching { stop() }
        arCoreTracker.close()
        worker.quitSafely()
    }

    @Synchronized
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAcceleration = Vec3(event.values[0], event.values[1], event.values[2])
                accelerationValid = event.accuracy != SensorManager.SENSOR_STATUS_UNRELIABLE
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                val quaternion = FloatArray(4)
                SensorManager.getQuaternionFromVector(quaternion, event.values)
                latestRotationVector = QuaternionWxyz(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
                rotationVectorValid = event.accuracy != SensorManager.SENSOR_STATUS_UNRELIABLE
            }
            Sensor.TYPE_GYROSCOPE -> {
                latestGyroscope = Vec3(event.values[0], event.values[1], event.values[2])
                maybeWrite(event.timestamp, event.accuracy != SensorManager.SENSOR_STATUS_UNRELIABLE)
            }
        }
    }

    @Synchronized
    private fun maybeWrite(timestampNs: Long, gyroValid: Boolean) {
        val session = active ?: return
        if (firstTimestampNs == 0L) firstTimestampNs = timestampNs
        if (lastWrittenTimestampNs != 0L && timestampNs - lastWrittenTimestampNs < samplePeriodNs * 8 / 10) return
        val reference = latestReference
        val referenceAgeNs = reference?.let { timestampNs - it.observedAtNs }
        val validArCorePose = useArCoreReference && reference != null && referenceAgeNs != null &&
            referenceAgeNs in -MAX_REFERENCE_FUTURE_SKEW_NS..MAX_REFERENCE_AGE_NS
        val validPosition = validArCorePose && reference?.velocityValid == true
        val orientation = if (useArCoreReference) reference?.orientation ?: QuaternionWxyz() else latestRotationVector
        val validOrientation = if (useArCoreReference) validArCorePose else rotationVectorValid
        val time = (timestampNs - firstTimestampNs) / 1_000_000_000.0
        val frame = SensorFrame(
            timestampSeconds = time,
            gyroscope = latestGyroscope,
            accelerometer = latestAcceleration,
            orientation = orientation,
            position = reference?.position ?: Vec3(),
            velocity = reference?.velocity ?: Vec3(),
            latitude = null,
            longitude = null,
            altitude = null,
            horizontalAccuracy = null,
            validImu = gyroValid && accelerationValid,
            validOrientation = validOrientation,
            validPosition = validPosition,
        )
        session.append(frame)
        mutableFrames.tryEmit(frame)
        lastWrittenTimestampNs = timestampNs
        sampleCount += 1

        if (timestampNs - lastUiTimestampNs >= UI_UPDATE_PERIOD_NS) {
            lastUiTimestampNs = timestampNs
            val currentState = mutableState.value
            val trajectorySnapshot = if (trajectoryDirty) trajectory.toList() else currentState.trajectory
            trajectoryDirty = false
            mutableState.value = currentState.copy(
                elapsedSeconds = time,
                samples = sampleCount,
                imuAvailable = accelerationValid && gyroValid,
                orientationAvailable = validOrientation,
                referenceAvailable = frame.validPosition,
                accelerometer = latestAcceleration,
                gyroscope = latestGyroscope,
                trajectory = trajectorySnapshot,
                trajectoryDistanceMeters = trajectoryDistanceMeters,
            )
        }
    }

    @Synchronized
    private fun onArCoreSample(sample: ArCoreReferenceSample) {
        if (active == null || !useArCoreReference) return
        latestReference = sample
        val stabilized = trajectoryStabilizer.update(
            timestampNs = sample.observedAtNs,
            rawPosition = sample.position,
            accelerometer = latestAcceleration,
            gyroscope = latestGyroscope,
        )
        val point = stabilized.position
        val previous = lastTrajectoryPoint
        val displacement = previous?.let { hypot(point.x - it.x, point.y - it.y) }
        if (previous == null || displacement != null && displacement >= MIN_TRAJECTORY_STEP_METERS) {
            if (displacement != null) trajectoryDistanceMeters += displacement
            trajectory.addLast(point)
            lastTrajectoryPoint = point
            compactTrajectoryIfNeeded()
            trajectoryDirty = true
        }
    }

    @Synchronized
    private fun onArCoreStatus(tracking: Boolean, status: String) {
        if (active == null || !useArCoreReference) return
        if (!tracking) {
            latestReference = null
            trajectoryStabilizer.markTrackingLost()
        }
        mutableState.value = mutableState.value.copy(
            orientationAvailable = tracking && latestReference != null,
            referenceAvailable = tracking && latestReference?.velocityValid == true,
            referenceStatus = status,
            message = status.takeUnless { tracking },
        )
    }

    private fun compactTrajectoryIfNeeded() {
        if (trajectory.size < MAX_TRAJECTORY_POINTS) return
        val reduced = trajectory.filterIndexed { index, _ ->
            index == 0 || index % 2 == 0 || index == trajectory.size - 1
        }
        trajectory.clear()
        trajectory.addAll(reduced)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        private const val MAX_TRAJECTORY_POINTS = 2_000
        private const val MIN_TRAJECTORY_STEP_METERS = 0.04f
        private const val MAX_REFERENCE_AGE_NS = 250_000_000L
        private const val MAX_REFERENCE_FUTURE_SKEW_NS = 50_000_000L
        private const val LIVE_FRAME_BUFFER_SIZE = 256
        private const val UI_UPDATE_PERIOD_NS = 100_000_000L
    }
}
