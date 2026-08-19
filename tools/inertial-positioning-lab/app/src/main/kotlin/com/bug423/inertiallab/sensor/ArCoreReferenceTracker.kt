package com.bug423.inertiallab.sensor

import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.bug423.inertiallab.data.QuaternionWxyz
import com.bug423.inertiallab.data.Vec3
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.math.sqrt

internal data class ArCoreReferenceSample(
    val observedAtNs: Long,
    val position: Vec3,
    val velocity: Vec3,
    val orientation: QuaternionWxyz,
    val velocityValid: Boolean,
)

internal data class CanonicalArCorePose(
    val position: Vec3,
    val orientation: QuaternionWxyz,
)

internal fun rebaseArCorePosition(
    rawPosition: Vec3,
    rawSegmentOrigin: Vec3,
    segmentOffset: Vec3,
): Vec3 = rawPosition - rawSegmentOrigin + segmentOffset

/** Maps ARCore's X-right/Y-up/Z-back axes to local X-right/Y-forward/Z-up. */
internal fun arCoreToCanonical(
    translation: FloatArray,
    rotationXyzw: FloatArray,
): CanonicalArCorePose {
    require(translation.size == 3) { "ARCore translation 必须包含 3 个数值" }
    require(rotationXyzw.size == 4) { "ARCore quaternion 必须包含 4 个数值" }
    require(translation.all(Float::isFinite) && rotationXyzw.all(Float::isFinite)) {
        "ARCore pose 包含 NaN 或 Inf"
    }
    val halfSqrt = sqrt(0.5f)
    val x = rotationXyzw[0]
    val y = rotationXyzw[1]
    val z = rotationXyzw[2]
    val w = rotationXyzw[3]
    val transformed = QuaternionWxyz(
        w = halfSqrt * (w - x),
        x = halfSqrt * (x + w),
        y = halfSqrt * (y - z),
        z = halfSqrt * (z + y),
    )
    val norm = sqrt(
        transformed.w * transformed.w + transformed.x * transformed.x +
            transformed.y * transformed.y + transformed.z * transformed.z,
    )
    require(norm > 0f && norm.isFinite()) { "ARCore quaternion 无效" }
    return CanonicalArCorePose(
        position = Vec3(translation[0], -translation[2], translation[1]),
        orientation = QuaternionWxyz(
            transformed.w / norm,
            transformed.x / norm,
            transformed.y / norm,
            transformed.z / norm,
        ),
    )
}

/**
 * Runs a headless ARCore session on its own GL thread and publishes the Android sensor pose.
 * ARCore still owns the camera internally for VIO, but the app never reads or retains its frames.
 */
internal class ArCoreReferenceTracker(context: Context) {
    private val appContext = context.applicationContext
    private val worker = HandlerThread("ArCoreReference").apply { start() }
    private val handler = Handler(worker.looper)
    private var session: Session? = null
    private var anchor: Anchor? = null
    private var textureContext: ExternalTextureContext? = null
    private var requested = false
    private var running = false
    private var closed = false
    private var restartAttempts = 0
    private var restartRunnable: Runnable? = null
    private var lastFrameTimestampNs = 0L
    private var lastPoseTimestampNs = 0L
    private var trackingStableSinceNs = 0L
    private var originCandidateTranslation: FloatArray? = null
    private var anchorUntrackedSinceNs = 0L
    private var segmentRawOrigin: Vec3? = null
    private var segmentOffset = Vec3()
    private var lastPosition: Vec3? = null
    private var lastPublishedStatus: Pair<Boolean, String>? = null
    private var onSample: ((ArCoreReferenceSample) -> Unit)? = null
    private var onStatus: ((Boolean, String) -> Unit)? = null

    fun start(
        onSample: (ArCoreReferenceSample) -> Unit,
        onStatus: (tracking: Boolean, status: String) -> Unit,
    ) {
        check(!closed) { "ARCore tracker 已关闭" }
        check(handler.post {
            stopInternal(clearCallbacks = false)
            this.onSample = onSample
            this.onStatus = onStatus
            lastPublishedStatus = null
            requested = true
            publishStatus(false, "正在初始化 ARCore")
            openSession()
        }) { "无法调度 ARCore 启动" }
    }

    private fun openSession() {
        if (!requested || closed) return
        try {
            val gl = ExternalTextureContext.create()
            textureContext = gl
            val newSession = Session(appContext)
            session = newSession
            val config = Config(newSession).apply {
                updateMode = Config.UpdateMode.BLOCKING
                planeFindingMode = Config.PlaneFindingMode.DISABLED
                lightEstimationMode = Config.LightEstimationMode.DISABLED
                focusMode = Config.FocusMode.AUTO
            }
            newSession.configure(config)
            newSession.setCameraTextureName(gl.textureId)
            newSession.resume()
            running = true
            handler.post(updateLoop)
        } catch (error: Throwable) {
            scheduleRestart(error)
        }
    }

    fun stop() {
        if (closed) return
        if (Looper.myLooper() == worker.looper) {
            stopInternal()
            return
        }
        val stopped = CountDownLatch(1)
        if (!handler.post {
                stopInternal()
                stopped.countDown()
            }
        ) return
        runCatching { stopped.await(STOP_TIMEOUT_SECONDS, TimeUnit.SECONDS) }
    }

    fun close() {
        if (closed) return
        stop()
        closed = true
        worker.quitSafely()
    }

    private val updateLoop = object : Runnable {
        override fun run() {
            if (!running) return
            try {
                val activeSession = checkNotNull(session)
                val gl = checkNotNull(textureContext)
                activeSession.setCameraTextureName(gl.textureId)
                val frame = activeSession.update()
                if (frame.timestamp != 0L && frame.timestamp != lastFrameTimestampNs) {
                    lastFrameTimestampNs = frame.timestamp
                    val camera = frame.camera
                    if (camera.trackingState == TrackingState.TRACKING) {
                        publishPose(activeSession, frame.androidSensorPose, frame.timestamp)
                    } else {
                        lastPoseTimestampNs = 0L
                        if (anchor == null) {
                            trackingStableSinceNs = 0L
                            originCandidateTranslation = null
                        }
                        publishStatus(
                            false,
                            "${camera.trackingFailureReason.message()}；IMU 仍在采集，等待自动恢复",
                        )
                    }
                }
            } catch (error: Throwable) {
                scheduleRestart(error)
                return
            }
            if (running) handler.post(this)
        }
    }

    private fun publishPose(activeSession: Session, sensorPose: Pose, frameTimestampNs: Long) {
        if (anchor == null) {
            val translation = sensorPose.translation
            val candidate = originCandidateTranslation
            val movedFromCandidate = candidate != null && translationDistance(candidate, translation) >
                ORIGIN_STABILITY_RADIUS_METERS
            if (candidate == null || movedFromCandidate) {
                originCandidateTranslation = translation.copyOf()
                trackingStableSinceNs = frameTimestampNs
            }
            val stableDurationNs = frameTimestampNs - trackingStableSinceNs
            if (stableDurationNs < ORIGIN_STABILIZATION_NS) {
                val remainingMs = (ORIGIN_STABILIZATION_NS - stableDurationNs) / 1_000_000L
                publishStatus(false, "请保持手机平稳，正在稳定原点（${remainingMs} ms）")
                return
            }
            createOriginAnchor(activeSession, sensorPose, preservePosition = false)
            originCandidateTranslation = null
        }

        var origin = checkNotNull(anchor)
        if (origin.trackingState != TrackingState.TRACKING) {
            if (anchorUntrackedSinceNs == 0L) anchorUntrackedSinceNs = frameTimestampNs
            val shouldRebase = origin.trackingState == TrackingState.STOPPED ||
                frameTimestampNs - anchorUntrackedSinceNs >= ANCHOR_REBASE_TIMEOUT_NS
            if (!shouldRebase) {
                lastPoseTimestampNs = 0L
                publishStatus(false, "ARCore 原点暂时不可跟踪；IMU 仍在采集，正在恢复")
                return
            }
            Log.w(TAG, "ARCore origin anchor lost; rebasing without breaking trajectory")
            createOriginAnchor(activeSession, sensorPose, preservePosition = true)
            origin = checkNotNull(anchor)
            if (origin.trackingState != TrackingState.TRACKING) {
                publishStatus(false, "ARCore 正在重建原点；IMU 仍在采集")
                return
            }
        }
        anchorUntrackedSinceNs = 0L
        restartAttempts = 0
        val relativePose = origin.pose.inverse().compose(sensorPose)
        val rawCanonical = arCoreToCanonical(relativePose.translation, relativePose.rotationQuaternion)
        val rawOrigin = segmentRawOrigin ?: rawCanonical.position.also { segmentRawOrigin = it }
        val position = rebaseArCorePosition(rawCanonical.position, rawOrigin, segmentOffset)
        val previous = lastPosition
        val elapsed = (frameTimestampNs - lastPoseTimestampNs) / 1_000_000_000.0
        val velocityValid = previous != null && elapsed in MIN_VELOCITY_DT_SECONDS..MAX_VELOCITY_DT_SECONDS
        val velocity = if (velocityValid) {
            Vec3(
                ((position.x - previous.x) / elapsed).toFloat(),
                ((position.y - previous.y) / elapsed).toFloat(),
                ((position.z - previous.z) / elapsed).toFloat(),
            )
        } else {
            Vec3()
        }
        lastPosition = position
        lastPoseTimestampNs = frameTimestampNs
        onSample?.invoke(
            ArCoreReferenceSample(
                observedAtNs = SystemClock.elapsedRealtimeNanos(),
                position = position,
                velocity = velocity,
                orientation = rawCanonical.orientation,
                velocityValid = velocityValid,
            ),
        )
        publishStatus(true, "ARCore TRACKING")
    }

    private fun createOriginAnchor(
        activeSession: Session,
        sensorPose: Pose,
        preservePosition: Boolean,
    ) {
        if (preservePosition) segmentOffset = lastPosition ?: segmentOffset
        anchor?.let { runCatching { it.detach() } }
        anchor = activeSession.createAnchor(
            Pose(sensorPose.translation, floatArrayOf(0f, 0f, 0f, 1f)),
        )
        segmentRawOrigin = null
        lastPoseTimestampNs = 0L
        anchorUntrackedSinceNs = 0L
    }

    private fun scheduleRestart(error: Throwable) {
        if (!requested || closed) return
        closeSessionResources(preservePosition = true)
        restartAttempts += 1
        val delayMs = min(
            RESTART_MAX_DELAY_MS,
            RESTART_INITIAL_DELAY_MS * (1L shl (restartAttempts - 1).coerceAtMost(3)),
        )
        publishStatus(
            false,
            "ARCore 暂时中断：${error.readableMessage()}；IMU 仍在采集，${delayMs / 1000.0} 秒后自动重连",
        )
        Log.w(TAG, "ARCore session interrupted; retrying in $delayMs ms", error)
        restartRunnable?.let(handler::removeCallbacks)
        val retry = Runnable {
            restartRunnable = null
            if (requested && !closed) {
                publishStatus(false, "正在重新连接 ARCore；IMU 仍在采集")
                openSession()
            }
        }
        restartRunnable = retry
        handler.postDelayed(retry, delayMs)
    }

    private fun publishStatus(tracking: Boolean, status: String) {
        val value = tracking to status
        if (lastPublishedStatus == value) return
        lastPublishedStatus = value
        onStatus?.invoke(tracking, status)
    }

    private fun stopInternal(clearCallbacks: Boolean = true) {
        requested = false
        restartRunnable?.let(handler::removeCallbacks)
        restartRunnable = null
        handler.removeCallbacks(updateLoop)
        closeSessionResources(preservePosition = false)
        restartAttempts = 0
        segmentOffset = Vec3()
        segmentRawOrigin = null
        lastPosition = null
        trackingStableSinceNs = 0L
        originCandidateTranslation = null
        anchorUntrackedSinceNs = 0L
        lastPublishedStatus = null
        if (clearCallbacks) {
            onSample = null
            onStatus = null
        }
    }

    private fun closeSessionResources(preservePosition: Boolean) {
        running = false
        if (preservePosition) segmentOffset = lastPosition ?: segmentOffset
        anchor?.let { runCatching { it.detach() } }
        anchor = null
        session?.let { activeSession ->
            runCatching { activeSession.pause() }
            runCatching { activeSession.close() }
        }
        session = null
        textureContext?.let { runCatching { it.close() } }
        textureContext = null
        lastFrameTimestampNs = 0L
        lastPoseTimestampNs = 0L
        trackingStableSinceNs = 0L
        originCandidateTranslation = null
        anchorUntrackedSinceNs = 0L
        segmentRawOrigin = null
    }

    private class ExternalTextureContext private constructor(
        private val display: EGLDisplay,
        private val context: EGLContext,
        private val surface: EGLSurface,
        val textureId: Int,
    ) {
        fun close() {
            GLES20.glDeleteTextures(1, intArrayOf(textureId), 0)
            EGL14.eglMakeCurrent(
                display,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_CONTEXT,
            )
            EGL14.eglDestroySurface(display, surface)
            EGL14.eglDestroyContext(display, context)
            EGL14.eglTerminate(display)
        }

        companion object {
            fun create(): ExternalTextureContext {
                val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
                require(display != EGL14.EGL_NO_DISPLAY) { "无法创建 EGL display" }
                require(EGL14.eglInitialize(display, IntArray(2), 0, IntArray(2), 0)) {
                    "无法初始化 EGL"
                }
                val attributes = intArrayOf(
                    EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                    EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                    EGL14.EGL_RED_SIZE, 8,
                    EGL14.EGL_GREEN_SIZE, 8,
                    EGL14.EGL_BLUE_SIZE, 8,
                    EGL14.EGL_ALPHA_SIZE, 8,
                    EGL14.EGL_NONE,
                )
                val configs = arrayOfNulls<EGLConfig>(1)
                val configCount = IntArray(1)
                require(EGL14.eglChooseConfig(display, attributes, 0, configs, 0, 1, configCount, 0)) {
                    "无法选择 EGL config"
                }
                require(configCount[0] > 0) { "设备没有可用的 EGL config" }
                val config = requireNotNull(configs[0])
                val context = EGL14.eglCreateContext(
                    display,
                    config,
                    EGL14.EGL_NO_CONTEXT,
                    intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE),
                    0,
                )
                require(context != EGL14.EGL_NO_CONTEXT) { "无法创建 EGL context" }
                val surface = EGL14.eglCreatePbufferSurface(
                    display,
                    config,
                    intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE),
                    0,
                )
                require(surface != EGL14.EGL_NO_SURFACE) { "无法创建 EGL surface" }
                require(EGL14.eglMakeCurrent(display, surface, surface, context)) { "无法激活 EGL context" }
                val textures = IntArray(1)
                GLES20.glGenTextures(1, textures, 0)
                require(textures[0] != 0) { "无法创建 ARCore camera texture" }
                GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
                GLES20.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES20.GL_TEXTURE_MIN_FILTER,
                    GLES20.GL_LINEAR,
                )
                GLES20.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES20.GL_TEXTURE_MAG_FILTER,
                    GLES20.GL_LINEAR,
                )
                GLES20.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES20.GL_TEXTURE_WRAP_S,
                    GLES20.GL_CLAMP_TO_EDGE,
                )
                GLES20.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES20.GL_TEXTURE_WRAP_T,
                    GLES20.GL_CLAMP_TO_EDGE,
                )
                require(GLES20.glGetError() == GLES20.GL_NO_ERROR) { "ARCore camera texture 配置失败" }
                return ExternalTextureContext(display, context, surface, textures[0])
            }
        }
    }

    companion object {
        private const val TAG = "InertialLab.ArTracker"
        private const val STOP_TIMEOUT_SECONDS = 3L
        private const val MIN_VELOCITY_DT_SECONDS = 0.001
        private const val MAX_VELOCITY_DT_SECONDS = 0.5
        private const val ORIGIN_STABILIZATION_NS = 1_200_000_000L
        private const val ORIGIN_STABILITY_RADIUS_METERS = 0.04f
        private const val ANCHOR_REBASE_TIMEOUT_NS = 2_000_000_000L
        private const val RESTART_INITIAL_DELAY_MS = 1_000L
        private const val RESTART_MAX_DELAY_MS = 8_000L
    }
}

private operator fun Vec3.plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
private operator fun Vec3.minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)

private fun translationDistance(first: FloatArray, second: FloatArray): Float {
    val dx = first[0] - second[0]
    val dy = first[1] - second[1]
    val dz = first[2] - second[2]
    return sqrt(dx * dx + dy * dy + dz * dz)
}

private fun TrackingFailureReason.message(): String = when (this) {
    TrackingFailureReason.NONE -> "ARCore 正在初始化"
    TrackingFailureReason.BAD_STATE -> "ARCore 状态异常"
    TrackingFailureReason.INSUFFICIENT_LIGHT -> "环境光线不足"
    TrackingFailureReason.EXCESSIVE_MOTION -> "移动过快"
    TrackingFailureReason.INSUFFICIENT_FEATURES -> "环境纹理不足"
    TrackingFailureReason.CAMERA_UNAVAILABLE -> "相机不可用"
}

private fun Throwable.readableMessage(): String = message ?: javaClass.simpleName
