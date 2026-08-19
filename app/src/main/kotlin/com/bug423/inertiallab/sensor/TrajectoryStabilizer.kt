package com.bug423.inertiallab.sensor

import com.bug423.inertiallab.data.Vec3
import kotlin.math.abs
import kotlin.math.sqrt

internal data class StabilizedTrajectoryPoint(
    val position: Vec3,
    val stationary: Boolean,
    val discontinuityRejected: Boolean,
)

/**
 * Stabilizes the on-screen trajectory without changing the ARCore samples persisted to disk.
 * It locks slow visual drift while the IMU is quiet, preserves continuity after tracking gaps,
 * rejects physically implausible frame jumps, and otherwise applies a short low-pass filter.
 */
internal class TrajectoryStabilizer {
    private var filteredPosition: Vec3? = null
    private var rawOffset = Vec3()
    private var lastRawPosition: Vec3? = null
    private var lastTimestampNs = NO_TIMESTAMP
    private var stationarySinceNs = 0L
    private var continuityPending = false

    fun reset() {
        filteredPosition = null
        rawOffset = Vec3()
        lastRawPosition = null
        lastTimestampNs = NO_TIMESTAMP
        stationarySinceNs = 0L
        continuityPending = false
    }

    fun markTrackingLost() {
        continuityPending = filteredPosition != null
        stationarySinceNs = 0L
        lastRawPosition = null
        lastTimestampNs = NO_TIMESTAMP
    }

    fun update(
        timestampNs: Long,
        rawPosition: Vec3,
        accelerometer: Vec3,
        gyroscope: Vec3,
    ): StabilizedTrajectoryPoint {
        val previousFiltered = filteredPosition
        if (previousFiltered == null) {
            filteredPosition = rawPosition
            lastRawPosition = rawPosition
            lastTimestampNs = timestampNs
            return StabilizedTrajectoryPoint(rawPosition, stationary = false, discontinuityRejected = false)
        }

        if (continuityPending || lastTimestampNs == NO_TIMESTAMP || timestampNs <= lastTimestampNs) {
            rawOffset = previousFiltered - rawPosition
            continuityPending = false
            lastRawPosition = rawPosition
            lastTimestampNs = timestampNs
            return StabilizedTrajectoryPoint(previousFiltered, stationary = false, discontinuityRejected = false)
        }

        val dtSeconds = (timestampNs - lastTimestampNs) / 1_000_000_000.0
        if (dtSeconds > MAX_CONTINUOUS_GAP_SECONDS) {
            rawOffset = previousFiltered - rawPosition
            lastRawPosition = rawPosition
            lastTimestampNs = timestampNs
            stationarySinceNs = 0L
            return StabilizedTrajectoryPoint(previousFiltered, stationary = false, discontinuityRejected = false)
        }

        val previousRaw = lastRawPosition ?: rawPosition
        val rawSpeed = rawPosition.distanceTo(previousRaw) / dtSeconds.toFloat()
        val adjusted = rawPosition + rawOffset
        val imuQuiet = gyroscope.magnitude() <= STATIONARY_GYROSCOPE_RAD_PER_SECOND &&
            abs(accelerometer.magnitude() - STANDARD_GRAVITY) <= STATIONARY_ACCELERATION_TOLERANCE
        val visuallyQuiet = rawSpeed <= STATIONARY_VISUAL_SPEED_METERS_PER_SECOND
        if (imuQuiet && visuallyQuiet) {
            if (stationarySinceNs == 0L) stationarySinceNs = timestampNs
        } else {
            stationarySinceNs = 0L
        }

        val stationary = stationarySinceNs != 0L &&
            timestampNs - stationarySinceNs >= STATIONARY_CONFIRMATION_NS
        if (stationary) {
            rawOffset = previousFiltered - rawPosition
            lastRawPosition = rawPosition
            lastTimestampNs = timestampNs
            return StabilizedTrajectoryPoint(previousFiltered, stationary = true, discontinuityRejected = false)
        }

        if (rawSpeed > MAXIMUM_PLAUSIBLE_SPEED_METERS_PER_SECOND) {
            rawOffset = previousFiltered - rawPosition
            lastRawPosition = rawPosition
            lastTimestampNs = timestampNs
            stationarySinceNs = 0L
            return StabilizedTrajectoryPoint(previousFiltered, stationary = false, discontinuityRejected = true)
        }

        val alpha = (dtSeconds / (SMOOTHING_TIME_CONSTANT_SECONDS + dtSeconds)).toFloat()
        val filtered = previousFiltered.lerp(adjusted, alpha)
        filteredPosition = filtered
        lastRawPosition = rawPosition
        lastTimestampNs = timestampNs
        return StabilizedTrajectoryPoint(filtered, stationary = false, discontinuityRejected = false)
    }

    companion object {
        private const val STANDARD_GRAVITY = 9.80665f
        private const val STATIONARY_GYROSCOPE_RAD_PER_SECOND = 0.10f
        private const val STATIONARY_ACCELERATION_TOLERANCE = 0.45f
        private const val STATIONARY_VISUAL_SPEED_METERS_PER_SECOND = 0.08f
        private const val STATIONARY_CONFIRMATION_NS = 450_000_000L
        private const val MAXIMUM_PLAUSIBLE_SPEED_METERS_PER_SECOND = 4.0f
        private const val MAX_CONTINUOUS_GAP_SECONDS = 0.5
        private const val SMOOTHING_TIME_CONSTANT_SECONDS = 0.12
        private const val NO_TIMESTAMP = Long.MIN_VALUE
    }
}

private fun Vec3.magnitude() = sqrt(x * x + y * y + z * z)

private fun Vec3.distanceTo(other: Vec3) = (this - other).magnitude()

private fun Vec3.lerp(other: Vec3, alpha: Float) = Vec3(
    x = x + (other.x - x) * alpha,
    y = y + (other.y - y) * alpha,
    z = z + (other.z - z) * alpha,
)

private operator fun Vec3.plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)

private operator fun Vec3.minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
