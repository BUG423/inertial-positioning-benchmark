package com.bug423.inertiallab.sensor

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min

data class SamplingPlan(
    val targetRateHz: Int,
    val effectiveRateHz: Int,
    val hardwareMaximumRateHz: Int,
) {
    val requiresResampling: Boolean get() = effectiveRateHz != targetRateHz
    val requestedPeriodUs: Int get() = ceil(1_000_000.0 / effectiveRateHz).toInt()
}

/**
 * IMUNet's canonical capture rate is 200 Hz. Android exposes a sensor's shortest supported
 * interval through minDelay (microseconds); the usable IMU rate is limited by the slower of the
 * accelerometer and gyroscope. A zero minDelay is treated as unknown rather than as zero Hz.
 */
fun createSamplingPlan(
    accelerometerMinDelayUs: Int,
    gyroscopeMinDelayUs: Int,
    targetRateHz: Int = CANONICAL_SAMPLE_RATE_HZ,
): SamplingPlan {
    require(targetRateHz > 0) { "目标采样率必须大于 0" }

    fun maximumRate(minDelayUs: Int): Double =
        if (minDelayUs > 0) 1_000_000.0 / minDelayUs else targetRateHz.toDouble()

    val commonMaximum = min(
        maximumRate(accelerometerMinDelayUs),
        maximumRate(gyroscopeMinDelayUs),
    )
    val hardwareMaximumRateHz = floor(commonMaximum + RATE_EPSILON).toInt().coerceAtLeast(1)
    return SamplingPlan(
        targetRateHz = targetRateHz,
        effectiveRateHz = min(targetRateHz, hardwareMaximumRateHz),
        hardwareMaximumRateHz = hardwareMaximumRateHz,
    )
}

const val CANONICAL_SAMPLE_RATE_HZ = 200
private const val RATE_EPSILON = 1e-6
