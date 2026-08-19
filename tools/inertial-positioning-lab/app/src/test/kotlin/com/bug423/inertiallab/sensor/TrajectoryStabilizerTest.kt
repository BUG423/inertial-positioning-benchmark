package com.bug423.inertiallab.sensor

import com.bug423.inertiallab.data.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TrajectoryStabilizerTest {
    @Test
    fun locksSlowVisualDriftWhileImuIsStationary() {
        val stabilizer = TrajectoryStabilizer()
        var output = Vec3()
        repeat(300) { index ->
            output = stabilizer.update(
                timestampNs = index * 33_333_333L,
                rawPosition = Vec3(index * .001f, 0f, 0f),
                accelerometer = Vec3(0f, 0f, 9.80665f),
                gyroscope = Vec3(),
            ).position
        }

        assertTrue("静止漂移应锁定在几厘米内，实际为 ${output.x}", output.x < .04f)
    }

    @Test
    fun followsDeliberateTranslation() {
        val stabilizer = TrajectoryStabilizer()
        var output = Vec3()
        repeat(101) { index ->
            output = stabilizer.update(
                timestampNs = index * 33_333_333L,
                rawPosition = Vec3(index * .01f, 0f, 0f),
                accelerometer = Vec3(0.6f, 0f, 9.80665f),
                gyroscope = Vec3(0f, 0f, .15f),
            ).position
        }

        assertTrue("真实平移不应被静止锁定，实际为 ${output.x}", output.x > .94f)
    }

    @Test
    fun trackingRecoveryDoesNotJumpTheDisplay() {
        val stabilizer = TrajectoryStabilizer()
        val acceleration = Vec3(0f, 0f, 9.80665f)
        stabilizer.update(0L, Vec3(), acceleration, Vec3())
        val beforeLoss = stabilizer.update(
            100_000_000L,
            Vec3(.1f, 0f, 0f),
            acceleration,
            Vec3(0f, 0f, .2f),
        ).position

        stabilizer.markTrackingLost()
        val recovered = stabilizer.update(
            1_000_000_000L,
            Vec3(4f, -3f, 1f),
            acceleration,
            Vec3(),
        ).position

        assertEquals(beforeLoss.x, recovered.x, 1e-6f)
        assertEquals(beforeLoss.y, recovered.y, 1e-6f)
        assertEquals(beforeLoss.z, recovered.z, 1e-6f)
    }

    @Test
    fun rejectsImplausibleSingleFrameJump() {
        val stabilizer = TrajectoryStabilizer()
        val acceleration = Vec3(0f, 0f, 9.80665f)
        stabilizer.update(0L, Vec3(), acceleration, Vec3())
        val output = stabilizer.update(
            33_333_333L,
            Vec3(1f, 0f, 0f),
            acceleration,
            Vec3(0f, 0f, .2f),
        )

        assertTrue(output.discontinuityRejected)
        assertEquals(Vec3(), output.position)
    }
}
