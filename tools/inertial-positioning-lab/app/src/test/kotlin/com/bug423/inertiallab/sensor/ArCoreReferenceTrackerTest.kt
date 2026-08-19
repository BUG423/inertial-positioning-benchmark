package com.bug423.inertiallab.sensor

import com.bug423.inertiallab.data.Vec3
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Test

class ArCoreReferenceTrackerTest {
    @Test
    fun mapsArCoreAxesToLocalRightForwardUp() {
        val pose = arCoreToCanonical(
            translation = floatArrayOf(1f, 2f, 3f),
            rotationXyzw = floatArrayOf(0f, 0f, 0f, 1f),
        )

        assertEquals(1f, pose.position.x, 1e-6f)
        assertEquals(-3f, pose.position.y, 1e-6f)
        assertEquals(2f, pose.position.z, 1e-6f)
        val halfSqrt = sqrt(0.5f)
        assertEquals(halfSqrt, pose.orientation.w, 1e-6f)
        assertEquals(halfSqrt, pose.orientation.x, 1e-6f)
        assertEquals(0f, pose.orientation.y, 1e-6f)
        assertEquals(0f, pose.orientation.z, 1e-6f)
    }

    @Test
    fun normalizesPublishedQuaternion() {
        val pose = arCoreToCanonical(
            translation = floatArrayOf(0f, 0f, 0f),
            rotationXyzw = floatArrayOf(0f, 0f, 0f, 2f),
        )
        val q = pose.orientation
        val norm = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)

        assertEquals(1f, norm, 1e-6f)
    }

    @Test
    fun firstPublishedPointIsExactOriginAndRebasePreservesContinuity() {
        val noisyAnchorPoint = Vec3(0.014f, -0.009f, 0.003f)
        val first = rebaseArCorePosition(noisyAnchorPoint, noisyAnchorPoint, Vec3())
        assertEquals(Vec3(), first)

        val previousPosition = Vec3(1.2f, -0.4f, 0.1f)
        val newWorldPoint = Vec3(8f, 3f, -2f)
        val firstAfterRecovery = rebaseArCorePosition(
            rawPosition = newWorldPoint,
            rawSegmentOrigin = newWorldPoint,
            segmentOffset = previousPosition,
        )
        assertEquals(previousPosition, firstAfterRecovery)
    }
}
