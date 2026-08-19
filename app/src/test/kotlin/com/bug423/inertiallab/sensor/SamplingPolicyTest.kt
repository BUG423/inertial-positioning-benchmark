package com.bug423.inertiallab.sensor

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SamplingPolicyTest {
    @Test
    fun keepsCanonicalRateWhenBothSensorsSupportIt() {
        val plan = createSamplingPlan(
            accelerometerMinDelayUs = 2_500,
            gyroscopeMinDelayUs = 5_000,
        )

        assertEquals(200, plan.effectiveRateHz)
        assertEquals(200, plan.hardwareMaximumRateHz)
        assertEquals(5_000, plan.requestedPeriodUs)
        assertFalse(plan.requiresResampling)
    }

    @Test
    fun fallsBackToSlowerSensorMaximum() {
        val plan = createSamplingPlan(
            accelerometerMinDelayUs = 10_000,
            gyroscopeMinDelayUs = 5_000,
        )

        assertEquals(100, plan.effectiveRateHz)
        assertEquals(100, plan.hardwareMaximumRateHz)
        assertEquals(10_000, plan.requestedPeriodUs)
        assertTrue(plan.requiresResampling)
    }

    @Test
    fun treatsUnknownMinimumDelayAsCanonicalCapability() {
        val plan = createSamplingPlan(
            accelerometerMinDelayUs = 0,
            gyroscopeMinDelayUs = 0,
        )

        assertEquals(200, plan.effectiveRateHz)
        assertFalse(plan.requiresResampling)
    }
}
