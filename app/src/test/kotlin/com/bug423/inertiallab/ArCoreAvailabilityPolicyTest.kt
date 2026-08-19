package com.bug423.inertiallab

import com.google.ar.core.ArCoreApk
import org.junit.Assert.assertEquals
import org.junit.Test

class ArCoreAvailabilityPolicyTest {
    @Test
    fun unknownLookupUsesLocalSessionAsFinalCheck() {
        assertEquals(
            ArCoreSetupAction.PROBE_LOCAL_SESSION,
            arCoreSetupAction(ArCoreApk.Availability.UNKNOWN_ERROR),
        )
        assertEquals(
            ArCoreSetupAction.PROBE_LOCAL_SESSION,
            arCoreSetupAction(ArCoreApk.Availability.UNKNOWN_TIMED_OUT),
        )
    }

    @Test
    fun unsupportedLookupIsNotAllowedToBlockLocalSessionProbe() {
        assertEquals(
            ArCoreSetupAction.PROBE_LOCAL_SESSION,
            arCoreSetupAction(ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE),
        )
    }

    @Test
    fun supportedResultUsesOfficialInstallFlow() {
        assertEquals(
            ArCoreSetupAction.REQUEST_INSTALL,
            arCoreSetupAction(ArCoreApk.Availability.SUPPORTED_INSTALLED),
        )
    }
}
