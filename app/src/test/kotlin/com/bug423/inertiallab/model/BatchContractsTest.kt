package com.bug423.inertiallab.model

import org.junit.Assert.assertEquals
import org.junit.Test

class BatchContractsTest {
    @Test
    fun `overall progress combines completed runs and current run`() {
        val progress = BenchmarkBatchProgress(
            completedRuns = 2,
            totalRuns = 4,
            currentModelDirectory = "model-a-1",
            currentSequenceId = "walk-01",
            currentRunProgress = 0.5f,
        )

        assertEquals(0.625f, progress.overallFraction, 0.0001f)
    }

    @Test
    fun `overall progress is bounded for callback jitter`() {
        assertEquals(
            1f,
            BenchmarkBatchProgress(3, 3, null, null, 2f).overallFraction,
            0f,
        )
        assertEquals(
            0f,
            BenchmarkBatchProgress(0, 0, null, null, 0.5f).overallFraction,
            0f,
        )
    }

    @Test
    fun `model batch import summary counts successes and failures`() {
        val imported = InstalledModel(
            manifest = validManifest(),
            sizeBytes = 42,
            directoryName = "model-a-1",
        )
        val result = ModelImportBatchResult(
            imported = listOf(imported),
            failures = listOf(ModelImportFailure("broken.zip", "缺少 manifest.json")),
        )

        assertEquals(2, result.requestedCount)
        assertEquals(1, result.successfulCount)
    }

    private fun validManifest() = ModelManifest(
        id = "model-a",
        name = "Model A",
        version = "1",
        input = ModelInput(
            shape = listOf(1, 200, 6),
            sample_rate_hz = 200,
            channels = ModelManifest.REQUIRED_CHANNELS,
        ),
        output = ModelOutput(shape = listOf(1, 3), dimensions = 3),
    )
}
