package com.bug423.inertiallab.data

import java.io.StringReader
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureContractTest {
    @Test
    fun validCsvReturnsMeasuredStats() {
        val stats = readCanonicalCsv(csv(row(0.0), row(0.01, mapOf(19 to "1"))))

        assertEquals(2L, stats.sampleCount)
        assertEquals(0.01, stats.durationSeconds, 1e-12)
        assertTrue(stats.hasReference)
    }

    @Test
    fun masksMustBeExactlyZeroOrOne() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            readCanonicalCsv(csv(row(0.0), row(0.01, mapOf(17 to "true"))))
        }

        assertTrue(error.message.orEmpty().contains("valid_imu 只能为 0 或 1"))
    }

    @Test
    fun numericValuesMustBeFinite() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            readCanonicalCsv(csv(row(0.0), row(0.01, mapOf(1 to "NaN"))))
        }

        assertTrue(error.message.orEmpty().contains("gyro_x 必须是有限数值"))
    }

    @Test
    fun quaternionsMustBeNormalized() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            readCanonicalCsv(csv(row(0.0), row(0.01, mapOf(7 to "0"))))
        }

        assertTrue(error.message.orEmpty().contains("四元数必须归一化"))
    }

    private fun csv(vararg rows: String) = StringReader(
        buildString {
            appendLine(CANONICAL_CSV_HEADER)
            rows.forEach(::appendLine)
        },
    ).buffered()

    private fun row(timestamp: Double, overrides: Map<Int, String> = emptyMap()): String {
        val values = MutableList(24) { "0" }
        values[0] = timestamp.toString()
        values[7] = "1"
        values[17] = "1"
        values[18] = "1"
        values[19] = "0"
        values[20] = ""
        values[21] = ""
        values[22] = ""
        values[23] = ""
        overrides.forEach { (index, value) -> values[index] = value }
        return values.joinToString(",")
    }
}
