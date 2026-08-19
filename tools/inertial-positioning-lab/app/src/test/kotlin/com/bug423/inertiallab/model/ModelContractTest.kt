package com.bug423.inertiallab.model

import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class ModelContractTest {
    private fun valid() = ModelManifest(
        id = "ronin-velocity",
        name = "RoNIN Velocity",
        version = "1.0.0",
        input = ModelInput(
            shape = listOf(1, 200, 6),
            sample_rate_hz = 100,
            channels = ModelManifest.REQUIRED_CHANNELS,
        ),
        output = ModelOutput(shape = listOf(1, 2), dimensions = 2),
    )

    @Test
    fun validContractHasNoErrors() {
        assertTrue(valid().validate().isEmpty())
    }

    @Test
    fun channelOrderIsPartOfContract() {
        val invalid = valid().copy(input = valid().input.copy(channels = valid().input.channels.reversed()))
        assertTrue(invalid.validate().any { it.contains("通道顺序") })
    }

    @Test
    fun normalizationMustBeFiniteAndPositive() {
        val negative = valid().copy(input = valid().input.copy(std = listOf(-1f, 1f, 1f, 1f, 1f, 1f)))
        val nonFinite = valid().copy(input = valid().input.copy(mean = listOf(Float.NaN, 0f, 0f, 0f, 0f, 0f)))
        assertTrue(negative.validate().any { it.contains("std 必须大于 0") })
        assertTrue(nonFinite.validate().any { it.contains("有限数值") })
    }

    @Test
    fun requiredDefaultedFieldsMustStillBeDeclaredInManifest() {
        val encoded = Json.encodeToJsonElement(ModelManifest.serializer(), valid()).jsonObject
        val inputWithoutMean = encoded.getValue("input").jsonObject.toMutableMap().apply {
            remove("mean")
        }
        val manifestWithoutMean = JsonObject(encoded.toMutableMap().apply {
            put("input", JsonObject(inputWithoutMean))
        })

        assertThrows(SerializationException::class.java) {
            Json.decodeFromJsonElement(ModelManifest.serializer(), manifestWithoutMean)
        }
    }
}
