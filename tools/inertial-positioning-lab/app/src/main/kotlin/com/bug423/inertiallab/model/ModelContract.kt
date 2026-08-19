package com.bug423.inertiallab.model

import kotlinx.serialization.Serializable
import kotlinx.serialization.Required

@Serializable
data class ModelInput(
    val shape: List<Int>,
    @Required
    val layout: String = "NTC",
    val sample_rate_hz: Int,
    val channels: List<String>,
    @Required
    val dtype: String = "float32",
    @Required
    val mean: List<Float> = List(6) { 0f },
    @Required
    val std: List<Float> = List(6) { 1f },
)

@Serializable
data class ModelOutput(
    val shape: List<Int>,
    @Required
    val task: String = "velocity",
    @Required
    val coordinate_frame: String = "body",
    val dimensions: Int,
    @Required
    val unit: String = "m/s",
)

@Serializable
data class BenchmarkSettings(
    val threads: Int = 2,
    val warmup_runs: Int = 10,
    val stride: Int = 10,
    val max_windows: Int = 5000,
)

@Serializable
data class ModelManifest(
    @Required
    val schema_version: Int = 1,
    val id: String,
    val name: String,
    val version: String,
    @Required
    val runtime: String = "tflite",
    @Required
    val model_file: String = "model.tflite",
    val input: ModelInput,
    val output: ModelOutput,
    val benchmark: BenchmarkSettings = BenchmarkSettings(),
    val description: String = "",
) {
    fun validate(): List<String> = buildList {
        if (schema_version != 1) add("schema_version 必须为 1")
        if (!ID.matches(id)) add("id 只能包含小写字母、数字、点、下划线和短横线")
        if (name.isBlank()) add("name 不能为空")
        if (version.isBlank()) add("version 不能为空")
        if (runtime != "tflite") add("runtime 目前仅支持 tflite")
        if (model_file != "model.tflite") add("model_file 必须为 model.tflite")
        if (input.layout != "NTC") add("input.layout 必须为 NTC")
        if (input.shape.size != 3 || input.shape.firstOrNull() != 1 || input.shape.lastOrNull() != 6) {
            add("input.shape 必须为 [1, T, 6]")
        }
        if (input.shape.getOrNull(1) !in 2..10000) add("输入窗口 T 必须在 2..10000")
        if (input.sample_rate_hz !in 10..1000) add("sample_rate_hz 必须在 10..1000")
        if (input.channels != REQUIRED_CHANNELS) add("输入通道顺序必须为 ${REQUIRED_CHANNELS.joinToString()}")
        if (input.dtype != "float32") add("input.dtype 必须为 float32")
        if (input.mean.size != 6 || input.std.size != 6 ||
            input.mean.any { !it.isFinite() } || input.std.any { !it.isFinite() || it <= 0f }
        ) {
            add("mean/std 必须各含 6 个有限数值，且 std 必须大于 0")
        }
        if (output.shape != listOf(1, output.dimensions) || output.dimensions !in 2..3) {
            add("output.shape 必须为 [1,2] 或 [1,3]，且与 dimensions 一致")
        }
        if (output.task != "velocity" || output.unit != "m/s") add("输出任务固定为 m/s 速度")
        if (output.coordinate_frame !in setOf("body", "world")) add("coordinate_frame 仅支持 body/world")
        if (benchmark.threads !in 1..8) add("threads 必须在 1..8")
        if (benchmark.warmup_runs !in 0..100) add("warmup_runs 必须在 0..100")
        if (benchmark.stride !in 1..input.shape.getOrElse(1) { 1 }) add("stride 超出窗口范围")
        if (benchmark.max_windows !in 1..100000) add("max_windows 必须在 1..100000")
    }

    companion object {
        val REQUIRED_CHANNELS = listOf(
            "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z",
        )
        private val ID = Regex("[a-z0-9][a-z0-9._-]{1,63}")
    }
}

data class InstalledModel(
    val manifest: ModelManifest,
    val sizeBytes: Long,
    val directoryName: String,
)

/** Result of importing several model packages in one picker operation. */
data class ModelImportBatchResult(
    val imported: List<InstalledModel>,
    val failures: List<ModelImportFailure>,
) {
    val requestedCount: Int get() = imported.size + failures.size
    val successfulCount: Int get() = imported.size
}

data class ModelImportFailure(
    val source: String,
    val reason: String,
)
