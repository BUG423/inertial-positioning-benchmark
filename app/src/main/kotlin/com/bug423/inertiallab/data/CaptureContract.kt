package com.bug423.inertiallab.data

import java.io.BufferedReader
import kotlin.math.abs
import kotlin.math.sqrt

internal data class CanonicalCsvStats(
    val sampleCount: Long,
    val durationSeconds: Double,
    val hasReference: Boolean,
)

internal fun readCanonicalCsv(
    reader: BufferedReader,
    onFrame: (SensorFrame) -> Unit = {},
): CanonicalCsvStats {
    require(reader.readLine() == CANONICAL_CSV_HEADER) { "CSV 字段不兼容" }
    var count = 0L
    var firstTimestamp = 0.0
    var lastTimestamp = Double.NEGATIVE_INFINITY
    var referenceSeen = false

    reader.forEachLine { line ->
        if (line.isNotBlank()) {
            val csvLine = count + 2
            require(line.length <= MAX_CSV_LINE_CHARS) { "CSV 第 $csvLine 行过长" }
            val frame = try {
                parseCanonicalFrame(line)
            } catch (error: IllegalArgumentException) {
                throw IllegalArgumentException("CSV 第 $csvLine 行：${error.message}", error)
            }
            require(frame.timestampSeconds > lastTimestamp) { "CSV 第 $csvLine 行：时间戳必须严格递增" }
            if (count == 0L) firstTimestamp = frame.timestampSeconds
            lastTimestamp = frame.timestampSeconds
            referenceSeen = referenceSeen || frame.validPosition
            count += 1
            onFrame(frame)
        }
    }
    require(count >= 2) { "数据集至少需要 2 个样本" }
    return CanonicalCsvStats(
        sampleCount = count,
        durationSeconds = lastTimestamp - firstTimestamp,
        hasReference = referenceSeen,
    )
}

internal fun parseCanonicalFrame(line: String): SensorFrame {
    val value = line.split(',')
    require(value.size == CSV_COLUMN_COUNT) { "字段数应为 $CSV_COLUMN_COUNT，实际为 ${value.size}" }

    fun double(index: Int, name: String): Double {
        val parsed = value[index].toDoubleOrNull()
        require(parsed != null && parsed.isFinite()) { "$name 必须是有限数值" }
        return parsed
    }

    fun float(index: Int, name: String): Float {
        val parsed = value[index].toFloatOrNull()
        require(parsed != null && parsed.isFinite()) { "$name 必须是有限数值" }
        return parsed
    }

    fun optionalDouble(index: Int, name: String): Double? = value[index].takeIf(String::isNotBlank)?.let {
        val parsed = it.toDoubleOrNull()
        require(parsed != null && parsed.isFinite()) { "$name 必须为空或有限数值" }
        parsed
    }

    fun optionalFloat(index: Int, name: String): Float? = value[index].takeIf(String::isNotBlank)?.let {
        val parsed = it.toFloatOrNull()
        require(parsed != null && parsed.isFinite()) { "$name 必须为空或有限数值" }
        parsed
    }

    fun mask(index: Int, name: String): Boolean = when (value[index]) {
        "0" -> false
        "1" -> true
        else -> throw IllegalArgumentException("$name 只能为 0 或 1")
    }

    val timestamp = double(0, "timestamp")
    require(timestamp >= 0.0) { "timestamp 不得为负数" }
    val orientation = QuaternionWxyz(
        float(7, "orientation_w"),
        float(8, "orientation_x"),
        float(9, "orientation_y"),
        float(10, "orientation_z"),
    )
    val orientationNorm = sqrt(
        orientation.w.toDouble() * orientation.w +
            orientation.x.toDouble() * orientation.x +
            orientation.y.toDouble() * orientation.y +
            orientation.z.toDouble() * orientation.z,
    )
    require(abs(orientationNorm - 1.0) <= QUATERNION_TOLERANCE) {
        "orientation 四元数必须归一化（容差 $QUATERNION_TOLERANCE）"
    }
    val latitude = optionalDouble(20, "latitude")
    val longitude = optionalDouble(21, "longitude")
    require((latitude == null) == (longitude == null)) { "latitude 与 longitude 必须同时为空或同时存在" }
    latitude?.let { require(it in -90.0..90.0) { "latitude 必须在 -90..90" } }
    longitude?.let { require(it in -180.0..180.0) { "longitude 必须在 -180..180" } }
    val accuracy = optionalFloat(23, "horizontal_accuracy")
    accuracy?.let { require(it >= 0f) { "horizontal_accuracy 不得为负数" } }

    return SensorFrame(
        timestampSeconds = timestamp,
        gyroscope = Vec3(
            float(1, "gyro_x"),
            float(2, "gyro_y"),
            float(3, "gyro_z"),
        ),
        accelerometer = Vec3(
            float(4, "accel_x"),
            float(5, "accel_y"),
            float(6, "accel_z"),
        ),
        orientation = orientation,
        position = Vec3(
            float(11, "position_x"),
            float(12, "position_y"),
            float(13, "position_z"),
        ),
        velocity = Vec3(
            float(14, "velocity_x"),
            float(15, "velocity_y"),
            float(16, "velocity_z"),
        ),
        validImu = mask(17, "valid_imu"),
        validOrientation = mask(18, "valid_orientation"),
        validPosition = mask(19, "valid_position"),
        latitude = latitude,
        longitude = longitude,
        altitude = optionalDouble(22, "altitude"),
        horizontalAccuracy = accuracy,
    )
}

private const val CSV_COLUMN_COUNT = 24
private const val MAX_CSV_LINE_CHARS = 16_384
private const val QUATERNION_TOLERANCE = 1e-3
