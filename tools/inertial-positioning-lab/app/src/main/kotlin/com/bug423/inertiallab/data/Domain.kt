package com.bug423.inertiallab.data

import kotlinx.serialization.Serializable

@Serializable
data class Vec3(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f,
) {
    fun toList(): List<Float> = listOf(x, y, z)
}

@Serializable
data class QuaternionWxyz(
    val w: Float = 1f,
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f,
)

data class SensorFrame(
    val timestampSeconds: Double,
    val gyroscope: Vec3,
    val accelerometer: Vec3,
    val orientation: QuaternionWxyz,
    val position: Vec3,
    val velocity: Vec3,
    val latitude: Double?,
    val longitude: Double?,
    val altitude: Double?,
    val horizontalAccuracy: Float?,
    val validImu: Boolean,
    val validOrientation: Boolean,
    val validPosition: Boolean,
)

@Serializable
data class CaptureMetadata(
    val archive_format: String = "inertial-lab/1",
    val schema_version: String = "0.1",
    val dataset: String = "InertialLab",
    val sequence_id: String,
    val display_name: String,
    val world_frame: String = "gravity_aligned_local_enu",
    val timestamp_type: String = "relative",
    val orientation_convention: String = "body_to_world_wxyz",
    val accelerometer_type: String = "specific_force",
    val position_source: String,
    val orientation_source: String = "Android rotation vector",
    val subject_id: String = "anonymous",
    val device_id: String,
    val source_license: String = "user_recorded",
    val sample_rate_hz: Int,
    val started_at_utc: String,
    val finished_at_utc: String? = null,
    val sample_count: Long = 0,
    val duration_seconds: Double = 0.0,
    val position_note: String = "The declared position source is a reference trajectory, not precision ground truth.",
)

@Serializable
data class SessionSummary(
    val sequenceId: String,
    val name: String,
    val startedAtUtc: String,
    val sampleRateHz: Int,
    val samples: Long,
    val durationSeconds: Double,
    val hasReference: Boolean,
    val positionSource: String,
    val worldFrame: String,
    val sizeBytes: Long,
)

data class CaptureState(
    val isRecording: Boolean = false,
    val elapsedSeconds: Double = 0.0,
    val samples: Long = 0,
    val sampleRateHz: Int = 200,
    val targetSampleRateHz: Int = 200,
    val hardwareMaximumRateHz: Int = 200,
    val requiresResampling: Boolean = false,
    val imuAvailable: Boolean = false,
    val orientationAvailable: Boolean = false,
    val referenceAvailable: Boolean = false,
    val accelerometer: Vec3 = Vec3(),
    val gyroscope: Vec3 = Vec3(),
    val trajectory: List<Vec3> = emptyList(),
    val trajectoryDistanceMeters: Float = 0f,
    val referenceStatus: String? = null,
    val activeSequenceId: String? = null,
    val message: String? = null,
)

const val ARCORE_WORLD_FRAME = "arcore_local_x_right_y_forward_z_up"
const val ARCORE_POSITION_SOURCE = "ARCore visual-inertial odometry"
const val ARCORE_ORIENTATION_SOURCE = "ARCore Android sensor pose"
val SUPPORTED_WORLD_FRAMES = setOf("gravity_aligned_local_enu", ARCORE_WORLD_FRAME)

val CANONICAL_CSV_HEADER = listOf(
    "timestamp",
    "gyro_x", "gyro_y", "gyro_z",
    "accel_x", "accel_y", "accel_z",
    "orientation_w", "orientation_x", "orientation_y", "orientation_z",
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "valid_imu", "valid_orientation", "valid_position",
    "latitude", "longitude", "altitude", "horizontal_accuracy",
).joinToString(",")
