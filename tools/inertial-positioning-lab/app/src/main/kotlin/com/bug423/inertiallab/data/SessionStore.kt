package com.bug423.inertiallab.data

import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.DocumentsContract
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlin.math.abs

class SessionStore(private val context: Context) {
    private val json = Json { prettyPrint = true; ignoreUnknownKeys = true }
    private val sessionsRoot = File(context.filesDir, "sessions").apply { mkdirs() }

    inner class ActiveSession internal constructor(
        val directory: File,
        val initialMetadata: CaptureMetadata,
        private val writer: BufferedWriter,
    ) {
        private var sampleCount = 0L
        private var lastTimestamp = 0.0
        private var referenceSeen = false

        @Synchronized
        fun append(frame: SensorFrame) {
            writer.append(frame.toCsv()).append('\n')
            sampleCount += 1
            lastTimestamp = frame.timestampSeconds
            referenceSeen = referenceSeen || frame.validPosition
            if (sampleCount % 100L == 0L) writer.flush()
        }

        @Synchronized
        fun finish(): SessionSummary {
            writer.flush()
            writer.close()
            val metadata = initialMetadata.copy(
                finished_at_utc = Instant.now().toString(),
                sample_count = sampleCount,
                duration_seconds = lastTimestamp,
                position_source = initialMetadata.position_source.takeIf { referenceSeen } ?: "unavailable",
            )
            File(directory, MANIFEST).writeText(json.encodeToString(metadata), Charsets.UTF_8)
            return metadata.toSummary(directory)
        }

        @Synchronized
        fun abort() {
            runCatching { writer.close() }
            directory.deleteRecursively()
        }
    }

    fun begin(name: String, sampleRateHz: Int, useArCoreReference: Boolean): ActiveSession {
        val now = Instant.now()
        val baseId = "${FILE_TIME.format(now)}_${slug(name)}"
        val (id, directory) = createUniqueDirectory(baseId)
        try {
            require(File(directory, "data").mkdir()) { "无法创建数据目录" }
            val metadata = CaptureMetadata(
                sequence_id = id,
                display_name = name.trim().ifBlank { "未命名采集" },
                device_id = listOf(Build.MANUFACTURER, Build.MODEL).joinToString(" ").trim(),
                sample_rate_hz = sampleRateHz,
                started_at_utc = now.toString(),
                world_frame = if (useArCoreReference) ARCORE_WORLD_FRAME else "gravity_aligned_local_enu",
                position_source = if (useArCoreReference) ARCORE_POSITION_SOURCE else "unavailable",
                orientation_source = if (useArCoreReference) ARCORE_ORIENTATION_SOURCE else "Android rotation vector",
                position_note = if (useArCoreReference) {
                    "Phone-side ground truth follows the original IMUNet software and uses ARCore VIO; it is not an independent Vicon/RTK measurement."
                } else {
                    "No position reference was requested."
                },
            )
            File(directory, MANIFEST).writeText(json.encodeToString(metadata), Charsets.UTF_8)
            val writer = BufferedWriter(OutputStreamWriter(FileOutputStream(File(directory, DATA), false), Charsets.UTF_8))
            writer.append(CANONICAL_CSV_HEADER).append('\n')
            writer.flush()
            return ActiveSession(directory, metadata, writer)
        } catch (error: Throwable) {
            directory.deleteRecursively()
            throw error
        }
    }

    suspend fun list(): List<SessionSummary> = withContext(Dispatchers.IO) {
        sessionsRoot.listFiles()
            .orEmpty()
            .filter { it.isDirectory && File(it, MANIFEST).isFile && File(it, DATA).isFile }
            .mapNotNull { directory ->
                runCatching {
                    json.decodeFromString<CaptureMetadata>(File(directory, MANIFEST).readText(Charsets.UTF_8))
                        .toSummary(directory)
                }.getOrNull()
            }
            .sortedByDescending(SessionSummary::startedAtUtc)
    }

    suspend fun frames(sequenceId: String): List<SensorFrame> = withContext(Dispatchers.IO) {
        val directory = safeSession(sequenceId)
        readFrames(File(directory, DATA))
    }

    suspend fun metadata(sequenceId: String): CaptureMetadata = withContext(Dispatchers.IO) {
        val directory = safeSession(sequenceId)
        json.decodeFromString<CaptureMetadata>(File(directory, MANIFEST).readText(Charsets.UTF_8)).also(::validateMetadata)
    }

    suspend fun export(sequenceId: String, destination: Uri) = withContext(Dispatchers.IO) {
        val directory = safeSession(sequenceId)
        context.contentResolver.openOutputStream(destination, "w")?.use { output ->
            ZipOutputStream(output.buffered()).use { zip ->
                addFile(zip, File(directory, MANIFEST), MANIFEST)
                addFile(zip, File(directory, DATA), DATA)
                zip.putNextEntry(ZipEntry("README.txt"))
                zip.write(ARCHIVE_NOTE.toByteArray(Charsets.UTF_8))
                zip.closeEntry()
            }
        } ?: error("无法打开导出位置")
    }

    suspend fun exportToDirectory(sequenceId: String, treeUri: Uri): Uri = withContext(Dispatchers.IO) {
        val parent = DocumentsContract.buildDocumentUriUsingTree(
            treeUri,
            DocumentsContract.getTreeDocumentId(treeUri),
        )
        val destination = DocumentsContract.createDocument(
            context.contentResolver,
            parent,
            "application/zip",
            "${slug(sequenceId)}.iplab",
        ) ?: error("无法在所选目录创建数据包")
        export(sequenceId, destination)
        destination
    }

    suspend fun importArchive(source: Uri): SessionSummary = withContext(Dispatchers.IO) {
        val staging = File(sessionsRoot, ".import-${System.nanoTime()}").apply { mkdirs() }
        try {
            var expandedBytes = 0L
            val seenEntries = mutableSetOf<String>()
            context.contentResolver.openInputStream(source)?.use { input ->
                ZipInputStream(input.buffered()).use { zip ->
                    var entry = zip.nextEntry
                    while (entry != null) {
                        if (entry.isDirectory) {
                            require(entry.name == "data/") { "归档包含未知目录：${entry.name}" }
                        } else {
                            require(entry.name in ALLOWED_ARCHIVE_FILES) { "归档包含未知文件：${entry.name}" }
                            require(seenEntries.add(entry.name)) { "归档包含重复文件：${entry.name}" }
                            val target = entry.name.takeIf { it in REQUIRED_ARCHIVE_FILES }?.let {
                                File(staging, it).also { file ->
                                    val parent = requireNotNull(file.parentFile)
                                    require(parent.isDirectory || parent.mkdirs()) {
                                        "无法创建归档暂存目录"
                                    }
                                }
                            }
                            target?.outputStream()?.use { output ->
                                expandedBytes = copyBounded(zip, output, expandedBytes)
                            } ?: run {
                                expandedBytes = copyBounded(zip, null, expandedBytes)
                            }
                        }
                        zip.closeEntry()
                        entry = zip.nextEntry
                    }
                }
            } ?: error("无法读取数据集")
            require(seenEntries.containsAll(REQUIRED_ARCHIVE_FILES)) {
                "归档缺少 manifest.json 或 data/sequence.csv"
            }
            val metadataFile = File(staging, MANIFEST)
            val dataFile = File(staging, DATA)
            require(metadataFile.length() <= MAX_MANIFEST_BYTES) { "manifest.json 超过 1 MiB" }
            val metadataText = metadataFile.readText(Charsets.UTF_8)
            val metadataObject = json.parseToJsonElement(metadataText) as? JsonObject
                ?: throw IllegalArgumentException("manifest.json 根节点必须是对象")
            val missingFields = REQUIRED_METADATA_FIELDS.filterNot(metadataObject::containsKey)
            require(missingFields.isEmpty()) { "manifest.json 缺少字段：${missingFields.joinToString()}" }
            val importedMetadata = json.decodeFromString<CaptureMetadata>(metadataText)
            validateMetadata(importedMetadata)
            val stats = dataFile.bufferedReader(Charsets.UTF_8).use(::readCanonicalCsv)
            if (importedMetadata.sample_count != 0L) {
                require(importedMetadata.sample_count == stats.sampleCount) { "manifest sample_count 与 CSV 不一致" }
            }
            if (importedMetadata.duration_seconds != 0.0) {
                require(abs(importedMetadata.duration_seconds - stats.durationSeconds) <= DURATION_TOLERANCE_SECONDS) {
                    "manifest duration_seconds 与 CSV 不一致"
                }
            }
            require(!stats.hasReference || importedMetadata.position_source != "unavailable") {
                "CSV 含有效位置，但 manifest position_source 为 unavailable"
            }
            val metadata = importedMetadata.copy(
                sample_count = stats.sampleCount,
                duration_seconds = stats.durationSeconds,
                position_source = importedMetadata.position_source.takeIf { stats.hasReference } ?: "unavailable",
            )
            metadataFile.writeText(json.encodeToString(metadata), Charsets.UTF_8)
            val destination = File(sessionsRoot, slug(metadata.sequence_id))
            require(!destination.exists()) { "序列 ${metadata.sequence_id} 已存在" }
            require(staging.renameTo(destination)) { "无法保存导入的数据集" }
            metadata.toSummary(destination)
        } catch (error: Throwable) {
            staging.deleteRecursively()
            throw error
        }
    }

    private fun safeSession(sequenceId: String): File {
        val directory = File(sessionsRoot, slug(sequenceId))
        require(directory.parentFile == sessionsRoot && directory.isDirectory) { "序列不存在" }
        return directory
    }

    private fun addFile(zip: ZipOutputStream, file: File, name: String) {
        zip.putNextEntry(ZipEntry(name))
        file.inputStream().use { it.copyTo(zip) }
        zip.closeEntry()
    }

    private fun readFrames(file: File): List<SensorFrame> {
        val frames = mutableListOf<SensorFrame>()
        file.bufferedReader(Charsets.UTF_8).use { reader -> readCanonicalCsv(reader, frames::add) }
        return frames
    }

    private fun validateMetadata(metadata: CaptureMetadata) {
        require(metadata.archive_format == "inertial-lab/1") { "仅支持 inertial-lab/1 归档" }
        require(metadata.schema_version == "0.1") { "仅支持 canonical schema 0.1" }
        require(metadata.sequence_id.isNotBlank()) { "sequence_id 不能为空" }
        require(metadata.display_name.isNotBlank()) { "display_name 不能为空" }
        require(metadata.dataset.isNotBlank()) { "dataset 不能为空" }
        require(metadata.world_frame in SUPPORTED_WORLD_FRAMES) {
            "不支持的 world_frame：${metadata.world_frame}"
        }
        require(metadata.timestamp_type == "relative") { "timestamp_type 必须为 relative" }
        require(metadata.orientation_convention == "body_to_world_wxyz") {
            "orientation_convention 必须为 body_to_world_wxyz"
        }
        require(metadata.accelerometer_type == "specific_force") { "accelerometer_type 必须为 specific_force" }
        require(metadata.position_source.isNotBlank()) { "position_source 不能为空" }
        require(metadata.orientation_source.isNotBlank()) { "orientation_source 不能为空" }
        require(metadata.subject_id.isNotBlank()) { "subject_id 不能为空" }
        require(metadata.device_id.isNotBlank()) { "device_id 不能为空" }
        require(metadata.source_license.isNotBlank()) { "source_license 不能为空" }
        require(metadata.sample_rate_hz in 10..1000) { "sample_rate_hz 必须在 10..1000" }
        val started = runCatching { Instant.parse(metadata.started_at_utc) }.getOrElse {
            throw IllegalArgumentException("started_at_utc 必须为 UTC ISO-8601 时间", it)
        }
        metadata.finished_at_utc?.let {
            val finished = runCatching { Instant.parse(it) }.getOrElse { error ->
                throw IllegalArgumentException("finished_at_utc 必须为 UTC ISO-8601 时间", error)
            }
            require(!finished.isBefore(started)) { "finished_at_utc 不得早于 started_at_utc" }
        }
        require(metadata.sample_count >= 0) { "sample_count 不得为负数" }
        require(metadata.duration_seconds.isFinite() && metadata.duration_seconds >= 0.0) {
            "duration_seconds 必须为非负有限数值"
        }
    }

    private fun createUniqueDirectory(baseId: String): Pair<String, File> {
        for (suffix in 0..MAX_SESSION_SUFFIX) {
            val id = if (suffix == 0) baseId else "$baseId-$suffix"
            val directory = File(sessionsRoot, id)
            if (directory.mkdir()) return id to directory
            require(directory.exists()) { "无法创建采集目录" }
        }
        error("同名采集过多，请稍后重试")
    }

    private fun copyBounded(
        input: ZipInputStream,
        output: OutputStream?,
        alreadyExpanded: Long,
    ): Long {
        var expanded = alreadyExpanded
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            expanded += count
            require(expanded <= MAX_ARCHIVE_BYTES) { "归档解压后超过 512 MiB" }
            output?.write(buffer, 0, count)
        }
        return expanded
    }

    private fun CaptureMetadata.toSummary(directory: File) = SessionSummary(
        sequenceId = sequence_id,
        name = display_name,
        startedAtUtc = started_at_utc,
        sampleRateHz = sample_rate_hz,
        samples = sample_count,
        durationSeconds = duration_seconds,
        hasReference = position_source != "unavailable",
        positionSource = position_source,
        worldFrame = world_frame,
        sizeBytes = directory.walkTopDown().filter(File::isFile).sumOf(File::length),
    )

    companion object {
        private const val MANIFEST = "manifest.json"
        private const val DATA = "data/sequence.csv"
        private const val README = "README.txt"
        private val REQUIRED_ARCHIVE_FILES = setOf(MANIFEST, DATA)
        private val ALLOWED_ARCHIVE_FILES = REQUIRED_ARCHIVE_FILES + README
        private val REQUIRED_METADATA_FIELDS = setOf(
            "archive_format", "schema_version", "dataset", "sequence_id", "display_name",
            "world_frame", "timestamp_type", "orientation_convention", "accelerometer_type",
            "position_source", "orientation_source", "subject_id", "device_id", "source_license",
            "sample_rate_hz", "started_at_utc",
        )
        private val FILE_TIME = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").withZone(ZoneOffset.UTC)
        private const val ARCHIVE_NOTE = "Inertial Lab canonical interchange archive v1. Convert to benchmark HDF5 with tools/convert_dataset.py.\n"
        private const val MAX_ARCHIVE_BYTES = 512L * 1024 * 1024
        private const val MAX_MANIFEST_BYTES = 1024L * 1024
        private const val MAX_SESSION_SUFFIX = 9999
        private const val DURATION_TOLERANCE_SECONDS = 1e-6

        private fun slug(value: String): String = value.lowercase(Locale.ROOT)
            .replace(Regex("[^a-z0-9._-]+"), "-")
            .trim('-')
            .take(80)
            .ifBlank { "sequence" }
    }
}

private fun SensorFrame.toCsv(): String {
    fun Float.text() = String.format(Locale.US, "%.9g", this)
    fun Double.text() = String.format(Locale.US, "%.12g", this)
    fun Boolean.text() = if (this) "1" else "0"
    return buildList {
        add(timestampSeconds.text())
        addAll(gyroscope.toList().map(Float::text))
        addAll(accelerometer.toList().map(Float::text))
        addAll(listOf(orientation.w, orientation.x, orientation.y, orientation.z).map(Float::text))
        addAll(position.toList().map(Float::text))
        addAll(velocity.toList().map(Float::text))
        add(validImu.text())
        add(validOrientation.text())
        add(validPosition.text())
        add(latitude?.text().orEmpty())
        add(longitude?.text().orEmpty())
        add(altitude?.text().orEmpty())
        add(horizontalAccuracy?.text().orEmpty())
    }.joinToString(",")
}
