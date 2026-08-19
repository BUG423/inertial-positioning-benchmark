package com.bug423.inertiallab.model

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.util.zip.ZipInputStream
import kotlin.coroutines.coroutineContext

class ModelStore(private val context: Context) {
    private val json = Json { ignoreUnknownKeys = false; prettyPrint = true }
    private val root = File(context.filesDir, "models").apply { mkdirs() }

    suspend fun list(): List<InstalledModel> = withContext(Dispatchers.IO) {
        root.listFiles().orEmpty().mapNotNull(::readInstalled).sortedBy { it.manifest.name }
    }

    suspend fun import(source: Uri): InstalledModel = withContext(Dispatchers.IO) {
        val staging = File(root, ".import-${System.nanoTime()}").apply { mkdirs() }
        try {
            var expandedBytes = 0L
            val seenEntries = mutableSetOf<String>()
            context.contentResolver.openInputStream(source)?.use { input ->
                ZipInputStream(input.buffered()).use { zip ->
                    var entry = zip.nextEntry
                    while (entry != null) {
                        require(!entry.isDirectory) { "模型包不得包含目录" }
                        require(entry.name in setOf(MANIFEST_FILE, MODEL_FILE)) { "模型包含未知文件：${entry.name}" }
                        require(seenEntries.add(entry.name)) { "模型包包含重复文件：${entry.name}" }
                        val target = File(staging, entry.name)
                        target.outputStream().use { output ->
                            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                            while (true) {
                                val count = zip.read(buffer)
                                if (count < 0) break
                                expandedBytes += count
                                require(expandedBytes <= MAX_PACKAGE_BYTES) { "模型包解压后超过 128 MiB" }
                                output.write(buffer, 0, count)
                            }
                        }
                        zip.closeEntry()
                        entry = zip.nextEntry
                    }
                }
            } ?: error("无法读取模型包")
            require(seenEntries == setOf(MANIFEST_FILE, MODEL_FILE)) {
                "模型包必须且只能包含 manifest.json 和 model.tflite"
            }
            val manifestFile = File(staging, MANIFEST_FILE)
            val modelFile = File(staging, MODEL_FILE)
            require(manifestFile.isFile && modelFile.isFile) { "模型包必须包含 manifest.json 和 model.tflite" }
            val manifest = json.decodeFromString<ModelManifest>(manifestFile.readText())
            val errors = manifest.validate()
            require(errors.isEmpty()) { errors.joinToString("；") }
            require(modelFile.length() >= 8L && modelFile.inputStream().use { stream ->
                val header = ByteArray(8)
                stream.read(header) == 8 && header.copyOfRange(4, 8).decodeToString() == "TFL3"
            }) { "model.tflite 不是有效的 TFLite FlatBuffer" }
            validateTensors(modelFile, manifest)
            val directory = File(root, "${manifest.id}-${safe(manifest.version)}")
            require(!directory.exists()) { "模型 ${manifest.id} ${manifest.version} 已安装" }
            require(staging.renameTo(directory)) { "无法保存模型" }
            readInstalled(directory) ?: error("模型保存后校验失败")
        } catch (error: Throwable) {
            staging.deleteRecursively()
            throw error
        }
    }

    /**
     * Imports every selected package without rolling back packages that were already accepted.
     * A malformed or duplicate package is reported and does not prevent the remaining packages
     * from being validated. The single-package [import] API remains the source of truth.
     */
    suspend fun importAll(
        sources: Collection<Uri>,
        onProgress: (completed: Int, total: Int) -> Unit = { _, _ -> },
    ): ModelImportBatchResult = withContext(Dispatchers.IO) {
        val uniqueSources = sources.distinctBy(Uri::toString)
        val imported = ArrayList<InstalledModel>(uniqueSources.size)
        val failures = ArrayList<ModelImportFailure>()
        uniqueSources.forEachIndexed { index, source ->
            coroutineContext.ensureActive()
            try {
                imported += import(source)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                failures += ModelImportFailure(
                    source = source.lastPathSegment?.takeIf(String::isNotBlank) ?: "model-${index + 1}",
                    reason = error.message ?: error::class.java.simpleName,
                )
            }
            onProgress(index + 1, uniqueSources.size)
        }
        ModelImportBatchResult(imported, failures)
    }

    fun modelFile(model: InstalledModel): File {
        val directory = File(root, model.directoryName)
        require(directory.parentFile == root && directory.isDirectory) { "模型不存在" }
        return File(directory, MODEL_FILE)
    }

    private fun readInstalled(directory: File): InstalledModel? = runCatching {
        val manifestFile = File(directory, MANIFEST_FILE)
        val modelFile = File(directory, MODEL_FILE)
        require(manifestFile.isFile && modelFile.isFile && modelFile.length() >= 8L)
        val manifest = json.decodeFromString<ModelManifest>(manifestFile.readText())
        require(manifest.validate().isEmpty())
        require(modelFile.inputStream().use { stream ->
            val header = ByteArray(8)
            stream.read(header) == 8 && header.copyOfRange(4, 8).decodeToString() == "TFL3"
        })
        InstalledModel(manifest, modelFile.length(), directory.name)
    }.getOrNull()

    private fun validateTensors(file: File, manifest: ModelManifest) {
        val mapped = file.inputStream().channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
        Interpreter(mapped, Interpreter.Options().setNumThreads(1)).use { interpreter ->
            require(interpreter.inputTensorCount == 1) { "模型必须只有一个输入张量" }
            require(interpreter.outputTensorCount == 1) { "模型必须只有一个输出张量" }
            val input = interpreter.getInputTensor(0)
            val output = interpreter.getOutputTensor(0)
            require(input.dataType() == DataType.FLOAT32) { "输入张量必须为 float32" }
            require(output.dataType() == DataType.FLOAT32) { "输出张量必须为 float32" }
            require(input.shape().toList() == manifest.input.shape) {
                "模型输入 ${input.shape().toList()} 与清单 ${manifest.input.shape} 不一致"
            }
            require(output.shape().toList() == manifest.output.shape) {
                "模型输出 ${output.shape().toList()} 与清单 ${manifest.output.shape} 不一致"
            }
        }
    }

    private fun safe(value: String) = value.replace(Regex("[^A-Za-z0-9._-]"), "-").take(40)

    companion object {
        private const val MANIFEST_FILE = "manifest.json"
        private const val MODEL_FILE = "model.tflite"
        private const val MAX_PACKAGE_BYTES = 128L * 1024 * 1024
    }
}
