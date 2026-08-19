package com.bug423.inertiallab.data

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import androidx.core.content.edit
import androidx.core.net.toUri

data class CaptureSettings(
    val targetSampleRateHz: Int = 200,
    val effectiveSampleRateHz: Int = 200,
    val hardwareMaximumRateHz: Int = 200,
    val useArCoreGroundTruth: Boolean = true,
    val saveDirectoryUri: String? = null,
    val saveLocationLabel: String = CapturePreferences.DEFAULT_SAVE_LOCATION_LABEL,
) {
    val requiresResampling: Boolean get() = effectiveSampleRateHz != targetSampleRateHz
    val hasCustomSaveDirectory: Boolean get() = saveDirectoryUri != null
}

class CapturePreferences(context: Context) {
    private val appContext = context.applicationContext
    private val preferences = appContext.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    fun load(
        effectiveSampleRateHz: Int,
        hardwareMaximumRateHz: Int,
    ): CaptureSettings {
        val uri = preferences.getString(SAVE_DIRECTORY_URI, null)
        return CaptureSettings(
            effectiveSampleRateHz = effectiveSampleRateHz,
            hardwareMaximumRateHz = hardwareMaximumRateHz,
            useArCoreGroundTruth = preferences.getBoolean(USE_ARCORE, true),
            saveDirectoryUri = uri,
            saveLocationLabel = uri?.let { directoryLabel(it.toUri()) } ?: DEFAULT_SAVE_LOCATION_LABEL,
        )
    }

    fun setUseArCoreGroundTruth(enabled: Boolean) {
        preferences.edit { putBoolean(USE_ARCORE, enabled) }
    }

    fun setSaveDirectory(uri: Uri) {
        preferences.edit { putString(SAVE_DIRECTORY_URI, uri.toString()) }
    }

    fun clearSaveDirectory() {
        preferences.edit { remove(SAVE_DIRECTORY_URI) }
    }

    private fun directoryLabel(treeUri: Uri): String = runCatching {
        val documentUri = DocumentsContract.buildDocumentUriUsingTree(
            treeUri,
            DocumentsContract.getTreeDocumentId(treeUri),
        )
        appContext.contentResolver.query(
            documentUri,
            arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }
    }.getOrNull()?.takeIf(String::isNotBlank)?.let { "自定义目录 · $it" }
        ?: "自定义目录"

    companion object {
        const val DEFAULT_SAVE_LOCATION_LABEL = "应用内部存储（自动保存）"
        private const val PREFERENCES_NAME = "capture-settings"
        private const val SAVE_DIRECTORY_URI = "save-directory-uri"
        private const val USE_ARCORE = "use-arcore-ground-truth"
    }
}
