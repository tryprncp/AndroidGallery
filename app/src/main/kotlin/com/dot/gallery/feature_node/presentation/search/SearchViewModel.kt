/*
 * SPDX-FileCopyrightText: 2023 IacobIacob01
 * SPDX-License-Identifier: Apache-2.0
 */

package com.dot.gallery.feature_node.presentation.search

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dot.gallery.core.MediaState
import com.dot.gallery.core.Resource
import com.dot.gallery.feature_node.domain.model.Media
import com.dot.gallery.feature_node.domain.model.MediaItem
import com.dot.gallery.feature_node.domain.use_case.MediaUseCases
import com.dot.gallery.feature_node.presentation.util.getDate
import com.dot.gallery.feature_node.presentation.util.getMonth
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.util.ArrayList
import javax.inject.Inject

@HiltViewModel
class SearchViewModel @Inject constructor(
    private val mediaUseCases: MediaUseCases,
    @ApplicationContext private val context: Context
) : ViewModel() {

    var lastQuery = mutableStateOf("")
        private set

    private val _mediaState = MutableStateFlow(MediaState())
    val mediaState = _mediaState.asStateFlow()

    val selectionState = mutableStateOf(false)
    val selectedMedia = mutableStateListOf<Media>()

    // Load YOLOv5 model
    private val yoloModel by lazy {
        loadYoloModel()
    }

    private fun String.assetFilePath(context: Context): String {
        val file = File(context.filesDir, this)
        try {
            val inputStream = context.assets.open(this)
            inputStream.use { inputStream ->
                val outputStream = FileOutputStream(file)
                outputStream.use { outputStream ->
                    val buffer = ByteArray(4 * 1024) // Buffer size
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
        } catch (e: IOException) {
            Log.e("assetFilePath", "Error copying asset to file system", e)
            throw RuntimeException("Error copying asset to file system", e)
        }

        return file.absolutePath
    }

    private fun loadYoloModel(): Module {
        try {
            val model = LiteModuleLoader.load("yolov5nbest.torchscript.ptl".assetFilePath(context))
            PrePostProcessor.mClasses = loadClasses().toTypedArray()
            return model
        } catch (e: Exception) {
            Log.w("SearchViewModel", "Error loading YOLOv5 model:", e)
            throw IllegalStateException("Failed to load YOLOv5 model", e)
        }
    }

    private fun loadClasses(): List<String> {
        try {
            // Assuming classes.txt is a plain text file with one class per line
            val inputStream = context.assets.open("classes.txt")
            val bufferedReader = BufferedReader(InputStreamReader(inputStream))
            val classes = mutableListOf<String>()
            var line: String?
            while (bufferedReader.readLine().also { line = it } != null) {
                classes.add(line.orEmpty())
            }
            bufferedReader.close()
            return classes
        } catch (e: IOException) {
            Log.e("SearchViewModel", "Error loading classes.txt:", e)
            throw IllegalStateException("Failed to load classes.txt", e)
        }
    }

    init {
        queryMedia()
    }

    private fun List<Media>.parseQuery(query: String): List<Media> {
        return try {
            if (query.isEmpty()) {
                return emptyList()
            }

            // Load classes
            val classes = loadClasses()

            // Check if the query matches any of the classes
            val queryClassIndex = getIndexOfLabel(query, classes)
            if (queryClassIndex == -1) {
                // If query doesn't match any class, return empty list
                return emptyList()
            }

            // Use YOLOv5 model for object detection
            val matchingMedia = ArrayList<Media>()
            for (media in this@parseQuery) {
                if (!media.isImage && !media.isVideo) continue  // Skip non-image/video media

                val bitmap = loadBitmap(media.path) // Load the bitmap for the current media
                val results = detectObjects(bitmap)

                // Check for matching objects based on label and confidence threshold
                val matchingObjects = results.filter {
                    it.classIndex == getIndexOfLabel(query) && it.score!! > CONFIDENCETHRESHOLD
                }
                if (matchingObjects.isNotEmpty()) {
                    matchingMedia.add(media)
                }
            }
            matchingMedia
        } catch (e: Exception) {
            Log.e("SearchViewModel", "Error parsing query:", e)
            emptyList()
        }
    }

    private fun getIndexOfLabel(label: String, classes: List<String>): Int {
        // Check if the label matches any of the classes
        return classes.indexOf(label)
    }
    
    private fun loadBitmap(path: String): Bitmap {
        val options = BitmapFactory.Options()
        options.inPreferredConfig = Bitmap.Config.ARGB_8888  // Ensure ARGB_8888 format
        return BitmapFactory.decodeFile(path, options)
    }

    private fun detectObjects(bitmap: Bitmap): List<Result> {
        try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB)
            val outputTuple = yoloModel.forward(IValue.from(inputTensor)).toTuple()
            val outputTensor = outputTuple[0].toTensor()
            val outputs = outputTensor.dataAsFloatArray
            val imgScaleX = bitmap.width.toFloat() / PrePostProcessor.mInputWidth
            val imgScaleY = bitmap.height.toFloat() / PrePostProcessor.mInputHeight
            val ivScaleX = if (bitmap.width.toFloat() > bitmap.height.toFloat()) bitmap.width.toFloat() / bitmap.width.toFloat() else bitmap.height.toFloat() / bitmap.height.toFloat()
            val ivScaleY = if (bitmap.height.toFloat() > bitmap.width.toFloat()) bitmap.height.toFloat() / bitmap.height.toFloat() else bitmap.width.toFloat() / bitmap.width.toFloat()
            val startX = (bitmap.width.toFloat() - ivScaleX * bitmap.width.toFloat()) / 2
            val startY = (bitmap.height.toFloat() - ivScaleY * bitmap.height.toFloat()) / 2

            return PrePostProcessor.outputsToNMSPredictions(outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, startX, startY)
        } catch (e: Exception) {
            Log.e("SearchViewModel", "Error detecting objects:", e)
            return emptyList()
        }
    }

    fun clearQuery() = queryMedia("")

    fun queryMedia(query: String = "") {
        viewModelScope.launch(Dispatchers.IO) {
            withContext(Dispatchers.Main) {
                lastQuery.value = query
            }
            if (query.isEmpty()) {
                _mediaState.tryEmit(MediaState(isLoading = false))
                return@launch
            }
            mediaUseCases.getMediaUseCase().flowOn(Dispatchers.IO).collectLatest { result ->
                val mappedData = ArrayList<MediaItem>()
                val monthHeaderList = ArrayList<String>()
                val data = result.data ?: emptyList()
                if (data == mediaState.value.media) return@collectLatest
                val error = if (result is Resource.Error) result.message
                    ?: "An error occurred" else ""
                if (data.isEmpty()) {
                    _mediaState.tryEmit(MediaState(isLoading = false))
                    return@collectLatest
                }
                _mediaState.tryEmit(MediaState())
                val parsedData = data.parseQuery(query)
                parsedData.groupBy {
                    it.timestamp.getDate(
                        stringToday = "Today"
                        /** Localized in composition */
                        ,
                        stringYesterday = "Yesterday"
                        /** Localized in composition */
                    )
                }.forEach { (date, data) ->
                    val month = getMonth(date)
                    if (month.isNotEmpty() && !monthHeaderList.contains(month)) {
                        monthHeaderList.add(month)
                    }
                    mappedData.add(MediaItem.Header("header_$date", date, data))
                    mappedData.addAll(data.map {
                        MediaItem.MediaViewItem(
                            "media_${it.id}_${it.label}",
                            it
                        )
                    })
                }
                _mediaState.tryEmit(
                    MediaState(
                        error = error,
                        media = parsedData,
                        mappedMedia = mappedData,
                        isLoading = false
                    )
                )
            }
        }
    }

    private fun getIndexOfLabel(label: String): Int {
        val index = PrePostProcessor.mClasses!!.indexOf(label)
        return if (index != -1) index else 0 // Default to the first class if not found
    }

    companion object {
        // Confidence threshold for considering a detected object a match (optional)
        private const val CONFIDENCETHRESHOLD = 0.1f  // Adjust based on your requirements
    }
}
