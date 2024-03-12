package com.dot.gallery.feature_node.presentation.search

import android.graphics.Rect
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.max

class Result(
    var classIndex: Int,
    var score: Float?,
    var rect: Rect
)

object PrePostProcessor {

    // for yolov5 model, no need to apply MEAN and STD
    val NO_MEAN_RGB = floatArrayOf(0.0f, 0.0f, 0.0f)
    val NO_STD_RGB = floatArrayOf(1.0f, 1.0f, 1.0f)

    // model input image size
    var mInputWidth = 640
    var mInputHeight = 640

    // model output is of size 25200*(num_of_class+5)
    private const val mOutputRow = 25200 // as decided by the YOLOv5 model for input image of size 640*640
    private const val mOutputColumn = 85 // left, top, right, bottom, score, and 80 class probability
    private const val mThreshold = 0.30f // score above which a detection is generated
    private const val mNmsLimit = 15

    var mClasses: Array<String>? = null

    // The two methods nonMaxSuppression and IOU below are ported from
    // https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift

    /**
     * Removes bounding boxes that overlap too much with other boxes that have a higher score.
     *
     * @param boxes an array of bounding boxes and their scores
     * @param limit the maximum number of boxes that will be selected
     * @param threshold used to decide whether boxes overlap too much
     * @return ArrayList of selected boxes
     */
    fun nonMaxSuppression(boxes: ArrayList<Result>, limit: Int, threshold: Float): ArrayList<Result> {
        // Do an argsort on the confidence scores, from high to low.
        boxes.sortByDescending { it.score }

        val selected = ArrayList<Result>()
        val active = BooleanArray(boxes.size) { true }
        var numActive = active.size

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        var done = false
        for (i in boxes.indices) {
            if (active[i]) {
                val boxA = boxes[i]
                selected.add(boxA)
                if (selected.size >= limit) break

                for (j in i + 1 until boxes.size) {
                    if (active[j]) {
                        val boxB = boxes[j]
                        if (IOU(boxA.rect, boxB.rect) > threshold) {
                            active[j] = false
                            numActive--
                            if (numActive <= 0) {
                                done = true
                                break
                            }
                        }
                    }
                }
            }
            if (done) break
        }
        return selected
    }

    /**
     * Computes intersection-over-union overlap between two bounding boxes.
     */
    private fun IOU(a: Rect, b: Rect): Float {
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        if (areaA <= 0.0) return 0.0f

        val areaB = (b.right - b.left) * (b.bottom - b.top)
        if (areaB <= 0.0) return 0.0f

        val intersectionMinX = max(a.left, b.left).toFloat()
        val intersectionMinY = max(a.top, b.top).toFloat()
        val intersectionMaxX = max(a.right, b.right).toFloat()
        val intersectionMaxY = max(a.bottom, b.bottom).toFloat()
        val intersectionArea = max(intersectionMaxY - intersectionMinY, 0f) *
                max(intersectionMaxX - intersectionMinX, 0f)
        return intersectionArea / (areaA + areaB - intersectionArea)
    }

    /**
     * Convert outputs to non-maximum suppression predictions.
     */
    fun outputsToNMSPredictions(
        outputs: FloatArray,
        imgScaleX: Float,
        imgScaleY: Float,
        ivScaleX: Float,
        ivScaleY: Float,
        startX: Float,
        startY: Float
    ): ArrayList<Result> {
        val results = ArrayList<Result>()
        for (i in 0 until mOutputRow) {
            if (outputs[i * mOutputColumn + 4] > mThreshold) {
                val x = outputs[i * mOutputColumn]
                val y = outputs[i * mOutputColumn + 1]
                val w = outputs[i * mOutputColumn + 2]
                val h = outputs[i * mOutputColumn + 3]

                val left = imgScaleX * (x - w / 2)
                val top = imgScaleY * (y - h / 2)
                val right = imgScaleX * (x + w / 2)
                val bottom = imgScaleY * (y + h / 2)

                var max = outputs[i * mOutputColumn + 5]
                var cls = 0
                for (j in 0 until mOutputColumn - 5) {
                    if (outputs[i * mOutputColumn + 5 + j] > max) {
                        max = outputs[i * mOutputColumn + 5 + j]
                        cls = j
                    }
                }

                val rect = Rect(
                    (startX + ivScaleX * left).toInt(),
                    (startY + top * ivScaleY).toInt(),
                    (startX + ivScaleX * right).toInt(),
                    (startY + ivScaleY * bottom).toInt()
                )
                val result = Result(cls, outputs[i * mOutputColumn + 4], rect)
                results.add(result)
            }
        }
        return nonMaxSuppression(results, mNmsLimit, mThreshold)
    }
}
