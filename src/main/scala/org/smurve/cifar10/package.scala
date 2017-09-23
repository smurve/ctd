package org.smurve

import java.io.InputStream

import com.sksamuel.scrimage.{Image, RGBColor}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4s.Implicits._
import org.smurve.cifar10.DataContext._
import org.smurve.nd4s.vec

package object cifar10 {

  val categories = Array(
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck")


  /**
    * read the next image out of an open stream: The structure is assumed to be 1 + 3 x 32 x 32.
    * 1 byte for the label and 3 x 1024 bytes for the three RGB Layers of the image
    *
    * @param stream an open data input stream
    * @return the image and a label
    */
  def nextImage(stream: InputStream): (Image, Int) = {
    val buffer = new Array[Byte](BUFFER_SIZE_PER_ENTRY)
    val check = stream.read(buffer)
    assert(check == BUFFER_SIZE_PER_ENTRY, s"Failed to read $BUFFER_SIZE_PER_ENTRY bytes. Got $check instead")
    val pixels = for (pos <- 1 to CHANNEL_SIZE) yield {
      val red = buffer(pos)
      val green = buffer(pos + CHANNEL_SIZE)
      val blue = buffer(pos + 2 * CHANNEL_SIZE)
      RGBColor(red & 0xFF, green & 0xFF, blue & 0xFF).toPixel
    }
    (Image(IMG_WIDTH, IMG_HEIGHT, pixels.toArray), buffer(0).toInt)
  }

  /**
    * Creates an image from the INDArray. Assumes that is is centered like ( byte / 256 - 0.5 ). Will rescale like (x+0.5)*256
    */
  def asImage(inda: INDArray): Image = {
    assert(inda.shape() sameElements Array(NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH), s"Strange shape: ${inda.shape().toList}")
    val pixels = for (r <- 0 until IMG_HEIGHT; c <- 0 until IMG_WIDTH ) yield {
      val red = ((inda.getDouble(0, r, c) + 0.5) * 255).toInt
      val green = ((inda.getDouble(1, r, c) + 0.5) * 255).toInt
      val blue = ((inda.getDouble(2, r, c) + 0.5) * 255).toInt
      try {
        RGBColor(red, green, blue).toPixel
      } catch {
        case e: Exception =>
          println(s"($red, $green, $blue) not good")
          throw e
      }
    }
    Image(IMG_WIDTH, IMG_HEIGHT, pixels.toArray)
  }

  /**
    * Creates an image from the INDArray. Assumes that is is centered like ( byte / 256 - 0.5 ). Will rescale like (x+0.5)*256
    */
  def asImage(bytes: Array[Byte], imageNr: Int): Image = {
    val imageOffset = imageNr * BUFFER_SIZE_PER_ENTRY + 1
    val pixels = for (pos <- 0 until CHANNEL_SIZE) yield {
      val red = bytes(imageOffset + pos) & 0xFF
      val green = bytes(imageOffset + CHANNEL_SIZE + pos) & 0xFF
      val blue = bytes(imageOffset + 2 * CHANNEL_SIZE + pos) & 0xFF
      try {
        RGBColor(red, green, blue).toPixel
      } catch {
        case e: Exception =>
          println(s"($red, $green, $blue) not good for color channels")
          throw e
      }
    }
    Image(IMG_WIDTH, IMG_HEIGHT, pixels.toArray)
  }


  def dumpAsImages(bytes: Array[Byte], images: INDArray, labels: INDArray, numSamples: Int): Unit = {

    val total = bytes.length / 3073
    val n_images = math.min(total, numSamples)
    if (n_images > 1)
      println(s"dumping $n_images samples")

    for (_ <- 0 until n_images) {
      val index = (math.random * total).toInt

      val b_img = asImage(bytes, index)
      val b_label = bytes(3073 * index)
      b_img.output(s"target/tmp/${categories(b_label)}-$index-b.png")

      val raw = images(index, ->, ->, ->)
      val n_img = asImage(raw)
      val category = vec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ** labels(index, ->).T
      val n_label = categories(category.getInt(0))
      n_img.output(s"target/tmp/$n_label-$index-n.png")

    }
    println("Done.")
  }


  def dumpAsImages(set: DataSet, sampleSize: Int): Unit = {
    dumpAsImages(set.getFeatures, set.getLabels, sampleSize)
  }

  def dumpAsImages(images: INDArray, labels: INDArray, sampleSize: Int): Unit = {

    val n_images = math.min(images.size(0), sampleSize)

    for (_ <- 0 until n_images) {
      val index = (math.random * images.size(0)).toInt
      val raw = images(index, ->, ->, ->).reshape(3, 32, 32)
      val img = asImage(raw)
      val category = vec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ** labels(index, ->).T
      val label = categories(category.getInt(0))
      img.output(s"target/tmp/$label-$index.png")
    }
  }

}
