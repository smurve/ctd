import org.nd4s.Implicits._
import org.smurve.cifar10._
import org.smurve.iter.SplitBasedCIFAR10BatchIterator

//val ctx = new DataContext("./input/cifar10/")
//val (img, lbl) = ctx.readSplit("data_batch_1.bin", 1)
val testIter = new SplitBasedCIFAR10BatchIterator("input/cifar10", Array("test_batch.bin"), 1)
val next = testIter.next
val nda = next.getFeatureMatrix
val img = asImage(nda)
img.output(new java.io.File("img.png"))