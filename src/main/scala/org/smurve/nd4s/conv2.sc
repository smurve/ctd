
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._


val l1 = "11\n12\n13"
val l2 = "21\n22\n23"
val l3 = "31\n32\n33"

val imgs = List(l1, l2, l3)

imgs.map(_.split("\n").toList).reduce((l, r) => {
  l.zip(r).map(p => p._1 + " | " + p._2)
  //val res1: Seq[String] = l.zip(r).map(p => p._1 + " | " + p._2)
  //res1.reduce((l,r) => l + "\n" + r )
}).reduce((l, r) => l + "\n" + r)



def aligned(imgs: String*) = {
  imgs.map(_.split("\n").toList).reduce((l, r) => {
    l.zip(r).map(p => p._1 + " | " + p._2)
  }).reduce((l, r) => l + "\n" + r)
}

aligned(l1, l2, l3)

val m = vec(1,2,3,4).reshape(2,2)

m.sumNumber().doubleValue()/m.length()
m.stdNumber().doubleValue()

"%5.3f".format(331.587)
