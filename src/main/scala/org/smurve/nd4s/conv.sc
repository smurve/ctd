import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._

val m = vec(
  1, 2, 3, 4,
  2, 3, 4, 5,
  4, 3, 2, 1,
  7, 6, 5, 4,

  2, 3, 4, 5,
  1, 2, 3, 4,
  7, 6, 5, -1,
  4, 3, 2, 4,


  5, 2, 8, 4,
  2, 3, 4, 5,
  6, 3, 7, 1,
  7, 6, 5, 4,

  4, 3, 1, 5,
  1, 2, 3, 4,
  3, 6, 2, 4,
  4, 3, 2, 0

).reshape(2, 2, 4, 4)

val height_stride = 2
val width_stride = 2
val depth_stride = 2

m.length
m.ravel.length


type IArr = Array[Array[Int]]

def _outer ( l: IArr, r: IArr ) : IArr = l.flatMap(i=>r.map(i ++ _))

def _iArr(ranges: Range*): IArr = {
  ranges.map(r=>r.toArray.map(Array(_))).toArray.reduce( _outer )
}

def _asString(arr: IArr): String = arr.toList.map(_.toList).toString
  .replace("List", "").replace("(","[").replace(")", "]")


def _reduceWithIndex(source: INDArray, mi: IArr, op: (Double, Double) => Boolean): (Double, Array[Int]) = {
  mi.map(i=>(source(i(0), i(1), i(2), i(3)), i))
    .reduce((a,e)=>if(op(a._1, e._1)) a else e)
}

def _minWithIndex(source: INDArray, mi: IArr): (Double, Array[Int]) = _reduceWithIndex( source, mi, _<_)
def _maxWithIndex(source: INDArray, mi: IArr): (Double, Array[Int]) = _reduceWithIndex( source, mi, _>_)




def domainOf(od: Int, or: Int, oc: Int): IArr =
  _iArr(
    od until od+1,
    0 until depth_stride,
    height_stride * or until height_stride * (or + 1),
    width_stride * oc until width_stride * (oc + 1))



val r02 = _iArr(0 until 2, 1 until 3, 1 until 3)
_asString(r02)

for (Array(i,j,k)<-r02) yield (i,j,k)

val (od, or, oc) = (0, 1, 1)

val domain = domainOf(od, or, oc)
_asString(domain)



val min = _minWithIndex(m, domain)
min._2.toList







