import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._

val img = vec(
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,

  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
).reshape(2, 14, 14)

val theta = vec(
  3, 0, 0, 0, 0,
  1, 1, 1, 1, 1,
  1, 2, 2, 2, 1,
  1, 2, 2, 2, 1,
  1, 2, 2, 2, 1,
  1, 1, 1, 1, 1,

  3, 0, 0, 0, 0,
  1, 1, 1, 1, 1,
  1, 0, 0, 0, 1,
  1, 0, 1, 0, 1,
  1, 0, 0, 0, 1,
  1, 1, 1, 1, 1,

  3, 0, 0, 0, 0,
  0, 0, 0, 0, 0,
  0, -1, -1, -1, 0,
  0, 1, 1, 1, 0,
  0, -1, -1, -1, 0,
  0, 0, 0, 0, 0
).reshape(3, 6, 5)

val out = Nd4j.zeros(6, 10, 10)

val (depth_o, height_o, width_o) = (6, 10, 10)
val (depth_i, height_i, width_i) = (2, 14, 14)
val (depth_t, height_t, width_t) = (3, 6, 5)


def id_od(od: Int) = od / depth_t
def td_od(od: Int) = od % depth_t
def ir_or_tr(or: Int, tr: Int) = or + tr - 1
def ic_oc_tc(oc: Int, tc: Int) = oc + tc
def idrc(or: Int, od: Int, oc: Int, tr: Int, tc: Int) = (id_od(od), ir_or_tr(or, tr), ic_oc_tc(oc, tc))



def convolve(): Unit =
  for (od <- 0 until depth_o)
    for (or <- 0 until height_o)
      for (oc <- 0 until width_o) {
        out(od, or, oc) = {
          val elems =
            for {tr <- 1 until height_t
                 tc <- 0 until width_t
            } yield {
              val (id, ir, ic) = idrc(or, od, oc, tr, tc)
              img(id, ir, ic) * theta(td_od(od), tr, tc)
            }
          elems.sum + theta(td_od(od), 0, 0)
        }
      }

val res = out(2, ->, ->)


























