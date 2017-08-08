import org.nd4s.Implicits._
import org.smurve.nd4s._

import scala.util.Random


val samples = vec(1,2,3,4,5,6,7,8,9,2,3,4,3,4,5).reshape(5,3)

val rnd = Random

samples.size(0)

samples.size(1)

( 0 until samples.size(0)).map(i=>(samples(i,->), rnd.nextInt())).sortBy(_._2)



