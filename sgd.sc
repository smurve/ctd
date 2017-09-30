var a: Double = 0.1
var b: Double = -0.1
def f(x: Double) = a * x + b
val (x1, y1) = (2, 3)
val (x2, y2) = (1, 0)
val epsilon = 1e-7

s"f(x1=$x1) = " + f(x1)
s"y1 =      " + y1

def c(x: Double, y: Double) = (f(x) - y) * (f(x) - y)
def C() = c(x1, y1) + c(x2, y2)
// initially, we're off by 2.9

val learn = true

while (
  learn &&
    C() > epsilon) {

  val C_o = C()

  a += epsilon
  a += (if (C() > C_o) -2 * epsilon else 0)

  b += epsilon
  b += (if (C() > C_o ) -2 * epsilon else 0)
}

"f(x1) = " + f(x1)
"y1 =    " + y1
a
b