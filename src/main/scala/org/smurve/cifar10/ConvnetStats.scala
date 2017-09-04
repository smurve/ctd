package org.smurve.cifar10

case class ConvnetStats(
                         n: Double,
                         score: Double,

                         l0_b_min: Double,
                         l0_b_max: Double,
                         l0_b_minG: Double,
                         l0_b_maxG: Double,
                         l0_W_min: Double,
                         l0_W_max: Double,
                         l0_W_minG: Double,
                         l0_W_maxG: Double,

                         l2_b_min: Double,
                         l2_b_max: Double,
                         l2_b_minG: Double,
                         l2_b_maxG: Double,
                         l2_W_min: Double,
                         l2_W_max: Double,
                         l2_W_minG: Double,
                         l2_W_maxG: Double,

                         l4_b_min: Double,
                         l4_b_max: Double,
                         l4_b_minG: Double,
                         l4_b_maxG: Double,
                         l4_W_min: Double,
                         l4_W_max: Double,
                         l4_W_minG: Double,
                         l4_W_maxG: Double,

                         l5_b_min: Double,
                         l5_b_max: Double,
                         l5_b_minG: Double,
                         l5_b_maxG: Double,
                         l5_W_min: Double,
                         l5_W_max: Double,
                         l5_W_minG: Double,
                         l5_W_maxG: Double
)
