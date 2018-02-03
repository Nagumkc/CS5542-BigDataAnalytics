import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
/**
  * Created by VenkatNag on 2/1/2018.
  */
object linear {

  def main(args:Array[String]): Unit =
  {
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    val data = sc.textFile("E:\\UMKC\\Spring_18\\week 3\\3D_spatial_network.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts.slice(1,4).map(_.toDouble)))
    }.cache()
    parsedData.take(1).foreach(f=>println(f))
    val Array(training, test) = parsedData.randomSplit(Array(0.70, 0.30))

    // Building the model
    val numIterations = 50
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("training Mean Squared Error = " + MSE)

    // Evaluate model on test examples and compute training error
    val valuesAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE2 = valuesAndPreds2.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("test Mean Squared Error = " + MSE2)
  }

}
