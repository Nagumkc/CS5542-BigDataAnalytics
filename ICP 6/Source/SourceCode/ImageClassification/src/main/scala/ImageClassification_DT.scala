import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Naga on 19-09-2016.
  */
object ImageClassification_DT {
  def main(args: Array[String]) {
    val IMAGE_CATEGORIES = Array("beaver", "bird", "sealion")
    System.setProperty("hadoop.home.dir", "E:\\UMKC\\Sum_May\\KDM\\winutils")
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);
    val sparkConf = new SparkConf().setAppName("ImageClassification").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val train = sc.textFile("data/train")
    val test = sc.textFile("data/test")
    val parsedData = train.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    val testData1 = test.map(line => {
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    })


    val trainingData = parsedData


    val numClasses = 3
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    val classify1 = testData1.map { line =>
      val prediction = model.predict(line.features)
      (line.label, prediction)
    }

    val prediction1 = classify1.groupBy(_._1).map(f => {
      var fuzzy_Pred = Array(0, 0, 0)
      f._2.foreach(ff => {
        fuzzy_Pred(ff._2.toInt) += 1
      })
      var count = 0.0
      fuzzy_Pred.foreach(f => {
        count += f
      })
      var i = -1
      var maxIndex = 3
      val max = fuzzy_Pred.max
      val pp = fuzzy_Pred.map(f => {
        val p = f * 100 / count
        i = i + 1
        if(f == max)
          maxIndex=i
        (i, p)
      })
      (f._1, pp, maxIndex)
    })
    prediction1.foreach(f => {
      println("\n\n\n" + "Initial Label : "+IMAGE_CATEGORIES(f._1.toInt) + " \n Predicted label : "+ IMAGE_CATEGORIES(f._3)+"\n")
      f._2.foreach(ff=>{
        println(IMAGE_CATEGORIES(ff._1.toInt)+":"+ff._2)
      })
    })
   /* val y: RDD[(Double, Double)] = prediction1.map(f => {
      (f._3.toDouble,f._1 )
    })

    y.collect().foreach(println(_))

    val metrics = new MulticlassMetrics(y)

    println("Accuracy:" + metrics.accuracy)

    println("Confusion Matrix:")
    println(metrics.confusionMatrix)*/
  }
}