import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by VenkatNag on 1/26/2018.
  */
object group {

  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "E:\\UMKC\\Sum_May\\KDM\\winutils")
    val conf = new SparkConf().setAppName("group by example").setMaster("local").set("spark.driver.host","localhost")
    val sc = new SparkContext(conf)
    val f=sc.textFile("E:\\UMKC\\Spring_18\\Week 2\\CS5542-Tutorial1B-SparkSourceCode\\Spark WordCount\\input")
    val wc=f.flatMap(line=>{line.split(" ")})
    val out=wc.groupBy(word=>word.charAt(0))
    out.foreach(println)
}
}
