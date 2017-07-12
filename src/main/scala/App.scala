import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{Row, SparkSession}

object App {

  def main(args: Array[String]) {

    val predictedField = "score_manual_postinstall"
    val trainFields = Array("location_country", "device_os", "device_vendor", "network_type", "proxy_type", "location_city", "device_app_category", "isp_maxmind")

    //data load and preparation
    val spark = SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate

    val csv = spark
      .read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("in-org16-june.csv")
      .na.fill("")
      .na.fill(0)
      .na.fill(0.0)

    import spark.sqlContext.implicits._

    //filtering out non 0|1 values
    val data = csv.filter((csv(predictedField) === "0" || csv(predictedField) === "1"))
      .withColumn(predictedField, csv(predictedField)
        .cast(DoubleType))

    data.show(100)

    val Array(training, test) = data.randomSplit(Array(0.8, 0.2))

    val stringIndexers = trainFields
      .map(f => new StringIndexer()
        .setInputCol(f)
        .setOutputCol(s"${f}_index")
        .setHandleInvalid("skip")
      )

    val vectorAssembler = new VectorAssembler()
      .setInputCols(trainFields.map(f => s"${f}_index"))
      .setOutputCol("features")

    val randomForestRegressor = new RandomForestRegressor()
      .setLabelCol("score_manual_postinstall")
      .setFeaturesCol("features")
      .setMaxBins(500)

    val pipeline = new Pipeline()
      .setStages(stringIndexers:+ vectorAssembler:+ randomForestRegressor)

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestRegressor.numTrees, Array(3, 4, 5 ,6 ,7))
      .addGrid(randomForestRegressor.maxDepth, Array(3, 4, 5, 6, 7, 8))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator().setLabelCol(predictedField))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val model = cv.fit(training)

    val fi = model
      .bestModel
      .asInstanceOf[PipelineModel]
      .stages
      .last
      .asInstanceOf[RandomForestRegressionModel]
      .featureImportances

    println("Feature Importances: " + fi)

    val valuesAndPreds = model
      .bestModel
      .transform(test)
      .select(predictedField, "features", "prediction")
      .rdd
      .map { r =>
        (r.getAs[Double]("prediction"), r.getAs[Double](predictedField))
      }

    val metrics = new RegressionMetrics(valuesAndPreds)

    // Squared error
    println(s"MSE = ${metrics.meanSquaredError}")
    println(s"RMSE = ${metrics.rootMeanSquaredError}")
    println(s"MASE = ${metrics.meanAbsoluteError}")
    println(s"R2 = ${metrics.r2}")

    spark.stop()
  }

}
