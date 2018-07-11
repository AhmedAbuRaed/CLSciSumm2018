package process;

import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import postprocess.Utilities;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class RegressionTesting {
    public static int heightW2V = 60;
    public static int widthW2V = 300;
    public static int nChannelsW2V = 3;
    public static int minTokenLimit = 5;
    public static int maxTokenLimit = 65;
    public static int tokensLimitCount = 15;
    public static int nEpochs = 50;
    public static int miniBatchSize = 64;
    public static double learningRate = 1e-3;
    public static int nFilters = 50;
    public static String updaterString = null;

    public static void main(String[] args) throws Exception {
        if (args.length > 0) {
            String workingDir = args[1];
            String limitOptions = args[2];
            minTokenLimit = Integer.parseInt(limitOptions.split("_")[0]);
            maxTokenLimit = Integer.parseInt(limitOptions.split("_")[1]);
            tokensLimitCount = Integer.parseInt(limitOptions.split("_")[2]);

            String dimensions = args[3];
            heightW2V = Integer.parseInt(dimensions.split("_")[0]);
            widthW2V = Integer.parseInt(dimensions.split("_")[1]);
            nChannelsW2V = Integer.parseInt(dimensions.split("_")[2]);

            learningRate = Double.parseDouble(args[4]);
            nEpochs = Integer.parseInt(args[5]);
            miniBatchSize = Integer.parseInt(args[6]);
            nFilters = Integer.parseInt(args[7]);
            updaterString = args[8];

            System.out.println("-------------------------------------");
            System.out.println("ARGUMENTS");
            System.out.println("-------------------------------------");
            System.out.println("height: " + heightW2V);
            System.out.println("width: " + widthW2V);
            System.out.println("learningRate: " + learningRate);
            System.out.println("nEpochs: " + nEpochs);
            System.out.println("miniBatchSize: " + miniBatchSize);
            System.out.println("nFilters: " + nFilters);
            System.out.println("Updater: " + updaterString);
            System.out.println("-------------------------------------");

            RecordReaderMultiDataSetIterator testingRecordReaderMultiDataSetIterator = null;

            System.out.println("Started Testing ...");

            File locationToMatch = new File(workingDir + File.separator + "matchModel_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + nEpochs + ".zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
            File locationToFacet = new File(workingDir + File.separator + "facetModel_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + nEpochs + ".zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally

            ComputationGraph matchRestored = ModelSerializer.restoreComputationGraph(locationToMatch);
            ComputationGraph facetRestored = ModelSerializer.restoreComputationGraph(locationToFacet);

            //File input1W2V = null;
            File input2W2V = null;
            File input3W2V = null;
            File matchOutputW2V = null;
            File facetOutputW2V = null;

            File testingDataSetFolder = new File(workingDir + File.separator + "datasets" + File.separator + "testing");

            for (File folder : testingDataSetFolder.listFiles()) {
                //input1W2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_Testing_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                input2W2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_Testing_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                input3W2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_Testing_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

                matchOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_MatchTesting_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");
                facetOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_FacetTesting_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");

                testingRecordReaderMultiDataSetIterator = Utilities.getTestingRecordReaderMultiDataSetIterator(/*input1W2V.getPath()*/"", input2W2V.getPath(), input3W2V.getPath(), heightW2V, widthW2V, nChannelsW2V, miniBatchSize);
                testingRecordReaderMultiDataSetIterator.setCollectMetaData(true);

                while (testingRecordReaderMultiDataSetIterator.hasNext()) {
                    MultiDataSet testData = testingRecordReaderMultiDataSetIterator.next();

                    INDArray[] matchPredictedM = matchRestored.output(false, testData.getFeatures());
                    INDArray[] facetPredictedM = facetRestored.output(false, testData.getFeatures());

                    INDArray predictedMatch = matchPredictedM[0];
                    INDArray predictedFacet = facetPredictedM[0];

                    PrintWriter pwMatch = new PrintWriter(new FileWriter(matchOutputW2V, true));
                    pwMatch.println(predictedMatch);
                    pwMatch.flush();
                    pwMatch.close();

                    PrintWriter pwFacet = new PrintWriter(new FileWriter(facetOutputW2V, true));
                    pwFacet.println(predictedFacet);
                    pwFacet.flush();
                    pwFacet.close();
                }
            }

        } else {
            System.out.println("Please insert arguments ...");
        }
    }
}
