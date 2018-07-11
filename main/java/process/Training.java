package process;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.file.FileMultiDataSetIterator;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import postprocess.Utilities;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

public class Training {
    public static final int seed = 123;
    public static int heightW2V = 60;
    public static int widthW2V = 300;
    public static int nChannelsW2V = 3;
    public static int minTokenLimit = 5;
    public static int maxTokenLimit = 65;
    public static int tokensLimitCount = 15;
    public static int negativeRatio = 3;
    public static int nEpochs = 50;
    public static int miniBatchSize = 64;
    public static double learningRate = 1e-3;
    public static int window = 0;
    public static int nFilters = 50;
    public static String updaterString = null;

    public static void main(String[] args) throws Exception {
        if (args.length > 0) {

            String workingDir = args[1];
            String limitOptions = args[2];
            minTokenLimit = Integer.parseInt(limitOptions.split("_")[0]);
            maxTokenLimit = Integer.parseInt(limitOptions.split("_")[1]);
            tokensLimitCount = Integer.parseInt(limitOptions.split("_")[2]);

            negativeRatio = Integer.parseInt(args[3]);

            String dimensions = args[4];
            heightW2V = Integer.parseInt(dimensions.split("_")[0]);
            widthW2V = Integer.parseInt(dimensions.split("_")[1]);
            nChannelsW2V = Integer.parseInt(dimensions.split("_")[2]);

            learningRate = Double.parseDouble(args[5]);
            nEpochs = Integer.parseInt(args[6]);
            miniBatchSize = Integer.parseInt(args[7]);
            nFilters = Integer.parseInt(args[8]);
            window = Integer.parseInt(args[9]);
            updaterString = args[10];

            boolean match = Boolean.parseBoolean(args[11]);

            boolean test = Boolean.parseBoolean(args[12]);

            System.out.println("-------------------------------------");
            System.out.println("ARGUMENTS");
            System.out.println("-------------------------------------");
            System.out.println("height: " + heightW2V);
            System.out.println("width: " + widthW2V);
            System.out.println("learningRate: " + learningRate);
            System.out.println("nEpochs: " + nEpochs);
            System.out.println("miniBatchSize: " + miniBatchSize);
            System.out.println("window: " + window);
            System.out.println("nFilters: " + nFilters);
            System.out.println("Updater: " + updaterString);
            System.out.println("-------------------------------------");

            RecordReaderMultiDataSetIterator testingRecordReaderMultiDataSetIterator = null;

            if (!test) {
                File matchMultiDataSetFolder = new File(workingDir + File.separator + "IO" + File.separator + "MatchMultiDataSetFiles");
                File facetMultiDataSetFolder = new File(workingDir + File.separator + "IO" + File.separator + "FacetMultiDataSetFiles");

                FileMultiDataSetIterator fileMultiDataSetIterator;
                ComputationGraphConfiguration configuration;
                if (match) {
                    fileMultiDataSetIterator = new FileMultiDataSetIterator(matchMultiDataSetFolder, false, new Random(), miniBatchSize, "bin");
                    configuration = getNetworkConfiguration(2);
                } else {
                    fileMultiDataSetIterator = new FileMultiDataSetIterator(facetMultiDataSetFolder, false, new Random(), miniBatchSize, "bin");
                    configuration = getNetworkConfiguration(5);
                }

                int listenerFrequency = 1;
                UIServer uiServer = UIServer.getInstance();
                StatsStorage statsStorage = new InMemoryStatsStorage();

                ComputationGraph model = new ComputationGraph(configuration);
                model.setListeners(new StatsListener(statsStorage, listenerFrequency));
                uiServer.attach(statsStorage);
                model.init();

                for (int i = 0; i < nEpochs; i++) {
                    System.out.println("Epoch: " + i);

                    model.fit(fileMultiDataSetIterator);
                    fileMultiDataSetIterator.reset();
                }
                System.out.println("Training Finished!!!");

                File locationToSave;
                if (match) {
                    locationToSave = new File(workingDir + File.separator + "matchModel_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" +  negativeRatio +  "_" + learningRate + "_" + nEpochs + ".zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
                } else {
                    locationToSave = new File(workingDir + File.separator + "facetModel_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + learningRate + "_" + nEpochs + ".zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
                }

                boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
                ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            } else {
                //TRAIN models
                ComputationGraph matchModel;
                ComputationGraph facetModel;

                File matchMultiDataSetFolder = new File(workingDir + File.separator + "IO" + File.separator + "MatchMultiDataSetFiles");
                File facetMultiDataSetFolder = new File(workingDir + File.separator + "IO" + File.separator + "FacetMultiDataSetFiles");

                FileMultiDataSetIterator fileMatchMultiDataSetIterator;
                FileMultiDataSetIterator fileFacetMultiDataSetIterator;

                fileMatchMultiDataSetIterator = new FileMultiDataSetIterator(matchMultiDataSetFolder, false, new Random(), miniBatchSize, "bin");
                matchModel = new ComputationGraph(getNetworkConfiguration(2));

                fileFacetMultiDataSetIterator = new FileMultiDataSetIterator(facetMultiDataSetFolder, false, new Random(), miniBatchSize, "bin");
                facetModel = new ComputationGraph(getNetworkConfiguration(5));

                matchModel.init();
                facetModel.init();
                for (int i = 0; i < nEpochs; i++) {
                    System.out.println("Epoch: " + i);

                    matchModel.fit(fileMatchMultiDataSetIterator);
                    fileMatchMultiDataSetIterator.reset();

                    facetModel.fit(fileFacetMultiDataSetIterator);
                    fileFacetMultiDataSetIterator.reset();
                }
                System.out.println("Training Finished!!!");
                System.out.println("Started Testing ...");

                File input1W2V = null;
                File input2W2V = null;
                File input3W2V = null;
                File matchOutputW2V = null;
                File facetOutputW2V = null;

                File testingDataSetFolder = new File(workingDir + File.separator + "datasets" + File.separator + "testing");

                for(File folder: testingDataSetFolder.listFiles()) {
                    input1W2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_Testing_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                    input2W2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_Testing_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                    input3W2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_Testing_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" +  tokensLimitCount + ".csv");

                    matchOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_MatchTesting_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount +  "_"  + learningRate + "_" + updaterString +  "_" + nFilters +  "_" + nEpochs + ".csv");
                    facetOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + folder.getName() + "_FacetTesting_output_" + minTokenLimit + "_" + maxTokenLimit +  "_" + tokensLimitCount + "_" + learningRate + "_" + updaterString +  "_" + nFilters +  "_" + nEpochs +  ".csv");

                    testingRecordReaderMultiDataSetIterator = Utilities.getTestingRecordReaderMultiDataSetIterator(input1W2V.getPath(), input2W2V.getPath(), input3W2V.getPath(), heightW2V, widthW2V, nChannelsW2V, miniBatchSize);
                    testingRecordReaderMultiDataSetIterator.setCollectMetaData(true);

                    while (testingRecordReaderMultiDataSetIterator.hasNext()) {
                        MultiDataSet testData = testingRecordReaderMultiDataSetIterator.next();

                        INDArray[] matchPredictedM = matchModel.output(false, testData.getFeatures());
                        INDArray[] facetPredictedM = facetModel.output(false, testData.getFeatures());

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
            }
        } else {
            System.out.println("Please insert arguments ...");
        }
    }

    private static ComputationGraphConfiguration getNetworkConfiguration(int outputSize) {
        final int cnnLayerFeatureMaps = nFilters;

        //setting updater
        IUpdater iUpdater = new Adam(learningRate);
        switch (updaterString) {
            case "RMSPROP":
                iUpdater = new RmsProp(learningRate);
                break;
            case "ADAGRAD":
                iUpdater = new AdaGrad(learningRate);
                break;
            case "NESTEROVS":
                iUpdater = new Nesterovs(learningRate);
                break;
            case "SGD":
                iUpdater = new Sgd(learningRate);
                break;
            case "ADAM":
                iUpdater = new Adam(learningRate);
                break;
            case "ADAMAX":
                iUpdater = new AdaMax(learningRate);
                break;
            case "NADAM":
                iUpdater = new Nadam(learningRate);
                break;
        }

        //setting loss function
        LossFunctions.LossFunction lossFunction;
        if (outputSize == 2) {
            lossFunction = LossFunctions.LossFunction.XENT;
        } else {
            lossFunction = LossFunctions.LossFunction.MCXENT;
        }

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        //build the network
        return new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(seed)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(iUpdater)
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .graphBuilder()
                .addInputs(/*"input1W2V", */"input2W2V", "input3")
                /*.addLayer("cnn1w2v_input1", new ConvolutionLayer.Builder()
                        .kernelSize(window + 2, widthW2V)
                        .stride(1, widthW2V)
                        .nIn(nChannelsW2V)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input1W2V")
                .addLayer("cnn2w2v_input1", new ConvolutionLayer.Builder()
                        .kernelSize(window + 4, widthW2V)
                        .stride(1, widthW2V)
                        .nIn(nChannelsW2V)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input1W2V")
                .addLayer("cnn3w2v_input1", new ConvolutionLayer.Builder()
                        .kernelSize(window + 6, widthW2V)
                        .stride(1, widthW2V)
                        .nIn(nChannelsW2V)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input1W2V")
                .addVertex("mergew2v_input1", new MergeVertex(), "cnn1w2v_input1", "cnn2w2v_input1", "cnn3w2v_input1")      //Perform depth concatenation

                .addLayer("globalPoolw2v_input1", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .dropOut(0.5)
                        .build(), "mergew2v_input1")*/

                .addLayer("cnn1w2v_input2", new ConvolutionLayer.Builder()
                        .kernelSize(window + 2, widthW2V)
                        .stride(1, widthW2V)
                        .nIn(nChannelsW2V)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input2W2V")
                .addLayer("cnn2w2v_input2", new ConvolutionLayer.Builder()
                        .kernelSize(window + 4, widthW2V)
                        .stride(1, widthW2V)
                        .nIn(nChannelsW2V)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input2W2V")
                .addLayer("cnn3w2v_input2", new ConvolutionLayer.Builder()
                        .kernelSize(window + 6, widthW2V)
                        .stride(1, widthW2V)
                        .nIn(nChannelsW2V)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input2W2V")
                .addVertex("mergew2v_input2", new MergeVertex(), "cnn1w2v_input2", "cnn2w2v_input2", "cnn3w2v_input2")      //Perform depth concatenation

                .addLayer("globalPoolw2v_input2", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .dropOut(0.5)
                        .build(), "mergew2v_input2")

                .addLayer("fully", new DenseLayer.Builder()
                        .nOut(cnnLayerFeatureMaps)
                        //.biasInit(bias)
                        //.dropOut(0.5)
                        .dist(new GaussianDistribution(0, 0.005))
                        .build(), /*"globalPoolw2v_input1", */"globalPoolw2v_input2")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(lossFunction)
                        .activation(Activation.SOFTMAX)
                        .nIn(cnnLayerFeatureMaps + 22)
                        .nOut(outputSize)    //classes
                        .build(), "fully", "input3")
                .setOutputs("out")
                .setInputTypes(/*InputType.convolutionalFlat(heightW2V, widthW2V, nChannelsW2V), */InputType.convolutionalFlat(heightW2V, widthW2V, nChannelsW2V), InputType.feedForward(22))
                .build();
    }
}