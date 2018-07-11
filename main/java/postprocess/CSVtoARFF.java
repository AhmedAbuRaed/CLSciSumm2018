package postprocess;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class CSVtoARFF {
    public static void main(String args[]) {
        if (args.length > 0) {
            try {
                String workingDir = args[1];
                int minTokenLimit = 0;
                int maxTokenLimit = 0;
                int tokensLimitCount = 0;

                try {
                    minTokenLimit = Integer.parseInt(args[2]);
                    maxTokenLimit = Integer.parseInt(args[3]);
                    tokensLimitCount = Integer.parseInt(args[4]);
                } catch (NumberFormatException e) {
                    System.err.println("Arguments " + args[2] + ", " + args[3] + " and " + args[4] + " must be integers.");
                    System.exit(1);
                }
                String[] targetOptions = args[5].split("\\_");
                String[] targetClusters = Arrays.copyOfRange(targetOptions, 0, targetOptions.length);

                for (String cluster : targetClusters) {
                    System.out.println("Cluster " + cluster);

                    CSVLoader loader = new CSVLoader();
                    // Set options
                    loader.setNumericAttributes("last");
                    loader.setMissingValue("?");
                    loader.setFile(new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_wekainput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv"));
                    loader.setNoHeaderRowPresent(true);
                    Instances data = null;
                    data = loader.getDataSet();

                    //Save the arff file
                    ArffSaver saver = new ArffSaver();
                    saver.setFile(new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_wekainput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv.arff"));
                    saver.setInstances(data);
                    saver.writeBatch();
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("Finished ...");

        } else {
            System.out.println("No Arguments !!!");
        }
    }
}
