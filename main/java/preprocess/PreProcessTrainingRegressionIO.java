package preprocess;

import org.datavec.api.records.mapper.RecordMapper;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import postprocess.Utilities;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class PreProcessTrainingRegressionIO {
    public static int heightW2V = 30;
    public static int widthW2V = 300;
    public static int nChannelsW2V = 3;
    public static int minTokenLimit = 0;
    public static int maxTokenLimit = 30;
    public static int tokensLimitCount = 15;
    public static int miniBatchSize = 16;

    public static void main(String args[]) throws Exception {
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

            miniBatchSize = Integer.parseInt(args[4]);

            String[] targetOptions = args[5].split("\\_");
            String[] targetClusters = Arrays.copyOfRange(targetOptions, 0, targetOptions.length);
            int i = 0;
            int j = 0;
            for (String cluster : targetClusters) {
                System.out.println("Cluster " + cluster);
                //File matchInput1W2V = new File(workingDir /*+ File.separator + "IO" */+ File.separator + cluster + "_MatchTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchInput2W2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchInput3W2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

                //File facetInput1W2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetInput2W2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetInput3W2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

                int totalRow = heightW2V * widthW2V * nChannelsW2V;

                //Match
                System.out.println("Started Matches");
                MultiDataSetIterator matchTrainingMultiDataSetIterator = null;
                try {
                    matchTrainingMultiDataSetIterator = Utilities.getRegrissionTrainingRecordReaderMultiDataSetIterator(/*matchInput1W2V.getPath()*/"", matchInput2W2V.getPath(), matchInput3W2V.getPath(), matchOutputW2V.getPath(), 1, totalRow, miniBatchSize);

                    while (matchTrainingMultiDataSetIterator.hasNext()) {
                        MultiDataSet multiDataSet = matchTrainingMultiDataSetIterator.next();
                        multiDataSet.save(new File(workingDir + File.separator + "IO" + File.separator + "MatchMultiDataSetFiles" + File.separator + "TrainingMultiDataSet_" + cluster + "_" + minTokenLimit + "_" + maxTokenLimit + "_" + miniBatchSize + "_" + i + ".bin"));
                        i++;
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Done Matches");

                //Facet
                System.out.println("Started Facets");
                MultiDataSetIterator facetTrainingMultiDataSetIterator = null;
                try {
                    facetTrainingMultiDataSetIterator = Utilities.getTrainingRecordReaderMultiDataSetIterator(/*facetInput1W2V.getPath()*/"", facetInput2W2V.getPath(), facetInput3W2V.getPath(), facetOutputW2V.getPath(), 5, totalRow, miniBatchSize);
                    while (facetTrainingMultiDataSetIterator.hasNext()) {
                        MultiDataSet multiDataSet = facetTrainingMultiDataSetIterator.next();
                        multiDataSet.save(new File(workingDir + File.separator + "IO" + File.separator + "FacetMultiDataSetFiles" + File.separator + "TrainingMultiDataSet_" + cluster + "_" + minTokenLimit + "_" + maxTokenLimit + "_" + miniBatchSize + "_" + j + ".bin"));
                        j++;
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Done Facets");
            }
        } else {
            System.out.println("No Arguments !!!");
        }
    }

    private static void removeFirstCol(String from, String to, int nCol) throws Exception {
        File removedColLabelFile = new File(to);
        if (removedColLabelFile.exists()) {
            removedColLabelFile.delete();
        }
        removedColLabelFile.createNewFile();
        Schema s = new Schema.Builder()
                .addColumnString("str")
                .addColumnsDouble("dbl_%d", 1, nCol)
                .build();
        TransformProcess tp = new TransformProcess.Builder(s)
                .removeColumns("str")
                .build();

        RecordMapper rm = RecordMapper.builder()
                .recordReader(new TransformProcessRecordReader(new CSVRecordReader(), tp))
                .recordWriter(new CSVRecordWriter())
                .inputUrl(new FileSplit(new File(from)))
                .outputUrl(new FileSplit(removedColLabelFile))
                .partitioner(new NumberOfRecordsPartitioner())
                .build();
        rm.copy();

        RecordReader outputRR = new CSVRecordReader(0, ',');
        outputRR.initialize(new FileSplit(removedColLabelFile));
    }
}
