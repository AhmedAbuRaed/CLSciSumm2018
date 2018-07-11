package postprocess;

import gate.*;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

public class Utilities {

    public static ArrayList<SciSummAnnotation> importSciSummOutput(File file) {
        ArrayList<SciSummAnnotation> annotationsList = new ArrayList<SciSummAnnotation>();

        BufferedReader reader;
        String line;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(file), "UTF-8"));

            while ((line = reader.readLine()) != null) {
                if (!line.equals("")) {
                    String[] fields = line.split("\\|");

                    SciSummAnnotation annotation = new SciSummAnnotation();
                    annotation.setCitance_Number(fields[0].trim().split(":")[1].trim());
                    if (fields[1].trim().split(":")[1].trim().contains(".")) {
                        annotation.setReference_Article(fields[1].trim().split(":")[1].trim().substring(0, fields[1].trim().split(":")[1].trim().lastIndexOf('.')));
                    } else {
                        annotation.setReference_Article(fields[1].trim().split(":")[1].trim());
                    }
                    if (fields[2].trim().split(":")[1].trim().contains(".")) {
                        annotation.setCiting_Article(fields[2].trim().split(":")[1].trim().substring(0, fields[2].trim().split(":")[1].trim().lastIndexOf('.')));
                    } else {
                        annotation.setCiting_Article(fields[2].trim().split(":")[1].trim());
                    }
                    annotation.setCitation_Marker_Offset(fields[3].trim().split(":")[1].trim().replaceAll("\\D+", ""));
                    annotation.setCitation_Marker(fields[4].trim().split(":")[1].trim());
                    for (String co : fields[5].trim().split(":")[1].trim().split(",")) {
                        annotation.getCitation_Offset().add(co.replaceAll("\\D+", ""));
                    }
                    annotation.setCitation_Text(fields[6].trim().split(":")[1].trim());

                    for (String ro : fields[7].trim().split(":")[1].trim().split(",")) {
                        annotation.getReference_Offset().add(ro.replaceAll("\\D+", ""));
                    }
                    annotation.setReference_Text(fields[8].trim().split(":")[1].trim());
                    for (String facet : fields[9].trim().split(":")[1].trim().split(",")) {
                        annotation.getDiscourse_Facet().add(facet.replaceAll("[^a-zA-Z0-9_]", "").trim());
                    }

                    annotationsList.add(annotation);
                }
            }

            reader.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return annotationsList;
    }

    public static void exportSciSummOutput(ArrayList<SciSummAnnotation> output, File file) {
        for (int anno = 0; anno < output.size(); anno++) {
            SciSummAnnotation sciSummAnnotation = output.get(anno);

            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.append("Citance Number: ");
            stringBuilder.append(sciSummAnnotation.getCitance_Number());
            stringBuilder.append(" | Reference Article: ");
            stringBuilder.append(sciSummAnnotation.getReference_Article());
            stringBuilder.append(" | Citing Article: ");
            stringBuilder.append(sciSummAnnotation.getCiting_Article());
            stringBuilder.append(" | Citation Marker Offset:  '");
            stringBuilder.append(sciSummAnnotation.getCitation_Marker_Offset());
            stringBuilder.append("' | Citation Marker: ");
            stringBuilder.append(sciSummAnnotation.getCitation_Marker());
            stringBuilder.append(" | Citation Offset:  ");

            for (int i = 0; i < sciSummAnnotation.getCitation_Offset().size(); i++) {
                stringBuilder.append("'" + sciSummAnnotation.getCitation_Offset().get(i) + "'");
                if (i != sciSummAnnotation.getCitation_Offset().size() - 1) {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append(" | Citation Text: ");
            stringBuilder.append(sciSummAnnotation.getCitation_Text());
            stringBuilder.append(" | Reference Offset:  ");
            for (int i = 0; i < sciSummAnnotation.getReference_Offset().size(); i++) {
                stringBuilder.append("'" + sciSummAnnotation.getReference_Offset().get(i) + "'");
                if (i != sciSummAnnotation.getReference_Offset().size() - 1) {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append(" | Reference Text: ");
            stringBuilder.append(sciSummAnnotation.getReference_Text());
            stringBuilder.append(" | Discourse Facet:  ");
            for (int i = 0; i < sciSummAnnotation.getDiscourse_Facet().size(); i++) {
                stringBuilder.append("'" + sciSummAnnotation.getDiscourse_Facet().get(i) + "'");
                if (i != sciSummAnnotation.getDiscourse_Facet().size() - 1) {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append(" | Annotator: ");
            stringBuilder.append(sciSummAnnotation.getAnnotator());
            stringBuilder.append(" |");
            //stringBuilder.append(System.getProperty("line.separator"));

            if (file.exists()) {
                try (FileWriter fw = new FileWriter(file, true);
                     BufferedWriter bw = new BufferedWriter(fw);
                     PrintWriter out = new PrintWriter(bw)) {
                    out.println(stringBuilder.toString());
                    //more code
                    bw.flush();
                    bw.close();
                } catch (IOException e) {
                    //exception handling left as an exercise for the reader
                }
            } else {
                BufferedWriter writer = null;
                try {
                    writer = new BufferedWriter(new FileWriter(file));

                    writer.write(stringBuilder.toString());
                    writer.newLine();
                    writer.flush();
                    writer.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void append2CSVFile(String lineCSV, File file) {
        PrintWriter out = null;
        BufferedWriter bufWriter;

        try {
            bufWriter =
                    Files.newBufferedWriter(
                            Paths.get(file.toURI()),
                            Charset.forName("UTF8"),
                            StandardOpenOption.WRITE,
                            StandardOpenOption.APPEND,
                            StandardOpenOption.CREATE);
            out = new PrintWriter(bufWriter, true);
        } catch (IOException e) {
            //Oh, no! Failed to create PrintWriter
            e.printStackTrace();
        }

        //After successful creation of PrintWriter
        out.println(lineCSV);

        //After done writing, remember to close!
        out.close();
    }

    public static Set<String> getStopWordsSet(String workingDir) {
        Set<String> stopWords = new HashSet<String>();
        BufferedReader readerStopWords = null;
        try {
            String line;
            readerStopWords = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(workingDir + File.separator + "full-stop-words.lst"), "UTF-8"));
            while ((line = readerStopWords.readLine()) != null) {
                if (!line.equals("")) {
                    stopWords.add(line.toLowerCase().trim());
                }
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stopWords;
    }


    public static RecordReaderMultiDataSetIterator getTrainingRecordReaderMultiDataSetIterator(String input1Path, String input2Path, String input3Path, String outputPath, int numClasses, int totalRow, int miniBatchSize) throws IOException, InterruptedException {
        RecordReaderMultiDataSetIterator recordReaderMultiDataSetIterator = null;

            /*RecordReader input1RR = new CSVRecordReader(0, ',');
            input1RR.initialize(new FileSplit(new File(input1Path)));*/

            RecordReader input2RR = new CSVRecordReader(0, ',');
            input2RR.initialize(new FileSplit(new File(input2Path)));

            RecordReader input3RR = new CSVRecordReader(0, ',');
            input3RR.initialize(new FileSplit(new File(input3Path)));

            RecordReader outputRR = new CSVRecordReader(0, ',');
            outputRR.initialize(new FileSplit(new File(outputPath)));

            recordReaderMultiDataSetIterator = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
                    //.addReader("input1W2V", input1RR)
                    .addReader("input2W2V", input2RR)
                    .addReader("input3W2V", input3RR)
                    .addReader("labelsW2V", outputRR)
                    //.addInput("input1W2V", 1, totalRow)
                    .addInput("input2W2V", 1, totalRow)
                    .addInput("input3W2V", 1, 22)
                    .addOutputOneHot("labelsW2V",1, numClasses)
                    .build();

        return recordReaderMultiDataSetIterator;
    }

    public static RecordReaderMultiDataSetIterator getRegrissionTrainingRecordReaderMultiDataSetIterator(String input1Path, String input2Path, String input3Path, String outputPath, int columnLast, int totalRow, int miniBatchSize) throws IOException, InterruptedException {
        RecordReaderMultiDataSetIterator recordReaderMultiDataSetIterator = null;

            /*RecordReader input1RR = new CSVRecordReader(0, ',');
            input1RR.initialize(new FileSplit(new File(input1Path)));*/

        RecordReader input2RR = new CSVRecordReader(0, ',');
        input2RR.initialize(new FileSplit(new File(input2Path)));

        RecordReader input3RR = new CSVRecordReader(0, ',');
        input3RR.initialize(new FileSplit(new File(input3Path)));

        RecordReader outputRR = new CSVRecordReader(0, ',');
        outputRR.initialize(new FileSplit(new File(outputPath)));

        recordReaderMultiDataSetIterator = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
                //.addReader("input1W2V", input1RR)
                .addReader("input2W2V", input2RR)
                .addReader("input3W2V", input3RR)
                .addReader("labelsW2V", outputRR)
                //.addInput("input1W2V", 1, totalRow)
                .addInput("input2W2V", 1, totalRow)
                .addInput("input3W2V", 1, 22)
                .addOutput("labelsW2V",1, columnLast)
                .build();

        return recordReaderMultiDataSetIterator;
    }

    public static RecordReaderMultiDataSetIterator getTestingRecordReaderMultiDataSetIterator(String input1Path, String input2Path, String input3Path, int heightW2V, int widthW2V, int nChannelsW2V, int miniBatchSize) throws IOException, InterruptedException {

        RecordReaderMultiDataSetIterator recordReaderMultiDataSetIterator = null;

        int totalRow = heightW2V * widthW2V * nChannelsW2V;

            /*RecordReader input1RR = new CSVRecordReader(0, ',');
            input1RR.initialize(new FileSplit(new File(input1Path)));*/

            RecordReader input2RR = new CSVRecordReader(0, ',');
            input2RR.initialize(new FileSplit(new File(input2Path)));

            RecordReader input3RR = new CSVRecordReader(0, ',');
            input3RR.initialize(new FileSplit(new File(input3Path)));

            recordReaderMultiDataSetIterator = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
                    //.addReader("input1W2V", input1RR)
                    .addReader("input2W2V", input2RR)
                    .addReader("input3W2V", input3RR)
                    //.addInput("input1W2V", 1, totalRow)
                    .addInput("input2W2V", 1, totalRow)
                    .addInput("input3W2V", 1, 22)
                    .build();

        return recordReaderMultiDataSetIterator;

    }

    public static List<Annotation> filterRefSentences(Document refDocument, Integer minTokenLimit, Integer maxTokenLimit)
    {
        AnnotationSet refReferenceSentences = refDocument.getAnnotations("REFERENCES");
        AnnotationSet refAnalysis= refDocument.getAnnotations("Analysis");
        AnnotationSet refTotalSentences = refAnalysis.get("Sentence");
        FeatureMap fm = Factory.newFeatureMap();
        fm.put("kind", "word");
        AnnotationSet refWordTokens = refAnalysis.get("Token", fm);

        List<Annotation> filtered = refAnalysis.get("Sentence").inDocumentOrder();
        filtered.clear();

        for(Annotation refTotalSentence: refTotalSentences)
        {
            AnnotationSet refReferenceSentence = refReferenceSentences.get(refTotalSentence.getStartNode().getOffset(), refTotalSentence.getEndNode().getOffset());
            if(refReferenceSentence.size() == 0)
            {
                AnnotationSet tokens = refWordTokens.get(refTotalSentence.getStartNode().getOffset(), refTotalSentence.getEndNode().getOffset());
                if((tokens.size() >= minTokenLimit) && (tokens.size() <= maxTokenLimit))
                {
                    filtered.add(refTotalSentence);
                }
            }
        }
        return filtered;
    }

    public static FeatureMap combineNormalizedVectors(FeatureMap normalizedVector1, FeatureMap
            normalizedVector2) {
        FeatureMap combineNormalizedVector = Factory.newFeatureMap();
        for (Object key : normalizedVector1.keySet()) {
            if (normalizedVector2.containsKey(key)) {
                combineNormalizedVector.put(key, String.valueOf((new Double(normalizedVector1.get(key).toString()) + new Double(normalizedVector2.get(key).toString())) / 2.0));
            } else {
                combineNormalizedVector.put(key, String.valueOf((new Double(normalizedVector1.get(key).toString()) / 2.0)));
            }
        }

        for (Object key : normalizedVector2.keySet()) {
            if (!normalizedVector1.containsKey(key)) {
                combineNormalizedVector.put(key, String.valueOf((new Double(normalizedVector2.get(key).toString()) / 2.0)));
            }
        }
        return combineNormalizedVector;
    }
}
