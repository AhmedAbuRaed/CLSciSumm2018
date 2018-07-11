package postprocess;

import org.apache.commons.lang3.tuple.Pair;

import java.io.*;
import java.util.*;

public class GenerateKokilFormatRegression {
    public static void main(String args[]) {
        if (args.length > 0) {
            String workingDir = args[1];

            int minTokenLimit = 0;
            int maxTokenLimit = 0;
            int tokensLimitCount = 0;
            double learningRate = 1e-3;
            int nFilters = 50;
            int nEpochs = 50;
            int matchOutputNumber = 0;
            int facetOutputNumber = 0;

            try {
                minTokenLimit = Integer.parseInt(args[2]);
                maxTokenLimit = Integer.parseInt(args[3]);
                tokensLimitCount = Integer.parseInt(args[4]);
                learningRate = Double.parseDouble(args[5]);
                nFilters = Integer.parseInt(args[6]);
                nEpochs = Integer.parseInt(args[7]);
                matchOutputNumber = Integer.parseInt(args[8]);
                facetOutputNumber = Integer.parseInt(args[9]);
            } catch (NumberFormatException e) {
                System.err.println("Arguments from index 2-8 must be integers.");
                System.exit(1);
            }
            String updaterString = args[10];

            String[] targetOptions = args[11].split("\\_");
            String[] targetClusters = Arrays.copyOfRange(targetOptions, 0, targetOptions.length);

            for (String cluster : targetClusters) {
                File input1 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File input2 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File input3 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File idMatchOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTesting_idoutput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");
                File idFacetOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTesting_idoutput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");

                File gold = new File(workingDir + File.separator + "Gold" + File.separator + "ref" + File.separator + "Task1" + File.separator + cluster + ".annv3.txt");
                File kokiloutput = new File(workingDir + File.separator + "IO" + File.separator + cluster + ".annv3.txt");

                HashMap<String, ArrayList<Pair<Double, Integer>>> matchProbabilities = new HashMap<String, ArrayList<Pair<Double, Integer>>>();
                HashMap<String, ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>>> facetProbabilities = new HashMap<String, ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>>>();
                BufferedReader matchReader;
                BufferedReader facetReader;
                String matchLine;
                String facetLine;
                try {
                    matchReader = new BufferedReader(
                            new InputStreamReader(
                                    new FileInputStream(idMatchOutputW2V), "UTF-8"));

                    facetReader = new BufferedReader(
                            new InputStreamReader(
                                    new FileInputStream(idFacetOutputW2V), "UTF-8"));

                    while (((matchLine = matchReader.readLine()) != null) && ((facetLine = facetReader.readLine()) != null)) {
                        if (!matchLine.equals("") && !facetLine.equals("")) {
                            String[] matchFields = matchLine.split(",");
                            String[] matchids = matchFields[0].split("_");

                            String[] facetFields = facetLine.split(",");
                            String[] facetids = facetFields[0].split("_");

                            if (!matchids[3].equals(facetids[3]) && !matchids[4].equals(facetids[4]) && !matchids[5].equals(facetids[5])) {
                                System.out.println("NOOOOOOOOOOOO");
                                System.exit(-1);
                            }

                            if (matchids.length == 6 && facetids.length == 6) {
                                //they both are the same but lets break them down anyway
                                //match
                                if (matchProbabilities.containsKey(matchids[3] + "_" + matchids[4] + "_" + matchids[5])) {
                                    ArrayList<Pair<Double, Integer>> temp = matchProbabilities.get(matchids[3] + "_" + matchids[4] + "_" + matchids[5]);
                                    temp.add(Pair.of(Double.parseDouble(matchFields[1]), Integer.parseInt(matchids[1])));

                                    matchProbabilities.put(matchids[3] + "_" + matchids[4] + "_" + matchids[5], temp);
                                } else {
                                    ArrayList<Pair<Double, Integer>> temp = new ArrayList<Pair<Double, Integer>>();
                                    temp.add(Pair.of(Double.parseDouble(matchFields[1]), Integer.parseInt(matchids[1])));

                                    matchProbabilities.put(matchids[3] + "_" + matchids[4] + "_" + matchids[5], temp);
                                }
                                //facet
                                if (facetProbabilities.containsKey(facetids[3] + "_" + facetids[4] + "_" + facetids[5])) {
                                    ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>> temp = facetProbabilities.get(facetids[3] + "_" + facetids[4] + "_" + facetids[5]);
                                    ArrayList<Pair<Integer, Double>> facets = new ArrayList();
                                    facets.add(Pair.of(0, Double.parseDouble(facetFields[1])));
                                    facets.add(Pair.of(1, Double.parseDouble(facetFields[2])));
                                    facets.add(Pair.of(2, Double.parseDouble(facetFields[3])));
                                    facets.add(Pair.of(3, Double.parseDouble(facetFields[4])));
                                    facets.add(Pair.of(4, Double.parseDouble(facetFields[5])));

                                    temp.add(Pair.of(facets, Integer.parseInt(facetids[1])));

                                    facetProbabilities.put(facetids[3] + "_" + facetids[4] + "_" + facetids[5], temp);
                                } else {
                                    ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>> temp = new ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>>();
                                    ArrayList<Pair<Integer, Double>> facets = new ArrayList();
                                    facets.add(Pair.of(0, Double.parseDouble(facetFields[1])));
                                    facets.add(Pair.of(1, Double.parseDouble(facetFields[2])));
                                    facets.add(Pair.of(2, Double.parseDouble(facetFields[3])));
                                    facets.add(Pair.of(3, Double.parseDouble(facetFields[4])));
                                    facets.add(Pair.of(4, Double.parseDouble(facetFields[5])));
                                    temp.add(Pair.of(facets, Integer.parseInt(facetids[1])));

                                    facetProbabilities.put(facetids[3] + "_" + facetids[4] + "_" + facetids[5], temp);
                                }
                            } else {
                                //they both are the same but lets break them down anyway
                                //match
                                if (matchProbabilities.containsKey(matchids[3] + "_" + matchids[4])) {
                                    ArrayList<Pair<Double, Integer>> temp = matchProbabilities.get(matchids[3] + "_" + matchids[4]);
                                    temp.add(Pair.of(Double.parseDouble(matchFields[1]), Integer.parseInt(matchids[1])));

                                    matchProbabilities.put(matchids[3] + "_" + matchids[4], temp);
                                } else {
                                    ArrayList<Pair<Double, Integer>> temp = new ArrayList<Pair<Double, Integer>>();
                                    temp.add(Pair.of(Double.parseDouble(matchFields[1]), Integer.parseInt(matchids[1])));

                                    matchProbabilities.put(matchids[3] + "_" + matchids[4], temp);
                                }
                                //facet
                                if (facetProbabilities.containsKey(facetids[3] + "_" + facetids[4])) {
                                    ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>> temp = facetProbabilities.get(facetids[3] + "_" + facetids[4]);
                                    ArrayList<Pair<Integer, Double>> facets = new ArrayList();
                                    facets.add(Pair.of(0, Double.parseDouble(facetFields[1])));
                                    facets.add(Pair.of(1, Double.parseDouble(facetFields[2])));
                                    facets.add(Pair.of(2, Double.parseDouble(facetFields[3])));
                                    facets.add(Pair.of(3, Double.parseDouble(facetFields[4])));
                                    facets.add(Pair.of(4, Double.parseDouble(facetFields[5])));
                                    temp.add(Pair.of(facets, Integer.parseInt(facetids[1])));

                                    facetProbabilities.put(facetids[3] + "_" + facetids[4], temp);
                                } else {
                                    ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>> temp = new ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>>();
                                    ArrayList<Pair<Integer, Double>> facets = new ArrayList();
                                    facets.add(Pair.of(0, Double.parseDouble(facetFields[1])));
                                    facets.add(Pair.of(1, Double.parseDouble(facetFields[2])));
                                    facets.add(Pair.of(2, Double.parseDouble(facetFields[3])));
                                    facets.add(Pair.of(3, Double.parseDouble(facetFields[4])));
                                    facets.add(Pair.of(4, Double.parseDouble(facetFields[5])));
                                    temp.add(Pair.of(facets, Integer.parseInt(facetids[1])));

                                    facetProbabilities.put(facetids[3] + "_" + facetids[4], temp);
                                }
                            }
                        }
                    }

                    for (String key : matchProbabilities.keySet()) {
                        Collections.sort(matchProbabilities.get(key), Comparator.comparing(p -> -p.getLeft()));
                    }

                    ArrayList<SciSummAnnotation> goldList = Utilities.importSciSummOutput(gold);
                    ArrayList<SciSummAnnotation> outputList = new ArrayList<SciSummAnnotation>();

                    for (SciSummAnnotation goldAnno : goldList) {
                        SciSummAnnotation outputAnno = new SciSummAnnotation();
                        outputAnno.setCiting_Article(goldAnno.getCiting_Article());
                        outputAnno.setReference_Article(goldAnno.getReference_Article());
                        outputAnno.setCitance_Number(goldAnno.getCitance_Number());
                        outputAnno.setCitation_Marker(goldAnno.getCitation_Marker());
                        outputAnno.setCitation_Marker_Offset(goldAnno.getCitation_Marker_Offset());
                        outputAnno.setAnnotator("Unknown");
                        outputAnno.setCitation_Text(goldAnno.getCitation_Text());
                        outputAnno.setReference_Text(goldAnno.getReference_Text());

                        for (String citoff : goldAnno.getCitation_Offset()) {
                            outputAnno.getCitation_Offset().add(citoff);
                            if (matchProbabilities.containsKey(outputAnno.getCiting_Article() + "_" + citoff)) {
                                ArrayList<Pair<Double, Integer>> matchtemp = matchProbabilities.get(outputAnno.getCiting_Article() + "_" + citoff);
                                ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>> facettemp = facetProbabilities.get(outputAnno.getCiting_Article() + "_" + citoff);
                                for (int i = 0; i < matchOutputNumber; i++) {
                                    if (!outputAnno.getReference_Offset().contains(matchtemp.get(i).getRight().toString())) {
                                        outputAnno.getReference_Offset().add(matchtemp.get(i).getRight().toString());

                                        for (int j = 0; j < facettemp.size(); j++) {
                                            if (facettemp.get(j).getRight() == matchtemp.get(i).getRight()) {
                                                ArrayList<Pair<Integer, Double>> facetsSorted = facettemp.get(j).getLeft();
                                                Collections.sort(facetsSorted, Comparator.comparing(p -> -p.getRight()));

                                                for (int k = 0; k < facetOutputNumber; k++) {
                                                    if (!outputAnno.getDiscourse_Facet().contains(getFacetFromIndex(facetsSorted.get(k).getLeft()))) {
                                                        outputAnno.getDiscourse_Facet().add(getFacetFromIndex(facetsSorted.get(k).getLeft()));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        //outputAnno.setDiscourse_Facet(goldAnno.getDiscourse_Facet());
                        outputList.add(outputAnno);
                    }

                    Utilities.exportSciSummOutput(outputList, kokiloutput);

                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            System.out.println("Finished ...");
        } else {
            System.out.println("No Arguments !!!");
        }
    }

    public static String getFacetFromIndex(int index) {
        String facet = null;
        switch (index) {
            case 0:
                facet = "Aim_Citation";
                break;
            case 1:
                facet = "Hypothesis_Citation";
                break;
            case 2:
                facet = "Method_Citation";
                break;
            case 3:
                facet = "Implication_Citation";
                break;
            case 4:
                facet = "Results_Citation";
                break;
        }
        if (facet == null) {
            System.out.println("Facet Index does not exists");
            System.exit(-1);
        }

        return facet;
    }
}
