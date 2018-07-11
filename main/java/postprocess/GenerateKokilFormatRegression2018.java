package postprocess;

import org.apache.commons.lang3.tuple.Pair;

import java.io.*;
import java.util.*;

public class GenerateKokilFormatRegression2018 {
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
                File idMatchOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTesting_idoutput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");
                File idFacetOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTesting_idoutput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");

                File csvInput = new File(workingDir + File.separator + "datasets" + File.separator + "testing" + File.separator + cluster + File.separator + "annotation" + File.separator + cluster + ".csv");
                File csvOutput = new File(workingDir + File.separator + "IO" + File.separator + cluster + ".csv");

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

                    BufferedReader csvInputReader;
                    csvInputReader = new BufferedReader(
                            new InputStreamReader(
                                    new FileInputStream(csvInput), "UTF-8"));
                    String csvInputLine;

                    //copy header to output
                    try (FileWriter fw = new FileWriter(csvOutput, true);
                         BufferedWriter bw = new BufferedWriter(fw);
                         PrintWriter out = new PrintWriter(bw)) {
                        out.println(csvInputReader.readLine());
                        //more code
                        bw.flush();
                        bw.close();
                    } catch (IOException e) {
                        //exception handling left as an exercise for the reader
                        e.printStackTrace();
                    }

                    while ((csvInputLine = csvInputReader.readLine()) != null) {
                        Boolean stopFacets = false;
                        StringBuilder stringBuilderMatches = new StringBuilder();
                        StringBuilder stringBuilderFacets = new StringBuilder();
                        stringBuilderMatches.append("\"[");
                        stringBuilderFacets.append("\"[");
                        String line[] = csvInputLine.split(",");
                        ArrayList<Pair<Double, Integer>> matchesProp = matchProbabilities.get(line[2] + "_" + line[0]);

                        for (int i = 0; i < matchOutputNumber; i++) {
                            stringBuilderMatches.append(matchesProp.get(i).getRight());
                            if (i != matchOutputNumber - 1) {
                                stringBuilderMatches.append(",");
                            }

                            ArrayList<Pair<ArrayList<Pair<Integer, Double>>, Integer>> facetProp = facetProbabilities.get(line[2] + "_" + line[0]);
                            for (int j = 0; j < facetProp.size(); j++) {
                                if (facetProp.get(j).getRight() == matchesProp.get(i).getRight()) {
                                    ArrayList<Pair<Integer, Double>> facetsSorted = facetProp.get(j).getLeft();
                                    Collections.sort(facetsSorted, Comparator.comparing(p -> -p.getRight()));

                                    if (!stopFacets) {
                                        for (int k = 0; k < facetOutputNumber; k++) {
                                            if (!stringBuilderFacets.toString().contains(getFacetFromIndex(facetsSorted.get(k).getLeft()))) {
                                                stringBuilderFacets.append(getFacetFromIndex(facetsSorted.get(k).getLeft()) + ",");
                                            }
                                        }
                                    }
                                    stopFacets = true;
                                }
                            }
                        }
                        if (stringBuilderFacets.toString().endsWith(",")) {
                            stringBuilderFacets.deleteCharAt(stringBuilderFacets.length() - 1);
                        }
                        stringBuilderMatches.append("]\"");
                        stringBuilderFacets.append("]\"");

                        String csvOutputLine = csvInputLine.substring(0, csvInputLine.length() - 2) + stringBuilderMatches.toString() + ",," + stringBuilderFacets.toString();
                        try (FileWriter fw = new FileWriter(csvOutput, true);
                             BufferedWriter bw = new BufferedWriter(fw);
                             PrintWriter out = new PrintWriter(bw)) {
                            out.println(csvOutputLine);
                            //more code
                            bw.flush();
                            bw.close();
                        } catch (IOException e) {
                            //exception handling left as an exercise for the reader
                            e.printStackTrace();
                        }
                    }

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
