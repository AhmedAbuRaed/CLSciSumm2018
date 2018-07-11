package postprocess;

import java.io.*;
import java.util.Arrays;

public class IDWEKAOutputInstances {
    public static void main(String args[]) {
        if (args.length > 0) {
            String workingDir = args[1];

            File testingXMLDatasetPath = new File(workingDir + File.separator + "datasets" + File.separator + "testing");
            int minTokenLimit = 0;
            int maxTokenLimit = 0;
            int tokensLimitCount = 0;
            /*double learningRate = 1e-3;
            int nFilters = 50;
            int nEpochs = 50;*/

            try {
                minTokenLimit = Integer.parseInt(args[2]);
                maxTokenLimit = Integer.parseInt(args[3]);
                tokensLimitCount = Integer.parseInt(args[4]);
                /*learningRate = Double.parseDouble(args[4]);
                nFilters = Integer.parseInt(args[5]);
                nEpochs = Integer.parseInt(args[6]);*/
            } catch (NumberFormatException e) {
                System.err.println("Arguments from index 2-6 must be integers.");
                System.exit(1);
            }
            //String updaterString = args[7];
            String[] targetOptions = args[5].split("\\_");
            String[] targetClusters = Arrays.copyOfRange(targetOptions, 0, targetOptions.length);

            for (String cluster : targetClusters) {

                File input1 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File input2 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File input3 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_Testing_wekainput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv.arff.csv");
                //File facetOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTesting_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");

                File idMatchOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTesting_idoutput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount/* + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs*/ + ".csv");
                //File idFacetOutputW2V = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_FacetTesting_idoutput_" + minTokenLimit + "_" + maxTokenLimit + "_" + learningRate + "_" + updaterString + "_" + nFilters + "_" + nEpochs + ".csv");

                BufferedReader brinput1 = null;
                BufferedReader brinput2 = null;
                BufferedReader brinput3 = null;
                BufferedReader brMatchoutput = null;
                //BufferedReader brFacetoutput = null;
                try {
                    brinput1 = new BufferedReader(new FileReader(input1));
                    brinput2 = new BufferedReader(new FileReader(input2));
                    brinput3 = new BufferedReader(new FileReader(input3));
                    brMatchoutput = new BufferedReader(new FileReader(matchOutputW2V));
                    //brFacetoutput = new BufferedReader(new FileReader(facetOutputW2V));

                    String lineinput1;
                    String lineinput2;
                    String lineinput3;
                    String matchLineoutput;
                    //String facetLineOutput;
                    while (((lineinput1 = brinput1.readLine()) != null) && ((lineinput2 = brinput2.readLine()) != null) && ((lineinput3 = brinput3.readLine()) != null) &&
                            ((matchLineoutput = brMatchoutput.readLine()) != null) /*&& ((facetLineOutput = brFacetoutput.readLine()) != null)*/) {
                        if ((!lineinput1.split(",")[0].equals(lineinput2.split(",")[0])) && (!lineinput2.split(",")[0].equals(lineinput3.split(",")[0]))) {
                            System.out.println("The inputs are not Siamese :'(");
                            System.exit(-1);
                        }

                        matchLineoutput = matchLineoutput.replaceAll("[\\[\\] ]", "");
                        if (matchLineoutput.endsWith(",") ) {
                            matchLineoutput = matchLineoutput.substring(0, matchLineoutput.length() - 1);
                        }
                        /*facetLineOutput = facetLineOutput.replaceAll("[\\[\\] ]", "");
                        if(facetLineOutput.endsWith(","))
                        {
                            facetLineOutput = facetLineOutput.substring(0, facetLineOutput.length() - 1);
                        }*/

                        PrintWriter pwMatch = new PrintWriter(new FileWriter(idMatchOutputW2V, true));
                        pwMatch.println(lineinput1.split(",")[0] + "," + matchLineoutput);
                        pwMatch.flush();
                        pwMatch.close();

                        /*PrintWriter pwFacet = new PrintWriter(new FileWriter(idFacetOutputW2V, true));
                        pwFacet.println(lineinput1.split(",")[0] + "," + facetLineOutput);
                        pwFacet.flush();
                        pwFacet.close();*/
                    }
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    try {
                        if (brinput1 != null) {
                            brinput1.close();
                        }
                        if (brinput2 != null) {
                            brinput2.close();
                        }
                        if (brinput3 != null) {
                            brinput3.close();
                        }
                        if (brMatchoutput != null) {
                            brMatchoutput.close();
                        }
                        /*if (brFacetoutput != null) {
                            brFacetoutput.close();
                        }*/
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            System.out.println("Finished ...");
        } else {
            System.out.println("No Arguments !!!");
        }
    }
}
