package preprocess;

import postprocess.Utilities;

import java.io.*;
import java.util.Arrays;

public class AppendClustersToTrainingDataset {
    public static void main(String args[]) {
        if (args.length > 0) {
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

            File finalmatchInput1 = new File(workingDir + File.separator + "MatchTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
            File finalmatchInput2 = new File(workingDir + File.separator + "MatchTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
            File finalmatchInput3 = new File(workingDir + File.separator + "MatchTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
            File finalmatchOutput = new File(workingDir + File.separator + "MatchTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

            File finalfacetInput1 = new File(workingDir + File.separator + "FacetTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
            File finalfacetInput2 = new File(workingDir + File.separator + "FacetTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
            File finalfacetInput3 = new File(workingDir + File.separator + "FacetTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
            File finalfacetOutput = new File(workingDir + File.separator + "FacetTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

            for (String cluster : targetClusters) {
                System.out.println("Processing CLuster " + cluster);
                File matchInput1 = new File(workingDir + File.separator + cluster + "_MatchTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchInput2 = new File(workingDir + File.separator + cluster + "_MatchTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchInput3 = new File(workingDir + File.separator + cluster + "_MatchTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchOutput = new File(workingDir + File.separator + cluster + "_MatchTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

                File facetInput1 = new File(workingDir + File.separator + cluster + "_FacetTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetInput2 = new File(workingDir + File.separator + cluster + "_FacetTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetInput3 = new File(workingDir + File.separator + cluster + "_FacetTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetOutput = new File(workingDir + File.separator + cluster + "_FacetTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

                BufferedReader matchbrinput1 = null;
                BufferedReader matchbrinput2 = null;
                BufferedReader matchbrinput3 = null;
                BufferedReader matchbroutput = null;

                BufferedReader facetbrinput1 = null;
                BufferedReader facetbrinput2 = null;
                BufferedReader facetbrinput3 = null;
                BufferedReader facetbroutput = null;

                try {
                    matchbrinput1 = new BufferedReader(new FileReader(matchInput1));
                    matchbrinput2 = new BufferedReader(new FileReader(matchInput2));
                    matchbrinput3 = new BufferedReader(new FileReader(matchInput3));
                    matchbroutput = new BufferedReader(new FileReader(matchOutput));

                    facetbrinput1 = new BufferedReader(new FileReader(facetInput1));
                    facetbrinput2 = new BufferedReader(new FileReader(facetInput2));
                    facetbrinput3 = new BufferedReader(new FileReader(facetInput3));
                    facetbroutput = new BufferedReader(new FileReader(facetOutput));

                    String matchlineinput1;
                    String matchlineinput2;
                    String matchlineinput3;
                    String matchlineoutput;

                    String facetlineinput1;
                    String facetlineinput2;
                    String facetlineinput3;
                    String facetlineoutput;

                    int matchcount = 0;
                    while (((matchlineinput1 = matchbrinput1.readLine()) != null) && ((matchlineinput2 = matchbrinput2.readLine()) != null) &&
                            ((matchlineinput3 = matchbrinput3.readLine()) != null) && ((matchlineoutput = matchbroutput.readLine()) != null)) {
                        if (!matchlineinput1.split(",")[0].equals(matchlineinput2.split(",")[0]) ||
                                !matchlineinput2.split(",")[0].equals(matchlineinput3.split(",")[0]) ||
                                !matchlineinput3.split(",")[0].equals(matchlineoutput.split(",")[0])) {
                            System.out.println(" Matches not Siamese :'(");
                            System.exit(-1);
                        }

                        Utilities.append2CSVFile(matchlineinput1, finalmatchInput1);
                        Utilities.append2CSVFile(matchlineinput2, finalmatchInput2);
                        Utilities.append2CSVFile(matchlineinput3, finalmatchInput3);
                        Utilities.append2CSVFile(matchlineoutput, finalmatchOutput);

                        matchcount++;
                        System.out.println("Maches " + matchcount / 10);
                    }

                    int facetcount = 0;
                    while (((facetlineinput1 = facetbrinput1.readLine()) != null) && ((facetlineinput2 = facetbrinput2.readLine()) != null) &&
                            ((facetlineinput3 = facetbrinput3.readLine()) != null) && ((facetlineoutput = facetbroutput.readLine()) != null)) {
                        if (!facetlineinput1.split(",")[0].equals(facetlineinput2.split(",")[0]) ||
                                !facetlineinput2.split(",")[0].equals(facetlineinput3.split(",")[0]) ||
                                !facetlineinput3.split(",")[0].equals(facetlineoutput.split(",")[0])) {
                            System.out.println("Facets not Siamese :'(");
                            System.exit(-1);
                        }

                        Utilities.append2CSVFile(facetlineinput1, finalfacetInput1);
                        Utilities.append2CSVFile(facetlineinput2, finalfacetInput2);
                        Utilities.append2CSVFile(facetlineinput3, finalfacetInput3);
                        Utilities.append2CSVFile(facetlineoutput, finalfacetOutput);

                        facetcount ++;
                        System.out.println("Facets " + facetcount / 10);
                    }
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    try {
                        if (matchbrinput1 != null) {
                            matchbrinput1.close();
                        }
                        if (matchbrinput2 != null) {
                            matchbrinput2.close();
                        }
                        if (matchbrinput3 != null) {
                            matchbrinput3.close();
                        }
                        if (matchbroutput != null) {
                            matchbroutput.close();
                        }

                        if (facetbrinput1 != null) {
                            facetbrinput1.close();
                        }
                        if (facetbrinput2 != null) {
                            facetbrinput2.close();
                        }
                        if (facetbrinput3 != null) {
                            facetbrinput3.close();
                        }
                        if (facetbroutput != null) {
                            facetbroutput.close();
                        }
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
