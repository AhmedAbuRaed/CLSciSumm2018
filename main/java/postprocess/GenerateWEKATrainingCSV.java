package postprocess;

import java.io.*;
import java.util.Arrays;

public class GenerateWEKATrainingCSV {
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

            File output = new File(workingDir + File.separator + "IO" + File.separator + "Training_wekainput_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

            for (String cluster : targetClusters) {
                System.out.println("Cluster " + cluster);

                File matchInput1 = new File(workingDir + File.separator + "IO" + File.separator + cluster +  "_MatchTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchInput2 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchInput3 = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File matchOutput = new File(workingDir + File.separator + "IO" + File.separator + cluster + "_MatchTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

                BufferedReader brinput2 = null;
                BufferedReader brinput3 = null;
                BufferedReader broutput = null;

                try {
                    brinput2 = new BufferedReader(new FileReader(matchInput2));
                    brinput3 = new BufferedReader(new FileReader(matchInput3));
                    broutput = new BufferedReader(new FileReader(matchOutput));

                    String lineinput2;
                    String lineinput3;
                    String lineoutput;

                    while ((((lineinput2 = brinput2.readLine()) != null) && ((lineinput3 = brinput3.readLine()) != null) && ((lineoutput = broutput.readLine()) != null))) {
                        if ((!lineinput2.split(",")[0].equals(lineinput3.split(",")[0])) && !lineinput3.split(",")[0].equals(lineoutput.split(",")[0])) {
                            System.out.println("The inputs are not Siamese :'(");
                            System.exit(-1);
                        }

                        PrintWriter pwMatch = new PrintWriter(new FileWriter(output, true));
                        pwMatch.println(String.join(",", Arrays.copyOfRange(lineinput2.split(","), 9001, lineinput2.split(",").length)) + "," +
                                String.join(",", Arrays.copyOfRange(lineinput3.split(","), 1, lineinput3.split(",").length)) + "," +
                                lineoutput.split(",")[1]);
                        pwMatch.flush();
                        pwMatch.close();

                    }
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    try {
                        if (brinput2 != null) {
                            brinput2.close();
                        }
                        if (brinput3 != null) {
                            brinput3.close();
                        }
                        if (broutput != null) {
                            broutput.close();
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
