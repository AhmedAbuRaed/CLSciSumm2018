package preprocess;

import gate.*;
import gate.creole.ResourceInstantiationException;
import gate.util.GateException;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.factory.Nd4j;
import postprocess.Utilities;
import preprocess.features.*;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

public class GenerateTrainingIO {
    public static void main(String args[]) {
        if (args.length > 0) {

            String workingDir = args[1];

            File trainingXMLDatasetPath = new File(workingDir + File.separator + "datasets" + File.separator + "training");

            int minTokenLimit = 0;
            int maxTokenLimit = 0;
            int tokensLimitCount = 0;
            int negativeRatio = 0;

            try {
                minTokenLimit = Integer.parseInt(args[2]);
                maxTokenLimit = Integer.parseInt(args[3]);
                tokensLimitCount = Integer.parseInt(args[4]);
                negativeRatio = Integer.parseInt(args[5]);
            } catch (NumberFormatException e) {
                System.err.println("Arguments 2-5 must be integers.");
                System.exit(1);
            }
            String[] targetOptions = args[6].split("\\_");
            String[] targetClusters = Arrays.copyOfRange(targetOptions, 0, targetOptions.length);

            Set<String> stopWords = Utilities.getStopWordsSet(workingDir);

            try {
                Gate.init();
            } catch (GateException e) {
                e.printStackTrace();
            }

            Word2Vec gvec = null;
            Word2Vec aclvec = null;
            Word2Vec bnvec = null;

            //Load Word2Vec models in case it is in the pipeline
            System.out.println("Loading Word2vec Models ...");
            //Get file from resources folder
            File gModel = new File(workingDir + File.separator + "GoogleNews-vectors-negative300.bin.gz");
            gvec = WordVectorSerializer.readWord2VecModel(gModel);

            File aclModel = new File(workingDir + File.separator + "ACL300.txt");
            aclvec = WordVectorSerializer.readWord2VecModel(aclModel);

            File bnModel = new File(workingDir + File.separator + "sw2v_synsets_cbow_wikipedia_vectors.bin");
            bnvec = WordVectorSerializer.readWord2VecModel(bnModel);
            System.out.println("Word2vec Models Loaded ...");

            //initialize featuers
            SentencePosition sidSentencePosition = new SentencePosition("sid");
            SentencePosition ssidSentencePosition = new SentencePosition("ssid");

            WordNetSimilarity jiangconrathWordNetSimilarity = new WordNetSimilarity("jiangconrath");
            WordNetSimilarity lchWordNetSimilarity = new WordNetSimilarity("lch");
            WordNetSimilarity leskWordNetSimilarity = new WordNetSimilarity("lesk");
            WordNetSimilarity linWordNetSimilarity = new WordNetSimilarity("lin");
            WordNetSimilarity pathWordNetSimilarity = new WordNetSimilarity("path");
            WordNetSimilarity resnikWordNetSimilarity = new WordNetSimilarity("resnik");
            WordNetSimilarity wupWordNetSimilarity = new WordNetSimilarity("wup");

            CosineSimilarity lemmaCosineSimilarity = new CosineSimilarity("LEMMA");
            CosineSimilarity babelnetCosineSimilarity = new CosineSimilarity("BABELNET");

            Jaccard jaccard = new Jaccard(8, 2);
            IdfWeightedJaccard idfWeightedJaccard = new IdfWeightedJaccard(8, 2);

            HighestProbFacet highestProbFacet = new HighestProbFacet();

            CitationMarkerCount rpCitationMarkerCount = new CitationMarkerCount(true, "RP");
            CitationMarkerCount bothCitationMarkerCount = new CitationMarkerCount(true, "BOTH");

            CauseAffectExistance rpCauseAffectExistance = new CauseAffectExistance(true, "RP");

            CoRefChainsCount rpCoRefChainsCount = new CoRefChainsCount(true, "RP");
            CoRefChainsCount bothCoRefChainsCount = new CoRefChainsCount(true, "BOTH");

            WorkNouns workNouns = new WorkNouns();

            GAZActionCount gazActionCount = new GAZActionCount();
            GAZConceptCount gazConceptCount = new GAZConceptCount();

            int count = 1;

            for (String clusterFolder : targetClusters) {
                System.out.println("Cluster " + clusterFolder);

                File matchInput1 = new File(workingDir + File.separator + clusterFolder + "_MatchTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + negativeRatio + ".csv");
                File matchInput2 = new File(workingDir + File.separator + clusterFolder + "_MatchTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + negativeRatio + ".csv");
                File matchInput3 = new File(workingDir + File.separator + clusterFolder + "_MatchTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + negativeRatio + ".csv");
                File matchOutput = new File(workingDir + File.separator + clusterFolder + "_MatchTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + "_" + negativeRatio + ".csv");

                File facetInput1 = new File(workingDir + File.separator + clusterFolder + "_FacetTraining_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetInput2 = new File(workingDir + File.separator + clusterFolder + "_FacetTraining_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetInput3 = new File(workingDir + File.separator + clusterFolder + "_FacetTraining_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File facetOutput = new File(workingDir + File.separator + clusterFolder + "_FacetTraining_output_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

                File inputFolder = new File(trainingXMLDatasetPath.getPath() + File.separator + clusterFolder + File.separator + clusterFolder);
                try {
                    Document refDocument = Factory.newDocument(new URL("file:///" + inputFolder.getPath() + File.separator + clusterFolder + ".xml"));
                    AnnotationSet refAnalysis = refDocument.getAnnotations("Analysis");
                    FeatureMap fmRefKindWord = Factory.newFeatureMap();
                    fmRefKindWord.put("kind", "word");
                    AnnotationSet refWordTokens = refAnalysis.get("Token", fmRefKindWord);
                    FeatureMap fmRefKindEntity = Factory.newFeatureMap();
                    fmRefKindEntity.put("kind", "entity");
                    AnnotationSet refSentencesBNEntities = refDocument.getAnnotations("Babelnet").get("Entity", fmRefKindEntity);
                    List<Annotation> refFilteredSentences = Utilities.filterRefSentences(refDocument, minTokenLimit, maxTokenLimit);

                    for (File file : inputFolder.listFiles()) {
                        if (!file.getName().substring(0, file.getName().indexOf(".")).equals(clusterFolder)) {
                            Document citDocument = Factory.newDocument(new URL("file:///" + file.getPath()));
                            AnnotationSet citCitations = citDocument.getAnnotations("CITATIONS");
                            AnnotationSet citAnalysis = citDocument.getAnnotations("Analysis");
                            FeatureMap fmCitKindWord = Factory.newFeatureMap();
                            fmCitKindWord.put("kind", "word");
                            AnnotationSet citWordTokens = citAnalysis.get("Token", fmCitKindWord);
                            FeatureMap fmCitKindEntity = Factory.newFeatureMap();
                            fmCitKindEntity.put("kind", "entity");
                            AnnotationSet citBNEntities = citDocument.getAnnotations("Babelnet").get("Entity", fmCitKindEntity);

                            for (Annotation citCitation : citCitations) {
                                System.out.println("Citation " + citCitation.getFeatures().get("Citance_Number"));
                                int citSentenceTokensAACLCount = 0;
                                int citSentenceTokensBNCount = 0;
                                int citSentenceTokensGNCount = 0;
                                StringBuilder citCitationACLCSVString = new StringBuilder();
                                StringBuilder citCitationBNCSVString = new StringBuilder();
                                StringBuilder citCitationGNCSVString = new StringBuilder();

                                List<Annotation> citCitWordTokens = citWordTokens.get(citCitation.getStartNode().getOffset(), citCitation.getEndNode().getOffset()).inDocumentOrder();
                                List<Annotation> citCitBNEntities = citBNEntities.get(citCitation.getStartNode().getOffset(), citCitation.getEndNode().getOffset()).inDocumentOrder();
                                //ACL channel citation representation
                                for (Annotation token : citCitWordTokens) {
                                    if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                            (citSentenceTokensAACLCount < tokensLimitCount) && aclvec.hasWord(token.getFeatures().get("string").toString().toLowerCase())) {
                                        citCitationACLCSVString.append(Arrays.toString(aclvec.getWordVector(token.getFeatures().get("string").toString().toLowerCase())).replaceAll("[\\[\\] ]", "") + ",");
                                        citSentenceTokensAACLCount++;
                                    }
                                }
                                while (citSentenceTokensAACLCount < tokensLimitCount) {
                                    citCitationACLCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                    citSentenceTokensAACLCount++;
                                }
                                citCitationACLCSVString.deleteCharAt(citCitationACLCSVString.length() - 1);
                                //Babelnet channel citation representation
                                for (Annotation entity : citCitBNEntities) {
                                    if ((citSentenceTokensBNCount < tokensLimitCount) && bnvec.hasWord(entity.getFeatures().get("synsetID").toString())) {
                                        citCitationBNCSVString.append(Arrays.toString(bnvec.getWordVector(entity.getFeatures().get("synsetID").toString())).replaceAll("[\\[\\] ]", "") + ",");
                                        citSentenceTokensBNCount++;
                                    }
                                }
                                while (citSentenceTokensBNCount < tokensLimitCount) {
                                    citCitationBNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                    citSentenceTokensBNCount++;
                                }
                                citCitationBNCSVString.deleteCharAt(citCitationBNCSVString.length() - 1);
                                //Google News channel citation representation
                                for (Annotation token : citCitWordTokens) {
                                    if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                            (citSentenceTokensGNCount < tokensLimitCount) && gvec.hasWord(token.getFeatures().get("string").toString().toLowerCase())) {
                                        citCitationGNCSVString.append(Arrays.toString(gvec.getWordVector(token.getFeatures().get("string").toString())).replaceAll("[\\[\\] ]", "") + ",");
                                        citSentenceTokensGNCount++;
                                    }
                                }
                                while (citSentenceTokensGNCount < tokensLimitCount) {
                                    citCitationGNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                    citSentenceTokensGNCount++;
                                }
                                citCitationGNCSVString.deleteCharAt(citCitationGNCSVString.length() - 1);

                                //set positive instances
                                FeatureMap refIDfm = Factory.newFeatureMap();
                                refIDfm.put("id", citCitation.getFeatures().get("id").toString());
                                AnnotationSet refpositiveInstances = refDocument.getAnnotations("REFERENCES").get(citCitation.getType(), refIDfm);

                                for (Annotation refpositiveInstance : refpositiveInstances) {
                                    int refSentenceTokensACLCount = 0;
                                    int refSentenceTokensBNCount = 0;
                                    int refSentenceTokensGNCount = 0;
                                    StringBuilder refSentenceACLCSVString = new StringBuilder();
                                    StringBuilder refSentenceBNCSVString = new StringBuilder();
                                    StringBuilder refSentenceGNCSVString = new StringBuilder();

                                    List<Annotation> refSentenceWordTokens = refWordTokens.get(refpositiveInstance.getStartNode().getOffset(), refpositiveInstance.getEndNode().getOffset()).inDocumentOrder();
                                    List<Annotation> refSentenceBNEntities = refSentencesBNEntities.get(refpositiveInstance.getStartNode().getOffset(), refpositiveInstance.getEndNode().getOffset()).inDocumentOrder();

                                    //ACL channel reference sentence representation
                                    for (Annotation token : refSentenceWordTokens) {
                                        if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                                (refSentenceTokensACLCount < tokensLimitCount) && aclvec.hasWord(token.getFeatures().get("string").toString().toLowerCase())) {
                                            refSentenceACLCSVString.append(Arrays.toString(aclvec.getWordVector(token.getFeatures().get("string").toString().toLowerCase())).replaceAll("[\\[\\] ]", "") + ",");
                                            refSentenceTokensACLCount++;
                                        }
                                    }
                                    while (refSentenceTokensACLCount < tokensLimitCount) {
                                        refSentenceACLCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                        refSentenceTokensACLCount++;
                                    }
                                    refSentenceACLCSVString.deleteCharAt(refSentenceACLCSVString.length() - 1);
                                    //Babelnet channel reference sentence representation
                                    for (Annotation entity : refSentenceBNEntities) {
                                        if ((refSentenceTokensBNCount < tokensLimitCount) && bnvec.hasWord(entity.getFeatures().get("synsetID").toString())) {
                                            refSentenceBNCSVString.append(Arrays.toString(bnvec.getWordVector(entity.getFeatures().get("synsetID").toString())).replaceAll("[\\[\\] ]", "") + ",");
                                            refSentenceTokensBNCount++;
                                        }
                                    }
                                    while (refSentenceTokensBNCount < tokensLimitCount) {
                                        refSentenceBNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                        refSentenceTokensBNCount++;
                                    }
                                    refSentenceBNCSVString.deleteCharAt(refSentenceBNCSVString.length() - 1);
                                    //Google News channel reference sentence representation
                                    for (Annotation token : refSentenceWordTokens) {
                                        if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                                (refSentenceTokensGNCount < tokensLimitCount) && gvec.hasWord(token.getFeatures().get("string").toString().toLowerCase())) {
                                            refSentenceGNCSVString.append(Arrays.toString(gvec.getWordVector(token.getFeatures().get("string").toString().toLowerCase())).replaceAll("[\\[\\] ]", "") + ",");
                                            refSentenceTokensGNCount++;
                                        }
                                    }
                                    while (refSentenceTokensGNCount < tokensLimitCount) {
                                        refSentenceGNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                        refSentenceTokensGNCount++;
                                    }
                                    refSentenceGNCSVString.deleteCharAt(refSentenceGNCSVString.length() - 1);

                                    //generating the third input file (features)
                                    DocumentCtx documentCtx = new DocumentCtx(citDocument, refDocument);
                                    TrainingExample trainingExample = new TrainingExample(Pair.of(citCitation.getStartNode().getOffset(),
                                            citCitation.getEndNode().getOffset()),
                                            Pair.of(refpositiveInstance.getStartNode().getOffset(), refpositiveInstance.getEndNode().getOffset()));

                                    StringBuilder features = new StringBuilder();

                                    features.append(sidSentencePosition.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(ssidSentencePosition.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(jiangconrathWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(lchWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(leskWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(linWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(pathWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(resnikWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(wupWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(lemmaCosineSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(babelnetCosineSimilarity.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(jaccard.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(idfWeightedJaccard.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(highestProbFacet.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(rpCitationMarkerCount.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(bothCitationMarkerCount.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(rpCauseAffectExistance.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(rpCoRefChainsCount.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(bothCoRefChainsCount.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(workNouns.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(gazActionCount.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(gazConceptCount.calculateFeature(trainingExample, documentCtx));

                                    Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            citCitationACLCSVString.toString() + "," + citCitationBNCSVString.toString() + "," + citCitationGNCSVString.toString(), matchInput1);
                                    Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            refSentenceACLCSVString.toString() + "," + refSentenceBNCSVString.toString() + "," + refSentenceGNCSVString.toString(), matchInput2);
                                    Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            features.toString(), matchInput3);
                                    Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            "1", matchOutput);

                                    String facetString = citCitation.getFeatures().get("Discourse_Facet").toString();
                                    int facetValue = -1;
                                    for (String facet : facetString.split(",")) {
                                        switch (facet) {
                                            case "Aim_Citation":
                                            case "Aim_Facet":
                                                facetValue = 0;
                                                break;
                                            case "Hypothesis_Citation":
                                            case "Hypothesis_Facet":
                                                facetValue = 1;
                                                break;
                                            case "Method_Citation":
                                            case "Method_Facet":
                                                facetValue = 2;
                                                break;
                                            case "Implication_Citation":
                                            case "Implication_Facet":
                                                facetValue = 3;
                                                break;
                                            case "Results_Citation":
                                            case "Results_Facet":
                                                facetValue = 4;
                                                break;
                                        }

                                        Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                                citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                                citCitationACLCSVString.toString() + "," + citCitationBNCSVString.toString() + "," + citCitationGNCSVString.toString(), facetInput1);
                                        Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                                citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                                refSentenceACLCSVString.toString() + "," + refSentenceBNCSVString.toString() + "," + refSentenceGNCSVString.toString(), facetInput2);
                                        Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                                citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                                features.toString(), facetInput3);
                                        Utilities.append2CSVFile(count + "_" + refpositiveInstance.getFeatures().get("Reference_Offset").toString() + "_" + refpositiveInstance.getFeatures().get("Reference_Article").toString() + "_" +
                                                citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                                String.valueOf(facetValue), facetOutput);
                                    }
                                    count++;
                                }

                                int limit = refpositiveInstances.size() * (negativeRatio / 1);
                                int negativeCount = 0;

                                for (Annotation refSentence : refFilteredSentences) {
                                    if (negativeCount == limit) {
                                        break;
                                    }
                                    int refSentenceTokensACLCount = 0;
                                    int refSentenceTokensBNCount = 0;
                                    int refSentenceTokensGNCount = 0;
                                    StringBuilder refSentenceACLCSVString = new StringBuilder();
                                    StringBuilder refSentenceBNCSVString = new StringBuilder();
                                    StringBuilder refSentenceGNCSVString = new StringBuilder();

                                    List<Annotation> refSentenceWordTokens = refWordTokens.get(refSentence.getStartNode().getOffset(), refSentence.getEndNode().getOffset()).inDocumentOrder();
                                    List<Annotation> refSentenceBNEntities = refSentencesBNEntities.get(refSentence.getStartNode().getOffset(), refSentence.getEndNode().getOffset()).inDocumentOrder();

                                    //ACL channel reference sentence representation
                                    for (Annotation token : refSentenceWordTokens) {
                                        if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                                (refSentenceTokensACLCount < tokensLimitCount) && aclvec.hasWord(token.getFeatures().get("string").toString().toLowerCase())) {
                                            refSentenceACLCSVString.append(Arrays.toString(aclvec.getWordVector(token.getFeatures().get("string").toString().toLowerCase())).replaceAll("[\\[\\] ]", "") + ",");
                                            refSentenceTokensACLCount++;
                                        }
                                    }
                                    while (refSentenceTokensACLCount < tokensLimitCount) {
                                        refSentenceACLCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                        refSentenceTokensACLCount++;
                                    }
                                    refSentenceACLCSVString.deleteCharAt(refSentenceACLCSVString.length() - 1);
                                    //Babelnet channel reference sentence representation
                                    for (Annotation entity : refSentenceBNEntities) {
                                        if ((refSentenceTokensBNCount < tokensLimitCount) && bnvec.hasWord(entity.getFeatures().get("synsetID").toString())) {
                                            refSentenceBNCSVString.append(Arrays.toString(bnvec.getWordVector(entity.getFeatures().get("synsetID").toString())).replaceAll("[\\[\\] ]", "") + ",");
                                            refSentenceTokensBNCount++;
                                        }
                                    }
                                    while (refSentenceTokensBNCount < tokensLimitCount) {
                                        refSentenceBNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                        refSentenceTokensBNCount++;
                                    }
                                    refSentenceBNCSVString.deleteCharAt(refSentenceBNCSVString.length() - 1);
                                    //Google News channel reference sentence representation
                                    for (Annotation token : refSentenceWordTokens) {
                                        if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                                (refSentenceTokensGNCount < tokensLimitCount) && gvec.hasWord(token.getFeatures().get("string").toString().toLowerCase())) {
                                            refSentenceGNCSVString.append(Arrays.toString(gvec.getWordVector(token.getFeatures().get("string").toString().toLowerCase())).replaceAll("[\\[\\] ]", "") + ",");
                                            refSentenceTokensGNCount++;
                                        }
                                    }
                                    while (refSentenceTokensGNCount < tokensLimitCount) {
                                        refSentenceGNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                        refSentenceTokensGNCount++;
                                    }
                                    refSentenceGNCSVString.deleteCharAt(refSentenceGNCSVString.length() - 1);

                                    //generating the third input file (features)
                                    DocumentCtx documentCtx = new DocumentCtx(citDocument, refDocument);
                                    TrainingExample trainingExample = new TrainingExample(Pair.of(citCitation.getStartNode().getOffset(),
                                            citCitation.getEndNode().getOffset()),
                                            Pair.of(refSentence.getStartNode().getOffset(), refSentence.getEndNode().getOffset()));

                                    StringBuilder features = new StringBuilder();

                                    features.append(sidSentencePosition.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(ssidSentencePosition.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(jiangconrathWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(lchWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(leskWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(linWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(pathWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(resnikWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(wupWordNetSimilarity.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(lemmaCosineSimilarity.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(babelnetCosineSimilarity.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(jaccard.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(idfWeightedJaccard.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(highestProbFacet.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(rpCitationMarkerCount.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(bothCitationMarkerCount.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(rpCauseAffectExistance.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(rpCoRefChainsCount.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(bothCoRefChainsCount.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(workNouns.calculateFeature(trainingExample, documentCtx) + ",");

                                    features.append(gazActionCount.calculateFeature(trainingExample, documentCtx) + ",");
                                    features.append(gazConceptCount.calculateFeature(trainingExample, documentCtx));

                                    Utilities.append2CSVFile(count + "_" + refSentence.getFeatures().get("sid").toString() + "_" + clusterFolder + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            citCitationACLCSVString.toString() + "," + citCitationBNCSVString.toString() + "," + citCitationGNCSVString.toString(), matchInput1);
                                    Utilities.append2CSVFile(count + "_" + refSentence.getFeatures().get("sid").toString() + "_" + clusterFolder + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            refSentenceACLCSVString.toString() + "," + refSentenceBNCSVString.toString() + "," + refSentenceGNCSVString.toString(), matchInput2);
                                    Utilities.append2CSVFile(count + "_" + refSentence.getFeatures().get("sid").toString() + "_" + clusterFolder + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            features.toString(), matchInput3);
                                    Utilities.append2CSVFile(count + "_" + refSentence.getFeatures().get("sid").toString() + "_" + clusterFolder + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            "0", matchOutput);

                                    negativeCount++;
                                    count++;
                                }
                                Collections.shuffle(refFilteredSentences);
                            }
                            Factory.deleteResource(citDocument);
                        }
                    }
                    Factory.deleteResource(refDocument);
                } catch (MalformedURLException e) {
                    e.printStackTrace();
                } catch (ResourceInstantiationException e) {
                    e.printStackTrace();
                }
            }
            System.out.println("DONE ...");
        } else {
            System.out.println("No Arguments !!!");
        }
    }
}

