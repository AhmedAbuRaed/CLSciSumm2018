package preprocess;

import com.twelvemonkeys.util.LinkedSet;
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

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class GenerateTestingAsTextSpansIO {
    public static void main(String args[]) {
        if (args.length > 0) {

            String workingDir = args[1];

            File testingXMLDatasetPath = new File(workingDir + File.separator + "datasets" + File.separator + "testing");
            int minTokenLimit = 0;
            int maxTokenLimit = 0;
            int tokensLimitCount = 0;

            try {
                minTokenLimit = Integer.parseInt(args[2]);
                maxTokenLimit = Integer.parseInt(args[3]);
                tokensLimitCount = Integer.parseInt(args[4]);
            } catch (NumberFormatException e) {
                System.err.println("Arguments 2-4 must be integers.");
                System.exit(1);
            }

            String[] targetPapersStrings = Arrays.copyOfRange(args, 4, args.length);

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

            for (String clusterFolder : targetPapersStrings) {

                System.out.println("Cluster " + clusterFolder);
                int count = 1;
                File input1 = new File(workingDir + File.separator + clusterFolder + "_Testing_input1_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File input2 = new File(workingDir + File.separator + clusterFolder + "_Testing_input2_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");
                File input3 = new File(workingDir + File.separator + clusterFolder + "_Testing_input3_" + minTokenLimit + "_" + maxTokenLimit + "_" + tokensLimitCount + ".csv");

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

                File inputFolder = new File(workingDir + File.separator + "datasets" + File.separator + "testing" + File.separator + clusterFolder + File.separator + clusterFolder);
                try {
                    Document refDocument = Factory.newDocument(new URL("file:///" + inputFolder.getPath() + File.separator + clusterFolder + ".xml"));
                    AnnotationSet refAnalysis = refDocument.getAnnotations("Analysis");
                    FeatureMap fmRefKindWord = Factory.newFeatureMap();
                    fmRefKindWord.put("kind", "word");
                    AnnotationSet refWordTokens = refAnalysis.get("Token", fmRefKindWord);
                    FeatureMap fmRefKindEntity = Factory.newFeatureMap();
                    fmRefKindEntity.put("kind", "entity");
                    AnnotationSet refSentencesBNEntities = refDocument.getAnnotations("Babelnet").get("Entity", fmRefKindEntity);
                    AnnotationSet refTotalSentences = refAnalysis.get("Sentence");

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
                            Set<String> processedFilteredCitations = new LinkedSet<String>();

                            for (Annotation citCitation : citCitations) {
                                if (processedFilteredCitations.contains(citCitation.getFeatures().get("id").toString())) {
                                    continue;
                                }
                                System.out.println("Citation " + citCitation.getFeatures().get("Citance_Number"));
                                int citSentenceTokensAACLCount = 0;
                                int citSentenceTokensBNCount = 0;
                                int citSentenceTokensGNCount = 0;
                                StringBuilder citCitationACLCSVString = new StringBuilder();
                                StringBuilder citCitationBNCSVString = new StringBuilder();
                                StringBuilder citCitationGNCSVString = new StringBuilder();

                                FeatureMap citIDfm = Factory.newFeatureMap();
                                citIDfm.put("id", citCitation.getFeatures().get("id").toString());
                                List<Annotation> filteredcitCitations = citDocument.getAnnotations("CITATIONS").get(citCitation.getType(), citIDfm).inDocumentOrder();

                                ////CITATIONS
                                long filteredCitStartOffset = filteredcitCitations.get(0).getStartNode().getOffset() - /*this is a hack to increase context*/ 50l;
                                long filteredCitEndOffset = filteredcitCitations.get(filteredcitCitations.size() - 1).getEndNode().getOffset() + /*this is a hack to increase context*/ 150l;

                                if (filteredCitStartOffset < 0)
                                    filteredCitStartOffset = 0;

                                if (filteredCitEndOffset > citDocument.getContent().size())
                                    filteredCitEndOffset = citDocument.getContent().size();

                                List<Annotation> filteredCitWordTokens = citWordTokens.get(filteredCitStartOffset, filteredCitEndOffset).inDocumentOrder();
                                List<Annotation> filteredCitBNEntities = citBNEntities.get(filteredCitStartOffset, filteredCitEndOffset).inDocumentOrder();

                                //ACL channel citation representation
                                for (Annotation token : filteredCitWordTokens) {
                                    AnnotationSet citMarkers = citDocument.getAnnotations("Analysis").get("CitMarker").get(token.getStartNode().getOffset(), token.getEndNode().getOffset());
                                    if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                            (citSentenceTokensAACLCount < tokensLimitCount) && aclvec.hasWord(token.getFeatures().get("string").toString().toLowerCase()) &&
                                            (citMarkers.size() == 0)) {
                                        citCitationACLCSVString.append(Arrays.toString(aclvec.getWordVector(token.getFeatures().get("string").toString().toLowerCase())).replaceAll("[\\[\\] ]", "") + ",");
                                        citSentenceTokensAACLCount++;
                                    }
                                }

                                //Babelnet channel citation representation
                                for (Annotation entity : filteredCitBNEntities) {
                                    AnnotationSet citMarkers = citDocument.getAnnotations("Analysis").get("CitMarker").get(entity.getStartNode().getOffset(), entity.getEndNode().getOffset());
                                    if ((citSentenceTokensBNCount < tokensLimitCount) && bnvec.hasWord(entity.getFeatures().get("synsetID").toString()) && (citMarkers.size() == 0)) {
                                        citCitationBNCSVString.append(Arrays.toString(bnvec.getWordVector(entity.getFeatures().get("synsetID").toString())).replaceAll("[\\[\\] ]", "") + ",");
                                        citSentenceTokensBNCount++;
                                    }
                                }

                                //Google News channel citation representation
                                for (Annotation token : filteredCitWordTokens) {
                                    AnnotationSet citMarkers = citDocument.getAnnotations("Analysis").get("CitMarker").get(token.getStartNode().getOffset(), token.getEndNode().getOffset());
                                    if ((!stopWords.contains(token.getFeatures().get("string").toString().toLowerCase())) && (token.getFeatures().get("string").toString().length() > 1) &&
                                            (citSentenceTokensGNCount < tokensLimitCount) && gvec.hasWord(token.getFeatures().get("string").toString().toLowerCase()) &&
                                            (citMarkers.size() == 0)) {
                                        citCitationGNCSVString.append(Arrays.toString(gvec.getWordVector(token.getFeatures().get("string").toString())).replaceAll("[\\[\\] ]", "") + ",");
                                        citSentenceTokensGNCount++;
                                    }
                                }

                                //fill Zeros if not completed
                                while (citSentenceTokensAACLCount < tokensLimitCount) {
                                    citCitationACLCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                    citSentenceTokensAACLCount++;
                                }
                                while (citSentenceTokensBNCount < tokensLimitCount) {
                                    citCitationBNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                    citSentenceTokensBNCount++;
                                }
                                while (citSentenceTokensGNCount < tokensLimitCount) {
                                    citCitationGNCSVString.append(Nd4j.zeros(300).toString().replaceAll("[\\[\\] ]", "") + ",");
                                    citSentenceTokensGNCount++;
                                }

                                //delete the last comma after filling all tokens
                                citCitationACLCSVString.deleteCharAt(citCitationACLCSVString.length() - 1);
                                citCitationBNCSVString.deleteCharAt(citCitationBNCSVString.length() - 1);
                                citCitationGNCSVString.deleteCharAt(citCitationGNCSVString.length() - 1);

                                for (Annotation refSentence : refTotalSentences) {
                                    int refSentenceTokensACLCount = 0;
                                    int refSentenceTokensBNCount = 0;
                                    int refSentenceTokensGNCount = 0;
                                    StringBuilder refSentenceACLCSVString = new StringBuilder();
                                    StringBuilder refSentenceBNCSVString = new StringBuilder();
                                    StringBuilder refSentenceGNCSVString = new StringBuilder();

                                    long refSentenceStartOffset = refSentence.getStartNode().getOffset() - /*this is a hack to increase context*/ 50l;
                                    long refSentenceEndOffset = refSentence.getEndNode().getOffset() + /*this is a hack to increase context*/ 150l;

                                    if (refSentenceStartOffset < 0)
                                        refSentenceStartOffset = 0;

                                    if (refSentenceEndOffset > refDocument.getContent().size())
                                        refSentenceEndOffset = refDocument.getContent().size();

                                    List<Annotation> refSentenceWordTokens = refWordTokens.get(refSentenceStartOffset, refSentenceEndOffset).inDocumentOrder();
                                    List<Annotation> refSentenceBNEntities = refSentencesBNEntities.get(refSentenceStartOffset, refSentenceEndOffset).inDocumentOrder();

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
                                    TrainingExample trainingExample = new TrainingExample(Pair.of(filteredCitStartOffset, filteredCitEndOffset),
                                            Pair.of(refSentence.getStartNode().getOffset(), refSentenceEndOffset));

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
                                            citCitationACLCSVString.toString() + "," + citCitationBNCSVString.toString() + "," + citCitationGNCSVString.toString(), input1);
                                    Utilities.append2CSVFile(count + "_" + refSentence.getFeatures().get("sid").toString() + "_" + clusterFolder + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            refSentenceACLCSVString.toString() + "," + refSentenceBNCSVString.toString() + "," + refSentenceGNCSVString.toString(), input2);
                                    Utilities.append2CSVFile(count + "_" + refSentence.getFeatures().get("sid").toString() + "_" + clusterFolder + "_" +
                                            citCitation.getFeatures().get("Citing_Article").toString() + "_" + citCitation.getFeatures().get("Citation_Offset").toString() + "," +
                                            features.toString(), input3);
                                    count++;
                                }
                                processedFilteredCitations.add(citCitation.getFeatures().get("id").toString());

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
        } else {
            System.out.println("No Arguments !!!");
        }
    }
}
