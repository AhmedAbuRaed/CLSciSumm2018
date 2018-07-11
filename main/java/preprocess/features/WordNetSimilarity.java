package preprocess.features;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.impl.*;
import edu.cmu.lti.ws4j.util.WS4JConfiguration;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by ahmed on 5/10/2016.
 */
public class WordNetSimilarity {
    boolean normalized;
    String similarityMeasure;
    ILexicalDatabase db;

    public WordNetSimilarity(String similarityMeasure) {
        this.similarityMeasure = similarityMeasure;
        db = new NictWordNet();
    }

    public Double calculateFeature(TrainingExample obj, DocumentCtx docs) {
        Double value = 0d;

        int count = 0;
        double sum = 0d;

        Document cp = docs.getCitationDoc();
        Document rp = docs.getReferenceDoc();

        List<Annotation> cpTokens = cp.getAnnotations("Analysis").get("Token").get(obj.getCitanceTextSpan().getLeft(),
                obj.getCitanceTextSpan().getRight()).inDocumentOrder();

        List<Annotation> rpTokens = rp.getAnnotations("Analysis").get("Token").get(obj.getReferenceTextSpan().getLeft(),
                obj.getReferenceTextSpan().getRight()).inDocumentOrder();

        WS4JConfiguration.getInstance().setMFS(true);


        List<String> cpTokensList = new ArrayList<String>();
        for (int i = 0; i < cpTokens.size(); i++) {
            cpTokensList.add(cpTokens.get(i).getFeatures().get("string").toString());
        }
        List<String> rpTokensList = new ArrayList<String>();
        for (int i = 0; i < rpTokens.size(); i++) {
            rpTokensList.add(rpTokens.get(i).getFeatures().get("string").toString());
        }

        double[][] val = new double[cpTokensList.size()][rpTokensList.size()];

        switch (similarityMeasure) {
            case "jiangconrath":

                val = new JiangConrath(db).getNormalizedSimilarityMatrix(cpTokensList.toArray(new String[0]), rpTokensList.toArray(new String[0]));
                break;
            case "lch":
                val = new LeacockChodorow(db).getNormalizedSimilarityMatrix(cpTokensList.toArray(new String[0]), rpTokensList.toArray(new String[0]));
                break;
            case "lesk":
                val = new Lesk(db).getNormalizedSimilarityMatrix(cpTokensList.toArray(new String[0]), rpTokensList.toArray(new String[0]));
                break;
            case "lin":
                val = new Lin(db).getNormalizedSimilarityMatrix(cpTokensList.toArray(new String[0]), rpTokensList.toArray(new String[0]));
                break;
            case "path":
                val = new Path(db).getNormalizedSimilarityMatrix(cpTokensList.toArray(new String[0]), rpTokensList.toArray(new String[0]));
                break;
            case "resnik":
                val = new Resnik(db).getNormalizedSimilarityMatrix(cpTokensList.toArray(new String[0]), rpTokensList.toArray(new String[0]));
                break;
            case "wup":
                val = new WuPalmer(db).getNormalizedSimilarityMatrix(cpTokensList.toArray(new String[0]), rpTokensList.toArray(new String[0]));
                break;
        }

        value = Arrays.stream(val).flatMapToDouble(Arrays::stream).average().getAsDouble();

        return value;
    }
}
