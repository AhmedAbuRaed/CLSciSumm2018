package preprocess.features;

import gate.Annotation;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

import java.util.List;

/**
 * Created by ahmed on 5/11/2016.
 */
public class SentencePosition {
    String positionID;

    public SentencePosition(String positionID) {
        this.positionID = positionID;
    }

    public Double calculateFeature(TrainingExample obj, DocumentCtx docs) {
        Double value = (0d);
        List<Annotation> refSentences = docs.getReferenceDoc().getAnnotations("Analysis").get("Sentence").get(obj.getReferenceTextSpan().getLeft(), obj.getReferenceTextSpan().getRight()).inDocumentOrder();
        Annotation fisrtRefSentence = refSentences.get(0);
        Annotation lastRefSentence = refSentences.get(refSentences.size() - 1);

        if (fisrtRefSentence.getFeatures().containsKey(positionID) && lastRefSentence.getFeatures().containsKey(positionID)) {
            if (fisrtRefSentence.getFeatures().get(positionID) != null && lastRefSentence.getFeatures().get(positionID) != null &&
                    !fisrtRefSentence.getFeatures().get(positionID).equals("") && !lastRefSentence.getFeatures().get(positionID).equals("")) {
                Integer refSentenceID;
                if (refSentences.size() > 1) {
                    refSentenceID = ((Integer.valueOf((String) fisrtRefSentence.getFeatures().get(positionID)) +
                            Integer.valueOf((String) lastRefSentence.getFeatures().get(positionID))) /
                            refSentences.size());
                } else {
                    refSentenceID = ((Integer.valueOf((String) fisrtRefSentence.getFeatures().get(positionID))));
                }

                Double sentenceIDValue = refSentenceID + 1d;
                if (!sentenceIDValue.isInfinite() && !sentenceIDValue.isNaN() && sentenceIDValue > 0) {
                    value = (1d / sentenceIDValue);
                }
            }
        }

        return value;
    }
}
