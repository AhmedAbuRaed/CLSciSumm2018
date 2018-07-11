package preprocess.features;

import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

public class HighestProbFacet {

    public Double calculateFeature(TrainingExample obj, DocumentCtx ctx) {
        Double value = 0d;

        Document rp = ctx.getReferenceDoc();

        AnnotationSet rpSentenceProbabilities = rp.getAnnotations("Analysis").get("Sentence_LOA").get(obj.getReferenceTextSpan().getLeft(),
                obj.getReferenceTextSpan().getRight());

        if (rpSentenceProbabilities.size() > 0) {
            Double max = 0d;
            String facet = "Unspecified";
            Annotation cpSentence = rpSentenceProbabilities.iterator().next();
            for (Object key : cpSentence.getFeatures().keySet()) {
                if (key.toString().startsWith("PROB_DRI")) {
                    if (cpSentence.getFeatures().get(key) != null && max < (Double) cpSentence.getFeatures().get(key)) {
                        max = (Double) cpSentence.getFeatures().get(key);
                        facet = key.toString().substring(key.toString().lastIndexOf("_") + 1);
                    }
                }
            }
            switch (facet) {
                case "Approach":
                    value = (1d);
                    break;
                case "Background":
                    value = (2d);
                    break;
                case "Challenge":
                    value = (3d);
                    break;
                case "FutureWork":
                    value = (4d);
                    break;
                case "Outcome":
                    value = (5d);
                    break;
                case "Unspecified":
                    value = (0d);
                    break;
            }
        }

        return value;
    }
}
