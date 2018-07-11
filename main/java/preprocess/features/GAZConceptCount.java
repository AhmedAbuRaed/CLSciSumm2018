package preprocess.features;

import gate.Annotation;
import gate.Document;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

public class GAZConceptCount  {
    public Double calculateFeature(TrainingExample obj, DocumentCtx ctx) {
        Double value = (0d);
        Document document = ctx.getReferenceDoc();
        Double count = 0d;

        for (Annotation annotation : document.getAnnotations("Analysis").get("Lookup").get(obj.getReferenceTextSpan().getLeft(), obj.getReferenceTextSpan().getRight())) {
            if (annotation.getFeatures().get("majorType").toString().equals("concept_lexicon")) {
                count++;
            }
        }
        if (count == 0) {
            value = 1d;
        } else {
            value = (1 / (count + 1));
        }
        return value;
    }
}
