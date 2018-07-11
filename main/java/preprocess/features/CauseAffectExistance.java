package preprocess.features;

import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

/**
 * Created by ahmed on 7/8/16.
 */
public class CauseAffectExistance {
    String target;

    public CauseAffectExistance(boolean normalized, String target) {
        this.target = target;
    }

    public Double calculateFeature(TrainingExample obj, DocumentCtx docs) {
        Double value = 0d;

        try
        {
            switch (target) {
                case "RP":
                    Document rp = docs.getReferenceDoc();

                    AnnotationSet rpCause = rp.getAnnotations("Causality").get("CAUSE").get(obj.getReferenceTextSpan().getLeft(),
                            obj.getReferenceTextSpan().getRight());

                    AnnotationSet rpEffect = rp.getAnnotations("Causality").get("EFFECT").get(obj.getReferenceTextSpan().getLeft(),
                            obj.getReferenceTextSpan().getRight());

                    if ((rpCause.size() > 0 || rpEffect.size() > 0)) {
                        value = 1d;
                    }
                    break;
                case "CP":
                    Document cp = docs.getCitationDoc();

                    AnnotationSet cpCause = cp.getAnnotations("Causality").get("CAUSE").get(obj.getCitanceTextSpan().getLeft(),
                            obj.getCitanceTextSpan().getRight());

                    AnnotationSet cpEffect = cp.getAnnotations("Causality").get("EFFECT").get(obj.getCitanceTextSpan().getLeft(),
                            obj.getCitanceTextSpan().getRight());

                    if (cpCause.size() > 0 || cpEffect.size() > 0) {
                        value = 1d;
                    }
                    break;
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        return value;
    }
}
