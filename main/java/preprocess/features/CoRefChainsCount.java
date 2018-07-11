package preprocess.features;

import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

/**
 * Created by ahmed on 7/8/16.
 */
public class CoRefChainsCount {
    boolean normalized;
    String target;

    public CoRefChainsCount(boolean normalized, String target) {
        this.normalized = normalized;
        this.target = target;
    }

    public Double calculateFeature(TrainingExample obj, DocumentCtx docs) {
        Double value = 0d;
        double val = 0;
        double totalCount = 0;

        try {
            Document rp = docs.getReferenceDoc();
            Document cp = docs.getCitationDoc();

            double rpTotalCount = rp.getAnnotations("CorefChains").size();
            double cpTotalCount = cp.getAnnotations("CorefChains").size();

            AnnotationSet rpCorefChains = rp.getAnnotations("CorefChains").get(obj.getReferenceTextSpan().getLeft(),
                    obj.getReferenceTextSpan().getRight());

            AnnotationSet cpCorefChains = cp.getAnnotations("CorefChains").get(obj.getCitanceTextSpan().getLeft(),
                    obj.getCitanceTextSpan().getRight());

            switch (target) {
                case "RP":
                    for (Annotation annotation : rpCorefChains) {
                        val++;
                    }
                    totalCount = rpTotalCount;
                    break;
                case "CP":
                    for (Annotation annotation : cpCorefChains) {
                        val++;
                    }
                    totalCount = cpTotalCount;
                    break;
                case "BOTH":
                    for (Annotation annotation : rpCorefChains) {
                        val++;
                    }

                    for (Annotation annotation : cpCorefChains) {
                        val++;
                    }
                    totalCount = rpTotalCount + cpTotalCount;
                    break;
            }

            if (normalized) {
                if (totalCount > 0) {
                    value = val / totalCount;
                }
            } else {
                value = val;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return value;
    }
}
