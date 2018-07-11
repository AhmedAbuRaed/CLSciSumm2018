package preprocess.features;

import gate.*;
import postprocess.Utilities;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

import java.util.List;

/**
 * Created by ahmed on 5/17/2016.
 */
public class CosineSimilarity {
    String cosineType;

    public CosineSimilarity(String cosineType) {
        this.cosineType = cosineType;
    }

    public Double calculateFeature(TrainingExample obj, DocumentCtx docs) {
        Double value = 0d;

        try {
            Document cp = docs.getCitationDoc();
            Document rp = docs.getReferenceDoc();

            List<Annotation> cpVictorNorm = cp.getAnnotations("Analysis").get("Vector_Norm").get(obj.getCitanceTextSpan().getLeft(),
                    obj.getCitanceTextSpan().getRight()).inDocumentOrder();
            FeatureMap cpVictorNormFM = Factory.newFeatureMap();
            for(int i=0; i<cpVictorNorm.size(); i++)
            {
                if (i == 0) {
                    cpVictorNormFM = cpVictorNorm.get(i).getFeatures();
                } else {
                    cpVictorNormFM = Utilities.combineNormalizedVectors(cpVictorNormFM, cpVictorNorm.get(i).getFeatures());
                }
            }

            List<Annotation> rpVictorNorm = rp.getAnnotations("Analysis").get("Vector_Norm").get(obj.getReferenceTextSpan().getLeft(),
                    obj.getReferenceTextSpan().getRight()).inDocumentOrder();
            FeatureMap rpVictorNormFM = Factory.newFeatureMap();
            for(int i=0; i<rpVictorNorm.size(); i++)
            {
                if (i == 0) {
                    rpVictorNormFM = rpVictorNorm.get(i).getFeatures();
                } else {
                    rpVictorNormFM = Utilities.combineNormalizedVectors(rpVictorNormFM, rpVictorNorm.get(i).getFeatures());
                }
            }

            List<Annotation> cpBabelnetVictorNorm = cp.getAnnotations("Babelnet").get("BNVector_Norm").get(obj.getCitanceTextSpan().getLeft(),
                    obj.getCitanceTextSpan().getRight()).inDocumentOrder();
            FeatureMap cpBabelnetVictorNormFM = Factory.newFeatureMap();
            for(int i=0; i<cpBabelnetVictorNorm.size(); i++)
            {
                if (i == 0) {
                    cpBabelnetVictorNormFM = cpBabelnetVictorNorm.get(i).getFeatures();
                } else {
                    cpBabelnetVictorNormFM = Utilities.combineNormalizedVectors(cpBabelnetVictorNormFM, cpBabelnetVictorNorm.get(i).getFeatures());
                }
            }

            List<Annotation> rpBabelnetVictorNorm = rp.getAnnotations("Babelnet").get("BNVector_Norm").get(obj.getReferenceTextSpan().getLeft(),
                    obj.getReferenceTextSpan().getRight()).inDocumentOrder();
            FeatureMap rpBabelnetVictorNormFM = Factory.newFeatureMap();
            for(int i=0; i<rpBabelnetVictorNorm.size(); i++)
            {
                if (i == 0) {
                    rpBabelnetVictorNormFM = rpBabelnetVictorNorm.get(i).getFeatures();
                } else {
                    rpBabelnetVictorNormFM = Utilities.combineNormalizedVectors(rpBabelnetVictorNormFM, rpBabelnetVictorNorm.get(i).getFeatures());
                }
            }

            if (cosineType.equals("LEMMA")) {
                if (cpVictorNormFM.size() > 0 && rpVictorNormFM.size() > 0) {
                    value = summa.scorer.Cosine.cosine1(cpVictorNormFM, rpVictorNormFM);
                }
            } else if (cosineType.equals("BABELNET")) {
                if (cpBabelnetVictorNormFM.size() > 0 && rpBabelnetVictorNormFM.size() > 0) {
                    value = summa.scorer.Cosine.cosine1(cpBabelnetVictorNormFM, rpBabelnetVictorNormFM);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return value;
    }
}
