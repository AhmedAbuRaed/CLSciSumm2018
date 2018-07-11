package preprocess.reader;

import org.apache.commons.lang3.tuple.Pair;

/**
 * Created by ahmed on 5/8/2016.
 */
public class TrainingExample {
    private Pair<Long, Long> citanceTextSpan;
    private Pair<Long, Long> referenceTextSpan;

    //Facet: AIM (1), HYPOTHESIS (2), METHOD (3), IMPLICATION (4) and RESULT (5)
    //classType 1 (match) or 2 (facet)
    public TrainingExample(Pair<Long, Long> citanceTextSpan, Pair<Long, Long> referenceTextSpan) {
        this.setCitanceTextSpan(citanceTextSpan);
        this.setReferenceTextSpan(referenceTextSpan);
    }

    public Pair<Long, Long> getCitanceTextSpan() {
        return citanceTextSpan;
    }

    public void setCitanceTextSpan(Pair<Long, Long> citanceTextSpan) {
        this.citanceTextSpan = citanceTextSpan;
    }

    public Pair<Long, Long> getReferenceTextSpan() {
        return referenceTextSpan;
    }

    public void setReferenceTextSpan(Pair<Long, Long> referenceTextSpan) {
        this.referenceTextSpan = referenceTextSpan;
    }
}