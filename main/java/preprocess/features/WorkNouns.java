package preprocess.features;

import java.util.*;

import gate.*;
import preprocess.reader.DocumentCtx;
import preprocess.reader.TrainingExample;

/**
 * Created by Ahmed on 11/13/15.
 */
public class WorkNouns {
    Set<String> determinersSet = new HashSet<String>(Arrays.asList("the", "this", "that", "those", "these", "his",
            "her", "their", "such", "previous", "other"));
    Set<String> workNounsSet = new HashSet<String>(Arrays.asList("account", "algorithm", "analysis", "analyses", "approach",
            "approaches", "application", "applications", "architecture", "architectures", "characterization", "characterisation",
            "component", "components", "corpus", "corpora", "design", "designs", "evaluation", "evaluations", "example", "examples",
            "experiment", "experiments", "extension", "extensions", "evaluation", "formalism", "formalisms", "formalization",
            "formalizations", "formalization", "formalizations", "formulation", "formulations", "framework", "frameworks",
            "implementation", "implementations", "investigation", "investigations", "machinery", "machineries", "method",
            "methods", "methodology", "methodologies", "model", "models", "module", "modules", "paper", "papers", "process",
            "processes", "procedure", "procedures", "program", "programs", "prototype", "prototypes", "research", "researches",
            "result", "results", "strategy", "strategies", "system", "systems", "technique", "techniques", "theory", "theories",
            "tool", "tools", "treatment", "treatments", "work", "works"));

    public Double calculateFeature(TrainingExample obj, DocumentCtx ctx) {
        Double value = (0d);
        Document document = ctx.getReferenceDoc();
        List<Annotation> tokens = document.getAnnotations("Analysis").get("Token").get(obj.getReferenceTextSpan().getLeft(), obj.getReferenceTextSpan().getRight()).inDocumentOrder();
        for(int i=0; i<tokens.size() -1;i++)
        {
            if(determinersSet.contains(tokens.get(i).getFeatures().get("string").toString().toLowerCase()) &&
                    workNounsSet.contains(tokens.get(i+1).getFeatures().get("string").toString().toLowerCase()))
            {
                value = (1d);
            }
        }
        return value;
    }
}
