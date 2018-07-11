package preprocess.reader;

import gate.Document;

/**
 * Created by ahmed on 5/7/2016.
 */
public class DocumentCtx {
    private Document cp, rp;

    public DocumentCtx(Document cpDocument, Document rpDocument) {
        this.cp = cpDocument;
        this.rp = rpDocument;
    }

    public Document getReferenceDoc() {
        return rp;
    }

    public Document getCitationDoc() {
        return cp;
    }
}