# =============================================================================
# FILE: src/models/predictor.py       (INFERENCE: use the trained models)
# PURPOSE: One class, ScamPredictor, that hides the differences between the
#          three deployed models (SVM, BiLSTM, DistilBERT). Callers just do
#              predictor.predict("some text", backend="svm")
#          and get back {label, probability, risk, is_scam}.
#          It also produces EXPLANATIONS (which words drove the decision).
# USED BY: app.py (the web API) and demo_predict.py (command-line demo).
# =============================================================================
import os
import sys
import joblib
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import BASE_DIR
from src.models.bilstm import SimpleTokenizer, ScamBiLSTM

MODELS_DIR = os.path.join(BASE_DIR, 'models')


class ScamPredictor:
    """Unified scam/text classifier with fast (SVM) and high-accuracy (DistilBERT) backends."""

    def __init__(self, backend='svm', threshold=0.5):
        self.backend = backend        # default model if predict() is called without one
        # Decision threshold: probability >= threshold -> classified as scam.
        # Raising it = fewer false alarms but more missed scams; lowering it = the opposite.
        self.threshold = threshold
        # Pick GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        # Cache of already-loaded models: {'svm': (...), 'bilstm': (...), ...}
        self._models = {}

    # ---- LAZY LOADING: a model is read from disk only the first time it is needed ----
    # (Loading DistilBERT takes seconds and ~250 MB RAM, so we avoid it unless requested.)
    def _load_svm(self):
        if 'svm' not in self._models:
            # The SVM needs TWO files: the TF-IDF vectorizer (text -> numbers) and the classifier
            vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
            clf = joblib.load(os.path.join(MODELS_DIR, 'svm.joblib'))
            self._models['svm'] = (vectorizer, clf)
        return self._models['svm']

    def _load_bilstm(self):
        if 'bilstm' not in self._models:
            # Needs the saved tokenizer (word->id) AND the saved network weights
            tokenizer = SimpleTokenizer.load(os.path.join(MODELS_DIR, 'lstm_tokenizer.json'))
            # Re-create the same architecture, then load the learned weights into it
            model = ScamBiLSTM(vocab_size=len(tokenizer.word2idx)).to(self.device)
            model.load_state_dict(torch.load(
                os.path.join(MODELS_DIR, 'bilstm_model.pt'), map_location=self.device))
            model.eval()     # inference mode: dropout OFF
            self._models['bilstm'] = (tokenizer, model)
        return self._models['bilstm']

    def _load_distilbert(self):
        if 'distilbert' not in self._models:
            from transformers import AutoTokenizer, DistilBertForSequenceClassification
            checkpoint = os.path.join(MODELS_DIR, 'distilbert_scamshield')
            # AutoTokenizer reads tokenizer.json directly (the "fast" tokenizer
            # format this checkpoint was saved with). The slow DistilBertTokenizer
            # class instead requires a vocab.txt file that isn't present here,
            # and depending on the installed transformers/tokenizers version,
            # failing to find it can surface as an unrelated-looking error
            # (e.g. "stat: path should be string... not NoneType") instead of
            # a clear "vocab.txt not found" message.
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
            model = DistilBertForSequenceClassification.from_pretrained(checkpoint).to(self.device)
            model.eval()
            self._models['distilbert'] = (tokenizer, model)
        return self._models['distilbert']

    @staticmethod
    def _risk_label(prob):
        """Turn a probability into a human-friendly level: >=0.8 high, >=0.5 medium, else low."""
        if prob >= 0.8:
            return "high"
        if prob >= 0.5:
            return "medium"
        return "low"

    # ---- RAW PROBABILITY FUNCTIONS: take a LIST of texts, return P(scam) for each ----
    def predict_svm(self, texts):
        vectorizer, clf = self._load_svm()
        X = vectorizer.transform(texts)          # text -> TF-IDF vectors (same vocabulary as training)
        probs = clf.predict_proba(X)[:, 1]       # column 1 = probability of class 1 (scam)
        return probs

    def predict_bilstm(self, texts):
        tokenizer, model = self._load_bilstm()
        sequences = [tokenizer.text_to_sequence(t) for t in texts]     # text -> padded id lists
        X = torch.tensor(sequences, dtype=torch.long, device=self.device)
        with torch.no_grad():                    # no gradients needed for plain prediction
            logits = model(X)
        return torch.sigmoid(logits).cpu().numpy()   # logit -> probability

    def predict_distilbert(self, texts):
        tokenizer, model = self._load_distilbert()
        import torch.nn.functional as F
        outputs = []
        batch_size = 64      # process 64 messages at a time to limit memory use
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # padding=True pads only to the longest message in the batch (cheaper than max_length)
            enc = tokenizer(batch, truncation=True, max_length=128,
                            padding=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                logits = model(**enc).logits
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()    # P(scam)
            outputs.extend(probs)
        return outputs

    # ── Explainability ───────────────────────────────────────────────────
    # SVM: exact — coefficient x TF-IDF weight per token (see app.py /explain).
    # BiLSTM / DistilBERT: gradient x input saliency — a standard, real
    # attribution method (Simonyan et al. 2013 "vanilla gradient" saliency;
    # closely related to the first-order term of Integrated Gradients).
    # We backprop the predicted-class score to the input embeddings and take
    # grad . embedding per token position as its signed contribution: not
    # exact like the linear SVM (these are non-linear models), but a real,
    # commonly-used explanation rather than a cosmetic approximation.
    #
    # PLAIN-ENGLISH IDEA of gradient x input: the gradient tells us "if this word's
    # embedding changed slightly, how much would the scam probability change?".
    # Multiplying by the embedding itself scales that by how strongly the word is
    # actually present. Positive total = word pushes TOWARD scam; negative = toward safe.

    def explain_bilstm(self, text, top_k=6):
        """Return ((top_scam_words, top_safe_words), probability) for one message."""
        tokenizer, model = self._load_bilstm()
        # words aligned with the first max_length positions of the id sequence
        words = tokenizer.clean_text(text)[: tokenizer.max_length]
        seq = tokenizer.text_to_sequence(text)
        x = torch.tensor([seq], dtype=torch.long, device=self.device)

        model.zero_grad(set_to_none=True)
        # We run the embedding layer MANUALLY so we can ask PyTorch to keep the gradient
        # at this point (retain_grad). Normally gradients of intermediate values are discarded.
        embedded = model.embedding(x)
        embedded.retain_grad()

        # Re-implement model.forward() step by step, starting from the embeddings
        # (same operations as ScamBiLSTM.forward in bilstm.py)
        lstm_out, (hn, cn) = model.lstm(embedded)
        forward_hidden = hn[-2, :, :]
        backward_hidden = hn[-1, :, :]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        out = model.fc1(hidden)
        out = model.relu(out)
        out = model.dropout(out)
        logit = model.fc2(out).squeeze(-1)
        prob = torch.sigmoid(logit)
        # Backpropagate the scam probability down to the embeddings -> fills embedded.grad
        prob.backward()

        grad = embedded.grad[0]                       # (seq_len, embed_dim)
        # Per position: dot product of gradient and embedding = one signed number per word
        saliency = (grad * embedded[0].detach()).sum(dim=1).cpu().numpy()  # signed, per position

        return self._tokens_from_saliency(words, saliency, top_k), float(prob.item())

    def explain_distilbert(self, text, top_k=6):
        """Same gradient x input method, adapted to DistilBERT's sub-word tokens."""
        import torch.nn.functional as F
        tokenizer, model = self._load_distilbert()
        enc = tokenizer([text], truncation=True, max_length=128, padding=True, return_tensors='pt').to(self.device)
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']

        # Look up the word embeddings ourselves, detach them from any graph, and mark them as
        # requiring gradients; then feed them to the model via inputs_embeds (instead of ids).
        embedding_layer = model.distilbert.embeddings.word_embeddings
        inputs_embeds = embedding_layer(input_ids).clone().detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        prob = F.softmax(outputs.logits, dim=1)[0, 1]     # P(scam)
        prob.backward()

        grad = inputs_embeds.grad[0]                   # (seq_len, hidden_dim)
        saliency = (grad * inputs_embeds[0].detach()).sum(dim=1).cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # merge WordPiece subtokens ("##ing" -> attach to previous word) and
        # drop special tokens ([CLS]/[SEP]/[PAD]) before ranking
        # Example: tokens ['pri', '##ze'] with scores [0.2, 0.1] -> word 'prize' with score 0.3
        merged_words, merged_sal = [], []
        for tok, sal in zip(tokens, saliency):
            if tok in ('[CLS]', '[SEP]', '[PAD]'):
                continue
            if tok.startswith('##') and merged_words:
                merged_words[-1] += tok[2:]
                merged_sal[-1] += float(sal)
            else:
                merged_words.append(tok)
                merged_sal.append(float(sal))

        return self._tokens_from_saliency(merged_words, merged_sal, top_k), float(prob.item())

    @staticmethod
    def _tokens_from_saliency(words, saliency, top_k):
        """Turn per-position scores into two ranked lists: words pushing to scam / to safe."""
        pairs = list(zip(words, saliency))
        # de-duplicate repeated words by summing their contributions
        agg = {}
        for w, s in pairs:
            agg[w] = agg.get(w, 0.0) + float(s)
        # sort highest score first. Top of the list = strongest "scam" evidence,
        # bottom of the list = strongest "safe" evidence.
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        top_scam = [{"token": w, "weight": round(s, 5)} for w, s in ranked if s > 0][:top_k]
        # take the last top_k negatives, then reverse so the MOST negative comes first
        top_safe = [{"token": w, "weight": round(s, 5)} for w, s in ranked if s < 0][-top_k:][::-1]
        return top_scam, top_safe

    def predict(self, text, backend=None):
        """PUBLIC entry point. `text` may be one string or a list of strings.

        Returns one dict for a single string, or a list of dicts for a list:
            {"label": 0/1, "probability": P(scam), "risk": low/medium/high, "is_scam": bool}
        """
        backend = backend or self.backend
        # Normalise the input so the rest of the code always works with a LIST
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)
            single = False

        # Route to the requested model (anything unrecognised falls back to the SVM)
        if backend == 'distilbert':
            probs = self.predict_distilbert(texts)
        elif backend == 'bilstm':
            probs = self.predict_bilstm(texts)
        else:
            probs = self.predict_svm(texts)

        results = []
        for prob in probs:
            # Apply the decision threshold to get the final 0/1 label
            label = 1 if prob >= self.threshold else 0
            results.append({
                "label": label,
                "probability": round(float(prob), 6),
                "risk": self._risk_label(prob),
                "is_scam": label == 1,
            })
        return results[0] if single else results
