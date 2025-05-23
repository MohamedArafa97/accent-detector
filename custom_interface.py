from speechbrain.pretrained import EncoderClassifier

class CustomEncoderWav2vec2Classifier(EncoderClassifier):
    def compute_forward(self, batch, stage):
        wavs, wav_lens = batch.sig
        feats = self.mods.compute_features(wavs)
        if self.mods.normalize:
            feats = self.mods.normalize(feats, wav_lens)
        x = self.mods.encoder(feats)
        outputs = self.mods.classifier(x)
        return outputs

    def classify_file(self, path):
        signal = self.load_audio(path)
        batch = self.make_batch(signal)
        probs = self.forward(batch)
        score, index = probs.max(1)
        label = self.hparams.label_encoder.decode(index)
        return probs, score.item(), index.item(), label
