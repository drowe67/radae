# coding: utf-8

# derived from: https://github.com/openai/whisper/blob/main/notebooks/LibriSpeech.ipynb

import os,argparse
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, n_mels, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device
        self.n_mels = n_mels
        print(n_mels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio,n_mels=self.n_mels)
        
        return (mel, text)


parser = argparse.ArgumentParser()
parser.add_argument('test_name', type=str, help='Librispeech dataset name (e.g. test-clean)')
parser.add_argument('-n', type=str, help='Number of dataset entries to use (default all of them)')
parser.add_argument('--model', default='base.en',type=str, help='Whisper model')
args = parser.parse_args()

model = whisper.load_model(args.model)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)
# predict without timestamps for short-form transcription
options = whisper.DecodingOptions(language="en", without_timestamps=True)

dataset = LibriSpeech(model.dims.n_mels, args.test_name)
if args.n:
    dataset = torch.utils.data.Subset(dataset,list(range(0,int(args.n))))
print("dataset length:", dataset.__len__())
loader = torch.utils.data.DataLoader(dataset, batch_size=16)


hypotheses = []
references = []

for mels, texts in loader:
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)

data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))


# # Calculating the word error rate
# 
# Now, we use our English normalizer implementation to standardize the transcription and calculate the WER.

import jiwer
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]
print(data)

wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")

