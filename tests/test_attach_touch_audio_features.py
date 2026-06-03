import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import attach_touch_audio_features as audio


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AttachTouchAudioFeaturesTests(unittest.TestCase):
    def test_audio_timbre_attachment_preserves_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            wav_dir = root / "wav"
            sr = 16000
            t = np.linspace(0, 2.0, sr * 2, endpoint=False)
            signal = 0.02 * np.sin(2 * np.pi * 220 * t)
            signal[int(1.0 * sr) : int(1.01 * sr)] += 0.2
            wav_dir.mkdir(parents=True)
            sf.write(wav_dir / "train.wav", signal, sr)
            sf.write(wav_dir / "test.wav", signal, sr)
            write_jsonl(
                dataset / "touch_training_candidates.jsonl",
                [
                    {
                        "video_id": "train",
                        "video_name": "train.MOV",
                        "split": "train",
                        "candidate_time_sec": 1.0,
                        "label_is_touch": True,
                    }
                ],
            )
            write_jsonl(
                dataset / "touch_training_test_frozen.jsonl",
                [
                    {
                        "video_id": "test",
                        "video_name": "test.MOV",
                        "split": "test_frozen",
                        "candidate_time_sec": 1.0,
                        "label_is_touch": False,
                    }
                ],
            )

            manifest = audio.attach_dataset(
                argparse.Namespace(
                    dataset_dir=dataset,
                    out_dir=root / "out",
                    audio_wav_dir=wav_dir,
                    window_sec=0.16,
                    n_fft=512,
                    hop_length=64,
                )
            )

            train_rows = read_jsonl(root / "out" / "touch_training_candidates.jsonl")
            test_rows = read_jsonl(root / "out" / "touch_training_test_frozen.jsonl")
            self.assertEqual(manifest["status"], "features_attached")
            self.assertEqual(manifest["train_val"]["ok_rows"], 1)
            self.assertEqual(manifest["test_frozen"]["ok_rows"], 1)
            self.assertFalse(any(row["split"] == "test_frozen" for row in train_rows))
            self.assertEqual(test_rows[0]["split"], "test_frozen")
            self.assertEqual(train_rows[0]["audio_timbre_status"], "ok")
            self.assertGreater(train_rows[0]["audio_window_rms"], 0)
            self.assertIsNotNone(train_rows[0]["audio_spectral_centroid"])


if __name__ == "__main__":
    unittest.main()
