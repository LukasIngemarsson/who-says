"""
Speaker change point detection using NVIDIA NeMo Sortformer model.
"""
import os
import torch
import numpy as np
from typing import Optional, List
from nemo.collections.asr.models import SortformerEncLabelModel

class NemoSCD:
    """
    Speaker change point detection using NVIDIA NeMo Sortformer.

    Uses the diar_sortformer_4spk-v1 model which outputs speaker diarization
    segments. Change points are extracted from segment boundaries.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        min_duration: float = 0.0,
    ):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to local .nemo file OR HuggingFace model ID.
            If None, defaults to "nvidia/diar_sortformer_4spk-v1".
        device : torch.device
            Device to run inference on
        min_duration : float
            Minimum duration between change points in seconds
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_duration = min_duration

        # Suppress verbose NeMo logging
        import logging
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)  # Only show errors, not warnings/info

        # Print NeMo version for debugging
        try:
            import nemo
            print(f"[NemoSCD] NeMo version: {nemo.__version__}")
        except Exception as e:
            print(f"[NemoSCD] Could not determine NeMo version: {e}")

        # Load model - local file or HuggingFace
        if model_path is None:
            model_path = "nvidia/diar_sortformer_4spk-v1"

        if model_path.endswith(".nemo") and os.path.isfile(model_path):
            # Load from local .nemo file
            self.model = SortformerEncLabelModel.restore_from(model_path, map_location=self.device)
        else:
            # Load from HuggingFace (downloads and caches locally)
            self.model = SortformerEncLabelModel.from_pretrained(model_path, map_location=self.device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _parse_segment(self, segment_str: str) -> tuple:
        """Parse a segment string 'start end speaker' into (start, end, speaker)."""
        parts = segment_str.strip().split()
        start = float(parts[0])
        end = float(parts[1])
        speaker = parts[2]
        return (start, end, speaker)

    def _inspect_diarize_signature(self):
        """Inspect the diarize method signature to understand the API."""
        import inspect
        if hasattr(self.model, 'diarize'):
            sig = inspect.signature(self.model.diarize)
            params = list(sig.parameters.keys())
            print(f"[NemoSCD] diarize() signature: {sig}")
            print(f"[NemoSCD] diarize() parameters: {params}")
            return params
        return []

    def _call_diarize(self, audio_path: str):
        """
        Call the diarize method with API compatibility handling.
        NeMo API varies across versions.
        """
        import json
        import inspect
        from omegaconf import OmegaConf
        import soundfile as sf

        errors = []

        # First, inspect the actual signature
        params = self._inspect_diarize_signature()

        # Build patterns based on what parameters are actually available
        patterns = []

        if 'audio' in params:
            # Current NeMo 2.x API (newer versions)
            if 'batch_size' in params:
                patterns.append(("audio=path, batch_size=1", lambda: self.model.diarize(audio=audio_path, batch_size=1)))
                patterns.append(("audio=[path], batch_size=1", lambda: self.model.diarize(audio=[audio_path], batch_size=1)))
            else:
                patterns.append(("audio=path", lambda: self.model.diarize(audio=audio_path)))
                patterns.append(("audio=[path]", lambda: self.model.diarize(audio=[audio_path])))

        if 'paths2audio_files' in params:
            patterns.append(("paths2audio_files", lambda: self.model.diarize(paths2audio_files=[audio_path])))

        # Try positional arguments
        if params:  # Only if diarize accepts parameters
            patterns.append(("positional [path]", lambda: self.model.diarize([audio_path])))
            patterns.append(("positional path", lambda: self.model.diarize(audio_path)))

        # Try direct forward inference first (most reliable for NeMo 2.1.0)
        try:
            print("[NemoSCD] Trying direct forward approach")
            result = self._direct_forward_inference(audio_path)
            print("[NemoSCD] Success with direct forward approach")
            return result
        except Exception as e:
            errors.append(f"direct_forward: {type(e).__name__} - {e}")

        # Try diarize() API patterns (for newer NeMo versions)
        for name, attempt in patterns:
            try:
                print(f"[NemoSCD] Trying pattern: {name}")
                result = attempt()
                print(f"[NemoSCD] Success with pattern: {name}")
                return result
            except TypeError as e:
                errors.append(f"{name}: TypeError - {e}")
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__} - {e}")

        # NeMo 2.1.0 style: configure test_ds then call diarize() with no args
        try:
            print("[NemoSCD] Trying NeMo 2.1.0 setup_test_data approach")
            result = self._diarize_via_test_data(audio_path)
            print("[NemoSCD] Success with setup_test_data approach")
            return result
        except Exception as e:
            errors.append(f"setup_test_data: {type(e).__name__} - {e}")

        # If none work, raise a clear error with version info
        try:
            import nemo
            nemo_version = nemo.__version__
        except:
            nemo_version = "unknown"

        raise RuntimeError(
            f"Could not find compatible diarize() API.\n"
            f"NeMo version: {nemo_version}\n"
            f"Available diarize() params: {params}\n"
            f"Tried patterns:\n" + "\n".join(f"  - {err}" for err in errors)
        )

    def _diarize_via_test_data(self, audio_path: str):
        """
        NeMo 2.1.0 style diarization: configure test data, then call diarize().
        Called for short audio (<30s). Long audio is chunked before reaching here.
        """
        import json
        import soundfile as sf
        from omegaconf import OmegaConf

        # Get audio duration
        audio_data, sr = sf.read(audio_path)
        duration = len(audio_data) / sr

        # Create manifest file
        manifest_path = audio_path.replace('.wav', '_manifest.json')
        if manifest_path == audio_path:
            manifest_path = audio_path + '_manifest.json'

        with open(manifest_path, 'w') as f:
            f.write(json.dumps({
                "audio_filepath": audio_path,
                "duration": duration,
                "label": "infer",
                "text": "-",
                "offset": 0,
                "rttm_filepath": None,
                "uem_filepath": None,
            }) + "\n")

        try:
            # Configure test data with all required keys for NeMo 2.1.0
            test_config = OmegaConf.create({
                "manifest_filepath": manifest_path,
                "sample_rate": 16000,
                "num_spks": 4,
                "session_len_sec": max(90, int(duration) + 10),
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
                "seq_eval_mode": True,
                "validation_mode": True,
                # Additional required keys for NeMo 2.1.0
                "soft_label_thres": 0.5,
                "soft_targets": False,
                "labels": None,
                "is_tarred": False,
                "tarred_audio_filepaths": None,
                "pin_memory": False,
                "drop_last": False,
                "window_stride": 0.01,
                "subsampling_factor": 8,
                "use_lhotse": False,
                "use_bucketing": False,
            })

            self.model.setup_test_data(test_config)

            # Call diarize with no arguments (NeMo 2.1.0 style)
            result = self.model.diarize()
            return result if result else [[]]

        finally:
            if os.path.exists(manifest_path):
                os.unlink(manifest_path)

    def _diarize_chunked(self, audio_data: np.ndarray, sr: int, chunk_sec: float, overlap_sec: float):
        """
        Process long audio in chunks and merge results.
        """
        import tempfile
        import soundfile as sf

        chunk_samples = int(chunk_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        step_samples = chunk_samples - overlap_samples

        all_segments = []
        chunk_idx = 0
        pos = 0
        total_chunks = (len(audio_data) - overlap_samples) // step_samples + 1

        while pos < len(audio_data):
            end_pos = min(pos + chunk_samples, len(audio_data))
            chunk = audio_data[pos:end_pos]
            chunk_offset = pos / sr  # Time offset for this chunk

            print(f"[NemoSCD] Processing chunk {chunk_idx + 1}/{total_chunks}: {chunk_offset:.1f}s - {end_pos/sr:.1f}s")

            # Clear GPU cache before each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Write chunk to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                chunk_path = f.name
                sf.write(chunk_path, chunk, sr)

            try:
                # Use _call_diarize which tries all available methods
                with torch.no_grad():
                    result = self._call_diarize(chunk_path)

                # Adjust segment times by chunk offset
                if result and len(result) > 0:
                    for seg_str in result[0]:
                        parts = seg_str.strip().split()
                        start = float(parts[0]) + chunk_offset
                        end = float(parts[1]) + chunk_offset
                        speaker = parts[2]
                        all_segments.append(f"{start:.3f} {end:.3f} {speaker}")

            except Exception as e:
                print(f"[NemoSCD] Warning: Chunk {chunk_idx + 1} failed: {e}")
                # Continue with next chunk instead of failing completely

            finally:
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)

            pos += step_samples
            chunk_idx += 1

        # Merge overlapping segments from different chunks
        merged_segments = self._merge_chunk_segments(all_segments, overlap_sec)
        return [merged_segments]

    def _merge_chunk_segments(self, segments: List[str], overlap_sec: float) -> List[str]:
        """
        Merge segments from overlapping chunks.
        Remove duplicate detections in overlap regions.
        """
        if not segments:
            return []

        # Parse all segments
        parsed = []
        for seg_str in segments:
            parts = seg_str.strip().split()
            start = float(parts[0])
            end = float(parts[1])
            speaker = parts[2]
            parsed.append((start, end, speaker))

        # Sort by start time
        parsed = sorted(parsed, key=lambda x: x[0])

        # Merge overlapping segments with same speaker
        merged = []
        for start, end, speaker in parsed:
            if merged and merged[-1][2] == speaker:
                # Same speaker - check if overlapping or very close
                prev_start, prev_end, prev_speaker = merged[-1]
                if start <= prev_end + 0.1:  # Allow 100ms gap
                    # Merge by extending previous segment
                    merged[-1] = (prev_start, max(prev_end, end), speaker)
                    continue
            merged.append((start, end, speaker))

        # Convert back to strings
        return [f"{s:.3f} {e:.3f} {spk}" for s, e, spk in merged]

    def _direct_forward_inference(self, audio_path: str):
        """
        Direct forward inference bypassing diarize() for older NeMo versions.
        Processes audio through the model's encoder and decoder directly.
        """
        import soundfile as sf
        import inspect

        # Load audio
        audio_data, sr = sf.read(audio_path)
        if sr != 16000:
            # Resample to 16kHz if needed
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Convert to tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        audio_length = torch.tensor([len(audio_data)], dtype=torch.long).to(self.device)

        # Process through model - try different forward methods
        with torch.no_grad():
            processed_signal = None
            processed_signal_length = None

            # Try process_signal with different parameter names
            if hasattr(self.model, 'process_signal'):
                sig = inspect.signature(self.model.process_signal)
                ps_params = list(sig.parameters.keys())
                print(f"[NemoSCD] process_signal params: {ps_params}")

                try:
                    if 'input_signal' in ps_params:
                        processed_signal, processed_signal_length = self.model.process_signal(
                            input_signal=audio_tensor, input_signal_length=audio_length
                        )
                    elif 'audio_signal' in ps_params:
                        processed_signal, processed_signal_length = self.model.process_signal(
                            audio_signal=audio_tensor, audio_signal_length=audio_length
                        )
                    else:
                        # Try positional
                        processed_signal, processed_signal_length = self.model.process_signal(
                            audio_tensor, audio_length
                        )
                except Exception as e:
                    print(f"[NemoSCD] process_signal failed: {e}")

            # Try preprocessor + encoder if process_signal didn't work
            if processed_signal is None and hasattr(self.model, 'preprocessor'):
                try:
                    processed_signal, processed_signal_length = self.model.preprocessor(
                        input_signal=audio_tensor, length=audio_length
                    )
                    if hasattr(self.model, 'encoder'):
                        processed_signal, processed_signal_length = self.model.encoder(
                            audio_signal=processed_signal, length=processed_signal_length
                        )
                except Exception as e:
                    print(f"[NemoSCD] preprocessor/encoder failed: {e}")

            if processed_signal is None:
                raise RuntimeError("Could not process audio through model")

            # Try forward first (works reliably on NeMo 2.1.0)
            preds = None
            if hasattr(self.model, 'forward'):
                try:
                    preds = self.model.forward(
                        audio_signal=audio_tensor, audio_signal_length=audio_length
                    )
                except Exception as e:
                    print(f"[NemoSCD] forward failed: {e}")

            # Try forward_infer as fallback
            if preds is None and hasattr(self.model, 'forward_infer'):
                sig = inspect.signature(self.model.forward_infer)
                fi_params = list(sig.parameters.keys())

                try:
                    if fi_params == ['emb_seq']:
                        preds = self.model.forward_infer(emb_seq=processed_signal)
                    elif 'emb_seq' in fi_params and 'emb_seq_length' in fi_params:
                        preds = self.model.forward_infer(
                            emb_seq=processed_signal, emb_seq_length=processed_signal_length
                        )
                    else:
                        preds = self.model.forward_infer(processed_signal)
                except Exception as e:
                    print(f"[NemoSCD] forward_infer failed: {e}")

            if preds is None:
                raise RuntimeError("Could not run inference through model")

        # Convert predictions to segments
        # preds shape: (batch, frames, num_speakers)
        if isinstance(preds, tuple):
            preds = preds[0]  # Some models return (preds, other_stuff)
        segments = self._preds_to_segments(preds[0], len(audio_data) / 16000)
        return [segments]

    def _preds_to_segments(self, preds: torch.Tensor, duration: float, threshold: float = 0.5):
        """
        Convert model predictions (speaker probabilities per frame) to segment strings.

        Parameters
        ----------
        preds : torch.Tensor
            Speaker probabilities, shape (num_frames, num_speakers)
        duration : float
            Total audio duration in seconds
        threshold : float
            Threshold for speaker activity

        Returns
        -------
        segments : List[str]
            List of segment strings "start end speaker"
        """
        # Sortformer uses 0.08s frame shift (80ms)
        frame_shift = 0.08
        num_frames = preds.shape[0]
        num_speakers = preds.shape[1]

        # Binarize predictions
        binary_preds = (preds > threshold).cpu().numpy()

        segments = []
        for spk_idx in range(num_speakers):
            spk_preds = binary_preds[:, spk_idx]

            # Find contiguous regions
            in_segment = False
            start_frame = 0

            for frame_idx in range(len(spk_preds)):
                if spk_preds[frame_idx] and not in_segment:
                    # Start of segment
                    in_segment = True
                    start_frame = frame_idx
                elif not spk_preds[frame_idx] and in_segment:
                    # End of segment
                    in_segment = False
                    start_time = start_frame * frame_shift
                    end_time = frame_idx * frame_shift
                    segments.append(f"{start_time:.3f} {end_time:.3f} speaker_{spk_idx}")

            # Handle segment that extends to end
            if in_segment:
                start_time = start_frame * frame_shift
                end_time = min(num_frames * frame_shift, duration)
                segments.append(f"{start_time:.3f} {end_time:.3f} speaker_{spk_idx}")

        return segments

    def _extract_change_points_from_segments(
        self,
        segments: List,
        audio_duration: float
    ) -> List[float]:
        """
        Extract speaker change points from diarization segments.

        Parameters
        ----------
        segments : List
            List of segment strings "start end speaker" from Sortformer
        audio_duration : float
            Total duration of audio in seconds

        Returns
        -------
        change_points : List[float]
            Timestamps where speaker changes occur
        """
        if not segments:
            return []

        # Parse segments - they come as strings "start end speaker"
        parsed = [self._parse_segment(s) for s in segments]

        # Sort segments by start time
        parsed = sorted(parsed, key=lambda x: x[0])

        change_points = []
        prev_speaker = None

        for start, end, speaker in parsed:
            if prev_speaker is not None and speaker != prev_speaker:
                change_points.append(start)
            prev_speaker = speaker

        # Filter by min_duration
        if self.min_duration > 0 and len(change_points) > 1:
            filtered = [change_points[0]]
            for cp in change_points[1:]:
                if cp - filtered[-1] >= self.min_duration:
                    filtered.append(cp)
            change_points = filtered

        return change_points

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> List[float]:
        """
        Detect speaker change points.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        change_points : List[float]
            List of timestamps (in seconds) where speaker changes occur
        """
        import tempfile
        import soundfile as sf

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Ensure 1D
        if waveform_np.ndim == 2:
            waveform_np = waveform_np.squeeze(0)
        if waveform_np.ndim == 2 and waveform_np.shape[0] > 1:
            waveform_np = waveform_np.mean(axis=0)

        audio_duration = len(waveform_np) / sample_rate

        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # For long audio (>30s), use chunking directly to avoid OOM
        max_chunk_sec = 30.0
        if audio_duration > max_chunk_sec:
            print(f"[NemoSCD] Audio is {audio_duration:.1f}s, using chunked processing")
            predicted_segments = self._diarize_chunked(waveform_np, sample_rate, max_chunk_sec, overlap_sec=5.0)
        else:
            # Short audio - use standard approach
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, waveform_np, sample_rate)

            try:
                # Run diarization - NeMo API varies by version
                with torch.no_grad():
                    predicted_segments = self._call_diarize(temp_path)
            finally:
                # Clean up temp file
                os.unlink(temp_path)

        # predicted_segments is a list (one per audio file) of lists of tuples
        # Each tuple is (start, end, speaker_id)
        if predicted_segments and len(predicted_segments) > 0:
            segments = predicted_segments[0]  # First (only) audio file
        else:
            segments = []

        # Extract change points from segments
        change_points = self._extract_change_points_from_segments(
            segments, audio_duration
        )

        return change_points

    def get_diarization_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> List[dict]:
        """
        Get full diarization segments (not just change points).

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform, shape (num_samples,) or (1, num_samples)
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        segments : List[dict]
            List of segments with 'start', 'end', 'speaker' keys
        """
        import tempfile
        import soundfile as sf

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Ensure 1D
        if waveform_np.ndim == 2:
            waveform_np = waveform_np.squeeze(0)
        if waveform_np.ndim == 2 and waveform_np.shape[0] > 1:
            waveform_np = waveform_np.mean(axis=0)

        # Sortformer requires audio file path, so write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, waveform_np, sample_rate)

        try:
            # Run diarization - NeMo API varies by version
            with torch.no_grad():
                predicted_segments = self._call_diarize(temp_path)

            # predicted_segments is a list (one per audio file) of lists of segment strings
            segments = []
            if predicted_segments and len(predicted_segments) > 0:
                raw_segments = predicted_segments[0]  # First (only) audio file
                for seg_str in raw_segments:
                    start, end, speaker = self._parse_segment(seg_str)
                    segments.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker
                    })

            return segments

        finally:
            # Clean up temp file
            os.unlink(temp_path)
