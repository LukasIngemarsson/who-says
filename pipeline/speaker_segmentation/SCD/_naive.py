"""
Naive speaker change point detection using cosine similarity with reference embeddings.

This approach compares audio segments against known speaker embeddings to detect
when the speaker identity changes.
"""
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from scipy.spatial.distance import cosine
from loguru import logger


class NaiveSCD:
    """
    Speaker change detection using cosine similarity with reference speaker embeddings.

    This naive approach:
    1. Loads reference embeddings for known speakers
    2. Slides a window across the audio
    3. Computes embedding for each window
    4. Compares each window to reference speakers using cosine similarity
    5. Detects change points where the best-matching speaker changes
    """

    def __init__(
        self,
        reference_dir: Optional[Union[str, Path]] = None,
        reference_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        embedding_model: str = "pyannote",
        window_duration: float = 1.0,
        step_duration: float = 0.5,
        similarity_threshold: float = 0.5,
        min_duration: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Parameters
        ----------
        reference_dir : str or Path, optional
            Directory containing reference speaker audio files.
            Each .wav file should be named after the speaker (e.g., "speaker_A.wav")
        reference_embeddings : Dict[str, torch.Tensor], optional
            Pre-computed reference embeddings {speaker_name: embedding_tensor}
            If provided, reference_dir is ignored.
        embedding_model : str
            Embedding model to use: "pyannote", "speechbrain", or "wav2vec2"
        window_duration : float
            Duration of sliding window in seconds
        step_duration : float
            Step size for sliding window in seconds
        similarity_threshold : float
            Minimum cosine similarity to consider a match (0-1).
            Below this threshold, segment is considered "unknown speaker"
        min_duration : float
            Minimum duration between change points in seconds
        device : torch.device
            Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.similarity_threshold = similarity_threshold
        self.min_duration = min_duration
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        self._embedding_model = None
        self._init_embedding_model()

        # Load or store reference embeddings
        self.reference_embeddings: Dict[str, torch.Tensor] = {}
        if reference_embeddings is not None:
            self.reference_embeddings = reference_embeddings
        elif reference_dir is not None:
            self._load_reference_embeddings(Path(reference_dir))

    def _init_embedding_model(self):
        """Initialize the embedding model lazily."""
        if self._embedding_model is not None:
            return

        if self.embedding_model_name == "pyannote":
            from pipeline.speaker_recognition.embedding._pyannote import PyAnnoteEmbedding
            self._embedding_model = PyAnnoteEmbedding()
        elif self.embedding_model_name == "speechbrain":
            from pipeline.speaker_recognition.embedding.speechbrain import SpeechBrainEmbedding
            self._embedding_model = SpeechBrainEmbedding()
        elif self.embedding_model_name == "wav2vec2":
            from pipeline.speaker_recognition.embedding.wav2vec2 import Wav2Vec2Embedding
            self._embedding_model = Wav2Vec2Embedding()
        else:
            raise ValueError(f"Unknown embedding model: {self.embedding_model_name}")

    def _load_reference_embeddings(self, reference_dir: Path):
        """
        Load reference speaker embeddings from audio files.

        Parameters
        ----------
        reference_dir : Path
            Directory containing speaker audio files (*.wav or *.mp3)
        """
        from utils import load_audio_from_file

        if not reference_dir.exists():
            raise ValueError(f"Reference directory does not exist: {reference_dir}")

        # Support both .wav and .mp3 files
        audio_files = list(reference_dir.glob("*.wav")) + list(reference_dir.glob("*.mp3"))
        if not audio_files:
            raise ValueError(f"No audio files (.wav or .mp3) found in reference directory: {reference_dir}")

        logger.info(f"Loading {len(audio_files)} reference speaker embeddings...")

        for audio_file in audio_files:
            speaker_name = audio_file.stem  # filename without extension
            audio, sr = load_audio_from_file(str(audio_file))

            # Get embedding
            embedding = self._embedding_model.embed(audio, sr)
            self.reference_embeddings[speaker_name] = embedding.cpu()
            logger.debug(f"  Loaded embedding for speaker: {speaker_name}")

        logger.info(f"Loaded embeddings for {len(self.reference_embeddings)} speakers")

    def add_reference_embedding(self, speaker_name: str, embedding: torch.Tensor):
        """
        Add a reference embedding for a speaker.

        Parameters
        ----------
        speaker_name : str
            Name/ID of the speaker
        embedding : torch.Tensor
            Speaker embedding tensor
        """
        self.reference_embeddings[speaker_name] = embedding.cpu()

    def add_reference_from_audio(self, speaker_name: str, audio: torch.Tensor, sample_rate: int):
        """
        Add a reference embedding from audio.

        Parameters
        ----------
        speaker_name : str
            Name/ID of the speaker
        audio : torch.Tensor
            Audio waveform
        sample_rate : int
            Sample rate of the audio
        """
        embedding = self._embedding_model.embed(audio, sample_rate)
        self.reference_embeddings[speaker_name] = embedding.cpu()

    def _compute_similarity(self, embedding: torch.Tensor, reference: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings.

        Returns value in range [0, 1] where 1 means identical.
        """
        emb_np = embedding.cpu().numpy().flatten()
        ref_np = reference.cpu().numpy().flatten()

        # cosine() returns distance, convert to similarity
        similarity = 1.0 - cosine(emb_np, ref_np)
        return float(similarity)

    def _find_best_speaker(self, embedding: torch.Tensor) -> tuple:
        """
        Find the reference speaker most similar to the given embedding.

        Returns
        -------
        (speaker_name, similarity, is_overlap) : tuple
            Best matching speaker name, similarity score, and whether overlap detected.
            Returns ("unknown", 0.0, False) if no references or below threshold.
        """
        if not self.reference_embeddings:
            return ("unknown", 0.0, False)

        # Calculate similarity to all speakers
        similarities = {}
        for speaker_name, ref_embedding in self.reference_embeddings.items():
            similarities[speaker_name] = self._compute_similarity(embedding, ref_embedding)

        # Find best and second best
        sorted_speakers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_speaker, best_similarity = sorted_speakers[0]

        # Check for overlap: if two speakers both have high similarity, it's likely overlap
        is_overlap = False
        if len(sorted_speakers) > 1:
            second_speaker, second_similarity = sorted_speakers[1]
            # If second best is also above threshold and close to best, likely overlap
            if (second_similarity >= self.similarity_threshold and
                second_similarity >= best_similarity * 0.8):  # Within 80% of best
                is_overlap = True

        # Apply threshold
        if best_similarity < self.similarity_threshold:
            return ("unknown", best_similarity, False)

        return (best_speaker, best_similarity, is_overlap)

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> List[float]:
        """
        Detect speaker change points using cosine similarity with reference embeddings.

        Speaker changes are only detected when transitioning from one clear speaker
        to another clear speaker. Overlapping speech (multiple speakers) is ignored.

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
        # Convert to 1D if needed
        if waveform.ndim == 2:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

        audio_duration = waveform.shape[-1] / sample_rate
        window_samples = int(self.window_duration * sample_rate)
        step_samples = int(self.step_duration * sample_rate)

        # Process audio in sliding windows
        speaker_timeline = []  # [(timestamp, speaker_name, similarity, is_overlap), ...]

        position = 0
        while position + window_samples <= waveform.shape[-1]:
            timestamp = position / sample_rate
            window = waveform[position:position + window_samples]

            # Get embedding for this window
            with torch.no_grad():
                embedding = self._embedding_model.embed(window, sample_rate)

            # Find best matching speaker
            speaker, similarity, is_overlap = self._find_best_speaker(embedding)
            speaker_timeline.append((timestamp, speaker, similarity, is_overlap))

            position += step_samples

        # Handle last partial window if significant
        remaining = waveform.shape[-1] - position
        if remaining > window_samples * 0.5:  # At least 50% of window
            timestamp = position / sample_rate
            window = waveform[position:]

            with torch.no_grad():
                embedding = self._embedding_model.embed(window, sample_rate)

            speaker, similarity, is_overlap = self._find_best_speaker(embedding)
            speaker_timeline.append((timestamp, speaker, similarity, is_overlap))

        # Extract change points from speaker timeline (ignoring overlaps)
        change_points = self._extract_change_points(speaker_timeline)

        return change_points

    def _extract_change_points(self, timeline: List[tuple]) -> List[float]:
        """
        Extract speaker change points from the speaker timeline.

        Only detects changes when transitioning from one clear (non-overlapping)
        speaker segment to another clear speaker segment. Overlapping regions
        are skipped.

        Parameters
        ----------
        timeline : List[tuple]
            List of (timestamp, speaker_name, similarity, is_overlap) tuples

        Returns
        -------
        change_points : List[float]
            Timestamps where speaker changes
        """
        if len(timeline) < 2:
            return []

        change_points = []

        # Find the last non-overlapping speaker before current position
        last_clear_speaker = None
        last_clear_timestamp = None

        for timestamp, speaker, similarity, is_overlap in timeline:
            if is_overlap:
                # Skip overlap regions - no change detection during overlap
                continue

            if last_clear_speaker is not None and speaker != last_clear_speaker:
                # Speaker changed from one clear speaker to another
                change_points.append(timestamp)

            last_clear_speaker = speaker
            last_clear_timestamp = timestamp

        # Filter by min_duration
        if self.min_duration > 0 and len(change_points) > 1:
            filtered = [change_points[0]]
            for cp in change_points[1:]:
                if cp - filtered[-1] >= self.min_duration:
                    filtered.append(cp)
            change_points = filtered

        return change_points

    def get_speaker_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000
    ) -> List[Dict]:
        """
        Get full speaker segmentation (not just change points).

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform
        sample_rate : int
            Sample rate of the audio

        Returns
        -------
        segments : List[Dict]
            List of segments with 'start', 'end', 'speaker', 'similarity', 'is_overlap' keys
        """
        # Convert to 1D if needed
        if waveform.ndim == 2:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

        audio_duration = waveform.shape[-1] / sample_rate
        window_samples = int(self.window_duration * sample_rate)
        step_samples = int(self.step_duration * sample_rate)

        # Process audio in sliding windows
        speaker_timeline = []

        position = 0
        while position + window_samples <= waveform.shape[-1]:
            timestamp = position / sample_rate
            window = waveform[position:position + window_samples]

            with torch.no_grad():
                embedding = self._embedding_model.embed(window, sample_rate)

            speaker, similarity, is_overlap = self._find_best_speaker(embedding)
            speaker_timeline.append((timestamp, speaker, similarity, is_overlap))

            position += step_samples

        # Convert timeline to segments
        if not speaker_timeline:
            return []

        segments = []
        current_speaker = speaker_timeline[0][1]
        current_start = speaker_timeline[0][0]
        current_similarities = [speaker_timeline[0][2]]
        current_is_overlap = speaker_timeline[0][3]

        for timestamp, speaker, similarity, is_overlap in speaker_timeline[1:]:
            # Segment ends when speaker changes OR overlap status changes
            if speaker != current_speaker or is_overlap != current_is_overlap:
                # End previous segment
                end_time = timestamp
                avg_similarity = np.mean(current_similarities)
                segments.append({
                    'start': current_start,
                    'end': end_time,
                    'speaker': current_speaker,
                    'similarity': float(avg_similarity),
                    'is_overlap': current_is_overlap
                })
                # Start new segment
                current_speaker = speaker
                current_start = timestamp
                current_similarities = [similarity]
                current_is_overlap = is_overlap
            else:
                current_similarities.append(similarity)

        # Add final segment
        segments.append({
            'start': current_start,
            'end': audio_duration,
            'speaker': current_speaker,
            'similarity': float(np.mean(current_similarities)),
            'is_overlap': current_is_overlap
        })

        return segments
