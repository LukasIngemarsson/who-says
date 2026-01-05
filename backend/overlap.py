"""
Speaker Overlap Detection (SOD) and Separation (SOS) for the WhoSays backend.
"""
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

import backend.config as cfg
from backend.speaker import get_speakers_in_range


def get_sod_detector() -> Optional[Any]:
    """
    Lazy-initialize the Speaker Overlap Detector.
    Uses PyannoteSOD (lighter weight for streaming).
    """
    if cfg.SOD_DETECTOR is not None:
        return cfg.SOD_DETECTOR

    if not cfg.PIPELINE_AVAILABLE or cfg.pipeline is None:
        logger.warning("SOD detector skipped: pipeline not available")
        return None

    with cfg.SOD_DETECTOR_LOCK:
        # Double-check after acquiring lock
        if cfg.SOD_DETECTOR is not None:
            return cfg.SOD_DETECTOR

        try:
            device = cfg.pipeline.config.device
            logger.info(f"Initializing PyannoteSOD detector on device: {device}")

            from pipeline.speaker_segmentation.SO.Detection import PyannoteSOD
            cfg.SOD_DETECTOR = PyannoteSOD(
                onset=0.5,
                offset=0.5,
                min_duration=cfg.OVERLAP_MIN_DURATION_SEC,
                device=device,
            )

            logger.info("Initialized PyannoteSOD detector for overlap detection successfully")
        except ImportError as e:
            logger.error(f"Failed to import PyannoteSOD - module not found: {e}")
            cfg.SOD_DETECTOR = None
        except Exception as e:
            logger.error(f"Failed to initialize SOD detector: {e}")
            import traceback
            logger.error(traceback.format_exc())
            cfg.SOD_DETECTOR = None

    return cfg.SOD_DETECTOR


def get_sos_separator() -> Optional[Any]:
    """
    Lazy-initialize the Speaker Overlap Separator.
    Uses SpeechBrain SepFormer for source separation.
    """
    if cfg.SOS_SEPARATOR is not None:
        return cfg.SOS_SEPARATOR

    if not cfg.PIPELINE_AVAILABLE or cfg.pipeline is None:
        logger.warning("SOS separator skipped: pipeline not available")
        return None

    with cfg.SOS_SEPARATOR_LOCK:
        # Double-check after acquiring lock
        if cfg.SOS_SEPARATOR is not None:
            return cfg.SOS_SEPARATOR

        try:
            device = cfg.pipeline.config.device
            logger.info(f"Initializing SpeechBrain SOS separator on device: {device}")

            from pipeline.speaker_segmentation.SO.Separation import SpeechBrainSOS
            cfg.SOS_SEPARATOR = SpeechBrainSOS(
                model_name="speechbrain/sepformer-wsj02mix",
                device=device,
            )

            logger.info("Initialized SpeechBrain SOS separator successfully")
        except ImportError as e:
            logger.error(f"Failed to import SpeechBrainSOS - module not found: {e}")
            cfg.SOS_SEPARATOR = None
        except Exception as e:
            logger.error(f"Failed to initialize SOS separator: {e}")
            import traceback
            logger.error(traceback.format_exc())
            cfg.SOS_SEPARATOR = None

    return cfg.SOS_SEPARATOR


def get_recent_overlaps(
    session_id: str,
    since_time: float,
) -> List[Dict[str, Any]]:
    """
    Get overlaps detected since the given time.
    """
    timeline = cfg.SESSION_OVERLAP_TIMELINE.get(session_id, [])
    return [o for o in timeline if o["start"] >= since_time]


def process_overlap_detection(
    session_id: str,
    sr: int,
    current_time: float,
) -> Optional[List[Dict[str, Any]]]:
    """
    Check if enough audio has accumulated for overlap detection.
    If so, run detection and return any new overlap segments.

    Returns:
        List of overlap segments detected, or None if detection not run.
    """
    if not cfg.OVERLAP_DETECTION_ENABLED:
        logger.debug("Overlap detection disabled")
        return None

    state = cfg.SESSION_OVERLAP_STATE.get(session_id)
    if state is None:
        logger.debug(f"No overlap state for session {session_id}")
        return None

    buffer = state["buffer"]
    buffer_start_time = state["buffer_start_time"]
    last_detection = state.get("last_detection_time", 0.0)

    # Check if we have enough audio and time has passed
    buffer_duration = buffer.shape[0] / float(sr)
    time_since_last = current_time - last_detection

    if buffer_duration < cfg.OVERLAP_BUFFER_SEC:
        logger.debug(f"Overlap buffer too short: {buffer_duration:.2f}s < {cfg.OVERLAP_BUFFER_SEC}s")
        return None

    if time_since_last < cfg.OVERLAP_DETECTION_INTERVAL_SEC:
        logger.debug(f"Too soon since last detection: {time_since_last:.2f}s < {cfg.OVERLAP_DETECTION_INTERVAL_SEC}s")
        return None

    logger.info(f"Running overlap detection: buffer={buffer_duration:.2f}s, time_since_last={time_since_last:.2f}s")

    # Get SOD detector
    detector = get_sod_detector()
    if detector is None:
        logger.warning("SOD detector not available - skipping overlap detection")
        return None

    # Run detection
    try:
        with torch.inference_mode():
            overlap_segments = detector(buffer, sr)
        logger.info(f"SOD detection returned {len(overlap_segments)} overlap segments")
    except Exception as e:
        logger.error(f"SOD detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    # Update state
    state["last_detection_time"] = current_time

    # Convert relative timestamps to absolute
    detected_overlaps = []
    for (start_rel, end_rel) in overlap_segments:
        start_abs = buffer_start_time + start_rel
        end_abs = buffer_start_time + end_rel

        # Identify speakers involved using speaker timeline (with lookback)
        speakers = get_speakers_in_range(session_id, start_abs, end_abs)

        # SOD detected overlap means 2+ voices - if we only found 1 speaker,
        # use all identified speakers from this session, or fall back to enrolled speakers
        if len(speakers) < 2:
            if cfg.CURRENT_SPEAKER and cfg.CURRENT_SPEAKER not in speakers:
                speakers.append(cfg.CURRENT_SPEAKER)
            # Add other known speakers from this session
            session_speakers = cfg.SESSION_IDENTIFIED_SPEAKERS.get(session_id, set())
            for spk in session_speakers:
                if spk not in speakers:
                    speakers.append(spk)
            # If still < 2 speakers, use enrolled speakers (SOD confirmed 2+ voices)
            if len(speakers) < 2:
                for spk in cfg.KNOWN_SPEAKERS.keys():
                    if spk not in speakers:
                        speakers.append(spk)
                    if len(speakers) >= 2:
                        break
            logger.info(f"[overlap] Expanded speakers list: {speakers}")

        overlap_info = {
            "start": start_abs,
            "end": end_abs,
            "duration": end_rel - start_rel,
            "speakers": speakers,
            "confidence": 1.0,
            "start_rel": start_rel,  # Keep relative times for SOS
            "end_rel": end_rel,
        }
        detected_overlaps.append(overlap_info)

    # Add to session timeline
    timeline = cfg.SESSION_OVERLAP_TIMELINE.setdefault(session_id, [])
    timeline.extend(detected_overlaps)

    # Keep only recent overlaps (last 30 seconds)
    cutoff = current_time - 30.0
    cfg.SESSION_OVERLAP_TIMELINE[session_id] = [
        o for o in timeline if o["end"] >= cutoff
    ]

    # --- Run SOS (Speaker Overlap Separation) on detected overlaps ---
    if detected_overlaps:
        separator = get_sos_separator()
        if separator is not None:
            try:
                # Convert relative timestamps to list of tuples for SOS
                overlap_regions = [(o["start_rel"], o["end_rel"]) for o in detected_overlaps]

                logger.info(f"Running SOS separation on {len(overlap_regions)} overlap regions")
                separated_regions = separator.separate_regions(buffer, sr, overlap_regions)

                # Store separated audio for this session
                session_separated = cfg.SESSION_SEPARATED_AUDIO.setdefault(session_id, {})

                # Map separated regions back to absolute timestamps
                for (start_rel, end_rel), speaker_waveforms in separated_regions.items():
                    start_abs = buffer_start_time + start_rel
                    end_abs = buffer_start_time + end_rel
                    session_separated[(start_abs, end_abs)] = speaker_waveforms
                    logger.info(f"[SOS] Separated {len(speaker_waveforms)} speakers for region [{start_abs:.2f}, {end_abs:.2f}]")

                # Also update the detected_overlaps with separated audio reference
                for ovlp in detected_overlaps:
                    key = (ovlp["start"], ovlp["end"])
                    if key in session_separated:
                        ovlp["has_separated_audio"] = True
                        ovlp["num_separated_speakers"] = len(session_separated[key])

            except Exception as e:
                logger.error(f"SOS separation error: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # Slide buffer forward (keep last 1 second for continuity)
    keep_samples = int(1.0 * sr)
    if buffer.shape[0] > keep_samples:
        new_start = buffer.shape[0] - keep_samples
        state["buffer"] = buffer[new_start:]
        state["buffer_start_time"] = buffer_start_time + (new_start / float(sr))

    return detected_overlaps if detected_overlaps else None


def transcribe_separated_audio(
    session_id: str,
    overlap_start: float,
    overlap_end: float,
    sr: int = 16000,
) -> Optional[List[Dict[str, Any]]]:
    """
    Transcribe separated audio for an overlap region.

    Returns:
        List of dicts with {speaker_idx, transcription, waveform_samples} for each separated speaker,
        or None if no separated audio available.
    """
    if not cfg.PIPELINE_AVAILABLE or cfg.pipeline is None:
        return None

    session_separated = cfg.SESSION_SEPARATED_AUDIO.get(session_id, {})

    # Find the closest matching overlap region (times might not match exactly)
    best_match = None
    best_diff = float('inf')
    for (start, end), waveforms in session_separated.items():
        diff = abs(start - overlap_start) + abs(end - overlap_end)
        if diff < best_diff and diff < 1.0:  # Within 1 second tolerance
            best_diff = diff
            best_match = (start, end)

    if best_match is None:
        logger.debug(f"No separated audio found for overlap [{overlap_start:.2f}, {overlap_end:.2f}]")
        return None

    speaker_waveforms = session_separated[best_match]
    results = []

    for speaker_idx, waveform in speaker_waveforms.items():
        try:
            # Ensure waveform is on the correct device
            device = cfg.pipeline.config.device
            if waveform.device != torch.device(device):
                waveform = waveform.to(device)

            # SepFormer outputs at 8kHz, need to resample to 16kHz for ASR
            if sr != 8000:
                import torchaudio
                # Waveform from SOS is at 8kHz
                resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sr)
                waveform = resampler(waveform.cpu()).to(device)

            # Run ASR on this separated stream
            duration = waveform.shape[-1] / sr
            segments = [{'start': 0.0, 'end': duration}]

            transcriptions = cfg.pipeline.asr.transcribe_segments(
                waveform.unsqueeze(0) if waveform.dim() == 1 else waveform,
                segments,
                return_timestamps=True
            )

            text = ""
            if transcriptions and len(transcriptions) > 0:
                text = transcriptions[0].get('text', '').strip()

            results.append({
                'speaker_idx': speaker_idx,
                'transcription': text,
                'waveform_samples': waveform.shape[-1],
            })

            logger.info(f"[SOS-ASR] Speaker {speaker_idx}: '{text}'")

        except Exception as e:
            logger.error(f"Failed to transcribe separated speaker {speaker_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return results if results else None
