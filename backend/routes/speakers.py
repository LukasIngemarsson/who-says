"""
Speaker enrollment and identification routes for the WhoSays backend.
"""
import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from flask import Blueprint, jsonify, request
from loguru import logger

import backend.config as cfg
from backend.helpers import similarity_to_confidence, squash_adjacent_short_repeats, convert_to_wav
from backend.speaker import save_embedding, assign_words_to_speakers
from backend.overlap import process_overlap_detection, transcribe_separated_audio
from backend.asr import get_live_snippet_for_session
from utils import load_audio_from_file, match_frequency

speakers_bp = Blueprint('speakers', __name__)


@speakers_bp.route("/upload_embeddings", methods=["POST"])
def upload_embeddings():
    if not cfg.PIPELINE_AVAILABLE:
        name = request.form.get("name", "Unknown")
        return jsonify({"message": f"Mock enrollment for {name}", "vector_size": 0})

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        speaker_name = request.form.get("name")
        if not speaker_name:
            return jsonify({"error": "Speaker 'name' is required"}), 400

        suffix = Path(str(file.filename)).suffix or ".webm"
        temp_file_path = None
        wav_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name

            wav_path = convert_to_wav(temp_file_path)

            waveform, sr = load_audio_from_file(wav_path, sr=16000)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.float().squeeze()

            device = cfg.pipeline.config.device
            waveform = waveform.to(device)

            with torch.inference_mode():
                embedding_tensor = cfg.pipeline.embedder.embed(waveform, sr)

            while embedding_tensor.dim() > 1:
                embedding_tensor = embedding_tensor.squeeze(0)

            embedding_tensor = F.normalize(embedding_tensor, p=2, dim=0)

            cfg.KNOWN_SPEAKERS[speaker_name] = embedding_tensor.cpu()
            save_embedding(speaker_name, embedding_tensor)

            return jsonify({
                "message": f"Enrolled {speaker_name}",
                "vector_size": str(tuple(embedding_tensor.shape)),
            })

        except Exception as e:
            import traceback
            logger.error(f"Error during upload_embeddings: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Error in upload_embeddings: {e}")
        return jsonify({"error": str(e)}), 500


@speakers_bp.route("/identify_speaker", methods=["POST"])
def identify_speaker():
    if not cfg.PIPELINE_AVAILABLE:
        return jsonify({"error": "Pipeline not available"}), 503

    try:
        if "audio_data" not in request.form:
            return jsonify({"error": "No audio_data provided"}), 400

        audio_data_b64 = request.form.get("audio_data")
        sample_rate = int(request.form.get("sample_rate"))
        session_id = request.form.get("session_id", "default")

        try:
            audio_bytes = base64.b64decode(audio_data_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

            waveform = torch.from_numpy(audio_array).float()
            # Ensure waveform is 1D (mono)
            while waveform.dim() > 1:
                if waveform.shape[0] <= 2:  # Likely channels dimension
                    waveform = waveform.mean(dim=0)
                else:
                    waveform = waveform.squeeze(0)
            if waveform.dim() == 0:
                waveform = waveform.unsqueeze(0)

            sr = sample_rate
            target_sr = cfg.pipeline.config.sr
            if sr != target_sr:
                wf_2d = waveform.unsqueeze(0)
                wf_2d = match_frequency(wf_2d, sr, sr=target_sr)
                waveform = wf_2d.squeeze(0)
                sr = target_sr

            device = cfg.pipeline.config.device
            waveform = waveform.to(device)

            # Final check: ensure waveform is 1D
            if waveform.dim() != 1:
                logger.warning(f"Unexpected waveform shape after processing: {waveform.shape}")
                while waveform.dim() > 1:
                    waveform = waveform.squeeze(0)

            # Log incoming audio duration
            incoming_sec = waveform.shape[0] / float(sr)
            logger.info(f"[identify_speaker] Incoming audio: {incoming_sec:.2f}s ({waveform.shape[0]} samples @ {sr}Hz)")

        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return jsonify({"error": "Invalid audio data"}), 400

        # --- ASR buffer per session ---
        if session_id not in cfg.SESSION_ASR_STATE:
            cfg.SESSION_ASR_STATE[session_id] = {
                "buffer": torch.zeros(0, device=device),
                "cursor": 0,
                # Absolute time (seconds) corresponding to buffer[0]
                "t0": 0.0,
                "last_asr_time": 0.0,
                "last_word_end": float("-inf"),
                "last_snippet": "",
                "prompt": "",
                "session_start_time": time.time(),  # Track when session started for warmup
            }
        asr_state = cfg.SESSION_ASR_STATE[session_id]

        # Helper: get speaker label respecting warmup period and speech accumulation
        def get_speaker_label_for_response():
            """Return speaker label, using None during warmup or if not enough speech yet."""
            session_start = asr_state.get("session_start_time", 0.0)
            elapsed = time.time() - session_start
            in_warmup = elapsed < cfg.SPEAKER_WARMUP_SEC

            # Also check if we've accumulated enough speech to make a proper identification
            # Don't show "Unknown" until we've had enough audio to actually try identification
            speech_buf_samples = cfg.SPEECH_BUFFER.shape[0] if cfg.SPEECH_BUFFER is not None else 0
            min_samples_for_id = int(cfg.MIN_SPEECH_SEC * sr)
            has_enough_speech = speech_buf_samples >= min_samples_for_id

            logger.info(f"[warmup] elapsed={elapsed:.2f}s, in_warmup={in_warmup}, speech_buf={speech_buf_samples}, min_needed={min_samples_for_id}, CURRENT_SPEAKER={cfg.CURRENT_SPEAKER}")

            if cfg.CURRENT_SPEAKER is not None:
                return cfg.CURRENT_SPEAKER
            elif in_warmup or not has_enough_speech:
                return None  # Don't show "Unknown" during warmup or before enough speech
            else:
                return "Unknown"

        # Ensure waveform is 1D for ASR buffer
        waveform_for_asr = waveform.detach()
        while waveform_for_asr.dim() > 1:
            waveform_for_asr = waveform_for_asr.squeeze(0)
        if waveform_for_asr.dim() == 1:
            asr_state["buffer"] = torch.cat([asr_state["buffer"], waveform_for_asr])

        # Cap ASR buffer (seconds) and keep an absolute time offset so
        # timestamps remain monotonic even when we trim old samples.
        MAX_ASR_BUFFER = int(sr * float(cfg.MAX_ASR_BUFFER_SEC))
        if asr_state["buffer"].shape[0] > MAX_ASR_BUFFER:
            overflow = asr_state["buffer"].shape[0] - MAX_ASR_BUFFER
            asr_state["cursor"] = max(0, asr_state["cursor"] - overflow)
            asr_state["t0"] = float(asr_state.get("t0", 0.0)) + (overflow / float(sr))
            asr_state["buffer"] = asr_state["buffer"][-MAX_ASR_BUFFER:]

        # --- Overlap detection buffer per session ---
        if cfg.OVERLAP_DETECTION_ENABLED:
            try:
                if session_id not in cfg.SESSION_OVERLAP_STATE:
                    cfg.SESSION_OVERLAP_STATE[session_id] = {
                        "buffer": torch.zeros(0, device=device),
                        "buffer_start_time": time.time(),
                        "last_detection_time": 0.0,
                    }

                overlap_state = cfg.SESSION_OVERLAP_STATE[session_id]
                # Ensure waveform is 1D before concatenating
                waveform_1d = waveform.detach().clone()
                while waveform_1d.dim() > 1:
                    waveform_1d = waveform_1d.squeeze(0)
                if waveform_1d.dim() == 0:
                    waveform_1d = waveform_1d.unsqueeze(0)

                # Verify it's 1D before concatenating
                if waveform_1d.dim() == 1:
                    overlap_state["buffer"] = torch.cat([
                        overlap_state["buffer"],
                        waveform_1d
                    ])

                    # Cap overlap buffer (keep max ~5 seconds to limit memory)
                    MAX_OVERLAP_BUFFER = int(sr * 5.0)
                    if overlap_state["buffer"].shape[0] > MAX_OVERLAP_BUFFER:
                        overflow = overlap_state["buffer"].shape[0] - MAX_OVERLAP_BUFFER
                        overlap_state["buffer"] = overlap_state["buffer"][-MAX_OVERLAP_BUFFER:]
                        overlap_state["buffer_start_time"] += overflow / float(sr)
            except Exception as e:
                logger.warning(f"Overlap buffer update failed: {e}")

        # --- Global rolling buffer (2s) + speech buffer (2.5s) ---
        if cfg.ROLLING_BUFFER is None:
            cfg.ROLLING_BUFFER = torch.zeros(sr * 2, device=device)  # 2 seconds for better VAD context
        if cfg.SPEECH_BUFFER is None:
            cfg.SPEECH_BUFFER = torch.zeros(0, device=device)

        chunk = waveform
        rolling_buffer_samples = sr * 2  # 2 seconds
        if chunk.shape[0] >= rolling_buffer_samples:
            cfg.ROLLING_BUFFER = chunk[-rolling_buffer_samples:]
        else:
            needed = rolling_buffer_samples - chunk.shape[0]
            cfg.ROLLING_BUFFER = torch.cat([cfg.ROLLING_BUFFER[-needed:], chunk])

        vad_input = cfg.ROLLING_BUFFER

        # VAD for diarization
        try:
            with torch.inference_mode():
                speech_segments = cfg.pipeline.vad(vad_input)
        except Exception as e:
            logger.error(f"VAD error in identify_speaker: {e}")
            return jsonify({"error": "VAD failure"}), 500

        debug_mode = request.args.get("debug") == "1"
        debug_payload = None
        if debug_mode:
            try:
                rms = float(torch.sqrt(torch.mean(vad_input.float() ** 2)).detach().cpu())
                mx = float(torch.max(torch.abs(vad_input.float())).detach().cpu())
            except Exception:
                rms, mx = None, None
            debug_payload = {
                "vad_threshold": float(getattr(cfg.pipeline.vad, "threshold", -1.0)),
                "vad_min_speech_duration_ms": int(getattr(cfg.pipeline.vad, "min_speech_duration_ms", -1)),
                "vad_min_silence_duration_ms": int(getattr(cfg.pipeline.vad, "min_silence_duration_ms", -1)),
                "vad_speech_pad_ms": int(getattr(cfg.pipeline.vad, "speech_pad_ms", -1)),
                "vad_segments": speech_segments,
                "vad_input_rms": rms,
                "vad_input_absmax": mx,
                "vad_input_samples": int(vad_input.shape[0]),
                "sr": int(sr),
            }

        if not speech_segments:
            # No speech detected - verify with next frame before confirming
            if not cfg.PREV_NO_SPEECH_DETECTED:
                # First frame with no speech: set flag and wait for next frame to verify
                cfg.PREV_NO_SPEECH_DETECTED = True
                logger.info("[identify_speaker] No speech detected, waiting for next frame to verify")
                # Return current state without updating - next frame will confirm
                snippet_obj = get_live_snippet_for_session(session_id, sr)
                live_text = snippet_obj.get("text") if isinstance(snippet_obj, dict) else (snippet_obj or "")
                ui_conf = similarity_to_confidence(cfg.CURRENT_CONFIDENCE)
                resp = {
                    "speaker": get_speaker_label_for_response(),
                    "has_speech": bool(live_text),
                    "confidence": ui_conf,
                    "transcript": live_text,
                    "overlap_detected": False,
                    "overlap_speakers": [],
                    "overlap_segments": [],
                }
                if debug_payload is not None:
                    resp["debug"] = debug_payload
                return jsonify(resp)

            # Second consecutive frame with no speech - confirmed no speech
            logger.info("[identify_speaker] No speech confirmed (two consecutive frames)")
            # If diarization-VAD misses speech, do NOT skip ASR.
            # Let the ASR-side VAD decide, so we don't drop words.
            snippet_obj = get_live_snippet_for_session(session_id, sr)
            live_text, transcript_segments = assign_words_to_speakers(
                session_id,
                snippet_obj,
                get_speaker_label_for_response() or "",
            )
            live_text = squash_adjacent_short_repeats(live_text)
            for seg in transcript_segments:
                seg["text"] = squash_adjacent_short_repeats(seg.get("text") or "")

            # Overlap detection even when no speech segments
            overlap_detected = False
            overlap_speakers_list: List[str] = []
            overlap_segments_list: List[Dict[str, Any]] = []
            if cfg.OVERLAP_DETECTION_ENABLED:
                try:
                    now_ovlp = time.time()
                    new_overlaps = process_overlap_detection(session_id, sr, now_ovlp)
                    if new_overlaps:
                        all_spk: set = set()
                        for ovlp in new_overlaps:
                            all_spk.update(ovlp.get("speakers", []))
                        overlap_speakers_list = list(all_spk)
                        overlap_segments_list = new_overlaps
                        # SOD detected overlap - report it even if we can't identify both speakers
                        # (speaker timeline only tracks one speaker at a time)
                        overlap_detected = True
                        logger.info(f"Overlap detected: {len(new_overlaps)} segments, speakers: {overlap_speakers_list}")
                except Exception as e:
                    logger.warning(f"Overlap detection failed: {e}")

            ui_conf = similarity_to_confidence(cfg.CURRENT_CONFIDENCE)
            resp = {
                "speaker": get_speaker_label_for_response(),
                "has_speech": bool(live_text),
                "confidence": ui_conf,
                "transcript": live_text,
                "overlap_detected": overlap_detected,
                "overlap_speakers": overlap_speakers_list,
                "overlap_segments": overlap_segments_list,
            }
            if transcript_segments:
                resp["transcript_segments"] = transcript_segments
            if debug_payload is not None:
                resp["debug"] = debug_payload
            return jsonify(resp)

        # Speech detected - reset the no-speech verification flag
        cfg.PREV_NO_SPEECH_DETECTED = False

        # Build speech buffer
        speech_portions: List[torch.Tensor] = []
        for seg in speech_segments:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            seg_audio = vad_input[start_sample:end_sample]
            if seg_audio.shape[0] > 0:
                speech_portions.append(seg_audio)

        if speech_portions:
            new_speech = torch.cat(speech_portions)
        else:
            new_speech = torch.zeros(0, device=device)

        cfg.SPEECH_BUFFER = torch.cat([cfg.SPEECH_BUFFER, new_speech])

        MAX_SPEECH_SAMPLES = int(cfg.MIN_SPEECH_SEC * sr)
        # Keep more audio for stable embeddings (up to 3x MIN_SPEECH_SEC, trim to 2x)
        if cfg.SPEECH_BUFFER.shape[0] > MAX_SPEECH_SAMPLES * 3:
            cfg.SPEECH_BUFFER = cfg.SPEECH_BUFFER[-MAX_SPEECH_SAMPLES * 2:]

        MIN_SPEECH_SAMPLES = int(max(0.3, float(cfg.MIN_SPEECH_SEC)) * float(sr))
        speech_buf_sec = cfg.SPEECH_BUFFER.shape[0] / float(sr)
        asr_buf_sec = asr_state["buffer"].shape[0] / float(sr)
        logger.info(f"[identify_speaker] Speech buffer: {speech_buf_sec:.2f}s, ASR buffer: {asr_buf_sec:.2f}s, min required: {cfg.MIN_SPEECH_SEC}s")

        if cfg.SPEECH_BUFFER.shape[0] < MIN_SPEECH_SAMPLES:
            ui_conf = similarity_to_confidence(cfg.CURRENT_CONFIDENCE)
            snippet_obj = get_live_snippet_for_session(session_id, sr)
            live_text = snippet_obj.get("text") if isinstance(snippet_obj, dict) else (snippet_obj or "")
            return jsonify({
                "speaker": get_speaker_label_for_response(),
                "has_speech": True,
                "confidence": ui_conf,
                "message": "Collecting more speech...",
                "transcript": live_text,
                "overlap_detected": False,
                "overlap_speakers": [],
                "overlap_segments": [],
            })

        # Compute embedding
        with torch.inference_mode():
            emb = cfg.pipeline.embedder.embed(cfg.SPEECH_BUFFER, sr)
        while emb.dim() > 1:
            emb = emb.squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)

        best_speaker = None
        best_score = -1.0
        second_best = -1.0

        if cfg.KNOWN_SPEAKERS:
            for name, ref_cpu in cfg.KNOWN_SPEAKERS.items():
                ref = F.normalize(ref_cpu.to(device), p=2, dim=0)
                score = torch.dot(emb, ref).item()
                logger.debug(f"Speaker {name}: score={score:.3f}")
                if score > best_score:
                    second_best = best_score
                    best_score = score
                    best_speaker = name
                elif score > second_best:
                    second_best = score
        else:
            logger.debug("No known speakers enrolled - cannot identify")

        now = time.time()
        MIN_CONFIDENCE = 0.21
        MARGIN_THRESHOLD = 0.09

        margin = best_score - second_best if second_best > -1.0 else best_score

        confident_speaker: Optional[str] = None
        if best_speaker is not None and best_score >= MIN_CONFIDENCE and margin >= MARGIN_THRESHOLD:
            confident_speaker = best_speaker
            logger.debug(f"Identified: {confident_speaker} (score={best_score:.3f}, margin={margin:.3f})")
        elif best_speaker is not None:
            logger.debug(f"Low confidence: {best_speaker} (score={best_score:.3f}, margin={margin:.3f}, min_conf={MIN_CONFIDENCE}, min_margin={MARGIN_THRESHOLD})")

        if confident_speaker is not None:
            # Track all identified speakers in this session (for overlap display)
            cfg.SESSION_IDENTIFIED_SPEAKERS.setdefault(session_id, set()).add(confident_speaker)

            if cfg.CURRENT_SPEAKER is None:
                cfg.CURRENT_SPEAKER = confident_speaker
                cfg.CURRENT_CONFIDENCE = best_score
                cfg.LAST_SWITCH_TIME = now
            else:
                if confident_speaker != cfg.CURRENT_SPEAKER:
                    if now - cfg.LAST_SWITCH_TIME >= cfg.SWITCH_COOLDOWN:
                        cfg.CURRENT_SPEAKER = confident_speaker
                        cfg.CURRENT_CONFIDENCE = best_score
                        cfg.LAST_SWITCH_TIME = now
                else:
                    ALPHA = 0.3  # higher alpha = faster response to actual matches
                    cfg.CURRENT_CONFIDENCE = ALPHA * best_score + (1 - ALPHA) * cfg.CURRENT_CONFIDENCE
                    cfg.LAST_SWITCH_TIME = now
        else:
            # Speech detected but no confident match - show as unknown
            # Only clear after cooldown to avoid flickering
            if cfg.CURRENT_SPEAKER is not None and now - cfg.LAST_SWITCH_TIME >= cfg.SWITCH_COOLDOWN:
                logger.info(f"[identify_speaker] Clearing speaker (low confidence: {best_score:.3f})")
                cfg.CURRENT_SPEAKER = None
                cfg.CURRENT_CONFIDENCE = 0.0
                cfg.LAST_SWITCH_TIME = now

        # Speaker timeline
        timeline = cfg.SESSION_SPEAKER_TIMELINE.setdefault(session_id, [])
        if cfg.CURRENT_SPEAKER is not None:
            if timeline and timeline[-1]["speaker"] == cfg.CURRENT_SPEAKER:
                timeline[-1]["end"] = now
            else:
                if timeline and timeline[-1].get("end") is None:
                    timeline[-1]["end"] = now
                timeline.append({"start": now, "end": now, "speaker": cfg.CURRENT_SPEAKER})

            WINDOW_SEC = 30.0
            cutoff = now - WINDOW_SEC
            while timeline and timeline[0].get("end") is not None and timeline[0]["end"] < cutoff:
                timeline.pop(0)

        ui_conf = similarity_to_confidence(cfg.CURRENT_CONFIDENCE)

        # Get speaker label (respects warmup period)
        current_speaker_label = get_speaker_label_for_response()

        # --- Overlap detection (batch processing) ---
        # IMPORTANT: Run BEFORE word assignment so overlap timeline is populated
        overlap_detected = False
        overlap_speakers: List[str] = []
        overlap_segments: List[Dict[str, Any]] = []

        if cfg.OVERLAP_DETECTION_ENABLED:
            try:
                now_ovlp = time.time()
                new_overlaps = process_overlap_detection(session_id, sr, now_ovlp)

                if new_overlaps:
                    # Collect unique speakers from all new overlaps
                    all_speakers: set = set()
                    for ovlp in new_overlaps:
                        all_speakers.update(ovlp.get("speakers", []))
                    overlap_speakers = list(all_speakers)
                    overlap_segments = new_overlaps
                    # SOD detected overlap - report it even if we can't identify both speakers
                    # (speaker timeline only tracks one speaker at a time)
                    overlap_detected = True
                    logger.info(f"Overlap detected: {len(new_overlaps)} segments, speakers: {overlap_speakers}")

                    # Transcribe separated audio for each overlap region
                    for ovlp in new_overlaps:
                        if ovlp.get("has_separated_audio"):
                            separated_transcripts = transcribe_separated_audio(
                                session_id,
                                ovlp["start"],
                                ovlp["end"],
                                sr=sr,
                            )
                            if separated_transcripts:
                                ovlp["separated_transcriptions"] = separated_transcripts
                                logger.info(f"[SOS] Got {len(separated_transcripts)} separated transcriptions for overlap [{ovlp['start']:.2f}, {ovlp['end']:.2f}]")
            except Exception as e:
                logger.warning(f"Overlap detection failed: {e}")

        snippet_obj = get_live_snippet_for_session(session_id, sr)
        live_text, transcript_segments = assign_words_to_speakers(
            session_id,
            snippet_obj,
            current_speaker_label or "",  # Use empty string to avoid None issues in function
        )

        live_text = squash_adjacent_short_repeats(live_text)
        for seg in transcript_segments:
            seg["text"] = squash_adjacent_short_repeats(seg.get("text") or "")

        # Debug: log transcript segments with speakers
        if transcript_segments:
            logger.info(f"[transcript_segments] {[(s.get('speaker'), s.get('text')[:30] + '...' if len(s.get('text', '')) > 30 else s.get('text')) for s in transcript_segments]}")

        # Use current_speaker_label already calculated above (respects warmup period)
        resp: Dict[str, Any] = {
            "speaker": current_speaker_label,  # None during warmup, "Unknown" after
            "has_speech": bool(live_text),
            "confidence": ui_conf,
            "transcript": live_text,
            "overlap_detected": overlap_detected,
            "overlap_speakers": overlap_speakers,
            "overlap_segments": overlap_segments,
        }
        if transcript_segments:
            resp["transcript_segments"] = transcript_segments

        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error in identify_speaker: {e}")
        return jsonify({"error": str(e)}), 500


@speakers_bp.route("/correct_speaker", methods=["POST"])
def correct_speaker():
    return jsonify({
        "error": "Speaker corrections are disabled during recording. Please enroll speakers before starting."
    }), 400
