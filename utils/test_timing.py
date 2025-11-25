#!/usr/bin/env python3
"""
Simple test script to verify timing functionality.
"""

from pathlib import Path
from main import WhoSays
from utils import format_timing_report

def test_timing():
    """Test timing feature with a sample audio file."""
    print("Testing timing feature...")

    # Initialize pipeline
    pipeline = WhoSays()

    # Use a short sample file for testing
    audio_file = Path("samples/single_speaker_sample.wav")

    if not audio_file.exists():
        print(f"Error: Test audio file not found: {audio_file}")
        return False

    print(f"Processing: {audio_file}")

    # Test without timing
    print("\n" + "="*60)
    print("Test 1: Without timing (include_timing=False)")
    print("="*60)
    result1 = pipeline(str(audio_file), num_speakers=2, include_timing=False)

    if 'timing' in result1:
        print("FAIL: timing should not be in result when include_timing=False")
        return False
    else:
        print("PASS: No timing data included")

    # Test with timing
    print("\n" + "="*60)
    print("Test 2: With timing (include_timing=True)")
    print("="*60)
    result2 = pipeline(str(audio_file), num_speakers=2, include_timing=True)

    if 'timing' not in result2:
        print("FAIL: timing should be in result when include_timing=True")
        return False

    if 'total_time' not in result2:
        print("FAIL: total_time should be in result when include_timing=True")
        return False

    print("PASS: Timing data included")

    # Verify timing components
    expected_components = ['audio_loading', 'vad', 'asr', 'scd', 'embedding', 'clustering', 'formatting']
    timing = result2['timing']

    print(f"\nTiming components found: {list(timing.keys())}")

    missing_components = [comp for comp in expected_components if comp not in timing]
    if missing_components:
        print(f"WARNING: Missing timing components: {missing_components}")

    # Display timing report
    print("\n" + "="*60)
    print("Test 3: Timing Report Visualization")
    print("="*60)
    report = format_timing_report(result2['timing'], result2['total_time'])
    print(report)

    # Verify all times are positive
    for component, time_val in timing.items():
        if time_val < 0:
            print(f"FAIL: Negative time for {component}: {time_val}")
            return False

    print("\nPASS: All timing values are positive")

    # Verify total time matches sum of components
    total_calculated = sum(timing.values())
    total_stored = result2['total_time']

    if abs(total_calculated - total_stored) > 0.001:  # Allow small floating point differences
        print(f"FAIL: Total time mismatch. Calculated: {total_calculated}, Stored: {total_stored}")
        return False

    print(f"PASS: Total time matches (sum: {total_calculated:.3f}s, stored: {total_stored:.3f}s)")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_timing()
    exit(0 if success else 1)
