import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { SegmentTable } from "../components/SegmentTable";
import type { Segment } from "../types/whisperx";


const demoSeg: Segment = {
id: 0,
start: 0,
end: 1.5,
text: "hello",
speaker: "SPEAKER_00",
words: []
};


describe("SegmentTable", () => {
it("renders and updates speaker field", () => {
const onChange = vi.fn();
render(
<table><tbody>
<SegmentTable
segments={[demoSeg]}
selected={0}
onSelect={() => {}}
onChange={onChange}
onInsertAfter={() => {}}
onRemove={() => {}}
/>
</tbody></table>
);
const speakerInput = screen.getByDisplayValue("SPEAKER_00") as HTMLInputElement;
fireEvent.change(speakerInput, { target: { value: "SPEAKER_01" } });
expect(onChange).toHaveBeenCalledWith(0, { speaker: "SPEAKER_01" });
});
});