import { describe, it, expect } from "vitest";
import { parseTime, formatTime } from "@/utils/time";


describe("time utils", () => {
it("parses seconds as number", () => {
expect(parseTime("62.345")).toBeCloseTo(62.345, 3);
});
it("parses m:ss.mmm format", () => {
expect(parseTime("1:02.345")).toBeCloseTo(62.345, 3);
});
it("parses h:mm:ss format", () => {
expect(parseTime("1:02:03")).toBe(3723);
});
it("formats seconds to m:ss.mmm", () => {
expect(formatTime(62.345)).toBe("1:02.345");
});
it("formats with hours when needed", () => {
expect(formatTime(3661.2)).toBe("1:01:01.200");
});
});