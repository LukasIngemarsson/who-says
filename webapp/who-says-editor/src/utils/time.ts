export function formatTime(s?: number | string | null) {
  if (s === undefined || s === null || isNaN(Number(s))) return "";
  const t = Math.max(0, Number(s));
  const ms = Math.floor((t % 1) * 1000).toString().padStart(3, "0");
  const sec = Math.floor(t) % 60;
  const min = Math.floor(t / 60) % 60;
  const hr = Math.floor(t / 3600);
  const base = `${hr > 0 ? hr + ":" : ""}${hr > 0 ? String(min).padStart(2, "0") : min}:${String(sec).padStart(2, "0")}`;
  return `${base}.${ms}`;
}

export function parseTime(v: string) {
  if (!v) return 0;
  const s = v.trim();
  if (s.includes(":")) {
    const parts = s.split(":").map(Number);
    let seconds = 0;
    while (parts.length) seconds = seconds * 60 + (parts.shift() || 0);
    return seconds;
  }
  const n = Number(s);
  return Number.isFinite(n) ? n : 0;
}