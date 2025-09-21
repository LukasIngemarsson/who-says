import React from "react";
import { Home } from "./pages/Home";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-6xl p-4 md:p-8">
        <header className="mb-6 flex items-end justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">Who Says Editor</h1>
            <p className="text-slate-600 text-sm">Clean and correct WhisperX transcripts • Export JSON</p>
          </div>
        </header>
        <Home />
      </div>
    </div>
  );
}