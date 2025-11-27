"use client";

import React, { useState, useRef } from "react";
import { Upload, X, Play, Pause, File } from "lucide-react";
import { AudioFile } from "@/types/audio";

export default function AudioUploader() {
  const [audioFile, setAudioFile] = useState<AudioFile | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  function handleFile(file: File) {
    const url = URL.createObjectURL(file);
    setAudioFile({ name: file.name, url });
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }

  function clearFile() {
    if (audioFile) URL.revokeObjectURL(audioFile.url);
    setAudioFile(null);
    setIsPlaying(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function togglePlay() {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-4">
      <div className="w-full max-w-2xl space-y-8">
        <div className="text-center">
          <h1 className="text-5xl font-bold mb-4 text-gradient">Who says?</h1>
          <p className="text-lg text-muted cursor-default">
            Upload your audio file to get started
          </p>
        </div>

        {!audioFile ? (
          <div
            className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 transition-all duration-300 cursor-pointer transition-glow ${
              isDragging
                ? "border-primary bg-secondary glow-border"
                : "border-border hover:border-primary hover:bg-secondary glow-border"
            }`}
            onDragEnter={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setIsDragging(false);
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload
              className={`w-16 h-16 transition-colors duration-300 cursor-pointer ${
                isDragging ? "text-primary" : "text-muted"
              }`}
            />
            <p className="text-xl font-semibold mt-4 cursor-pointer">
              Drag & drop your file here
            </p>
            <p className="text-muted mt-2 cursor-pointer">or</p>
            <label
              htmlFor="file-upload"
              className="mt-4 cursor-pointer rounded-lg bg-primary px-6 py-3 text-sm font-semibold hover:bg-primary/90 glow-primary-hover transition-all duration-300"
              onClick={(e) => e.stopPropagation()}
            >
              Browse Files
              <input
                ref={fileInputRef}
                id="file-upload"
                type="file"
                accept="audio/*"
                className="sr-only"
                onChange={handleFileSelect}
              />
            </label>
          </div>
        ) : (
          <div className="p-6 bg-secondary rounded-2xl border border-border glow-border transition-glow">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center cursor-default">
                <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center mr-4 cursor-default">
                  <File className="w-6 h-6 text-primary" />
                </div>
                <h4 className="font-semibold truncate max-w-xs cursor-default">
                  {audioFile.name}
                </h4>
              </div>
              <button
                onClick={clearFile}
                className="text-muted hover:text-foreground transition-colors p-2 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex items-center justify-center">
              <button
                onClick={togglePlay}
                className="flex items-center justify-center w-16 h-16 bg-primary hover:bg-primary/90 rounded-full text-white transition-all duration-300 glow-primary-hover hover:scale-105 cursor-pointer"
              >
                {isPlaying ? (
                  <Pause className="w-8 h-8" />
                ) : (
                  <Play className="w-8 h-8 ml-1" />
                )}
              </button>
            </div>

            <audio
              ref={audioRef}
              src={audioFile.url}
              onEnded={() => setIsPlaying(false)}
              className="hidden"
            />
          </div>
        )}
      </div>
    </div>
  );
}
