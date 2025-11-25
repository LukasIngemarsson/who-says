import React, { forwardRef } from "react";

type Props = { className?: string };
export const AudioPlayer = forwardRef<HTMLAudioElement, Props>(({ className }, ref) => {
  return <audio ref={ref} className={(className ?? "") + " w-full"} controls preload="metadata" />;
});
AudioPlayer.displayName = "AudioPlayer";