"use client";

import { useEffect, useRef, useState } from "react";

type AnimatedCounterProps = {
  value: number;
  durationMs?: number;
  format: (value: number) => string;
};

export function AnimatedCounter({
  value,
  durationMs = 900,
  format
}: AnimatedCounterProps) {
  const [displayValue, setDisplayValue] = useState(value);
  const previousValue = useRef(value);

  useEffect(() => {
    const startValue = previousValue.current;
    const delta = value - startValue;
    const startTime = performance.now();

    let frame = 0;
    const tick = (time: number) => {
      const progress = Math.min((time - startTime) / durationMs, 1);
      const eased = 1 - (1 - progress) * (1 - progress);
      setDisplayValue(startValue + delta * eased);
      if (progress < 1) {
        frame = requestAnimationFrame(tick);
      } else {
        previousValue.current = value;
      }
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [durationMs, value]);

  return <>{format(displayValue)}</>;
}
