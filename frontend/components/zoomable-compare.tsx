"use client";

import { motion } from "framer-motion";
import { MouseEvent, useEffect, useRef, useState } from "react";

type ZoomableCompareProps = {
  inputPreview: string | null;
  outputPreview: string | null;
};

const lensSize = 136;

export function ZoomableCompare({ inputPreview, outputPreview }: ZoomableCompareProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [compareValue, setCompareValue] = useState(50);
  const [showLens, setShowLens] = useState(false);
  const [lensPosition, setLensPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!outputPreview) {
      setCompareValue(50);
      return;
    }

    let frame = 0;
    const duration = 1600;
    const start = performance.now();

    const animate = (time: number) => {
      const progress = Math.min((time - start) / duration, 1);
      const wave = Math.sin(progress * Math.PI);
      setCompareValue(18 + wave * 64);
      if (progress < 1) {
        frame = requestAnimationFrame(animate);
      } else {
        setCompareValue(50);
      }
    };

    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [outputPreview]);

  function updateLens(event: MouseEvent<HTMLDivElement>) {
    if (!containerRef.current) {
      return;
    }
    const bounds = containerRef.current.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * 100;
    const y = ((event.clientY - bounds.top) / bounds.height) * 100;
    setLensPosition({
      x: Math.min(Math.max(x, 8), 92),
      y: Math.min(Math.max(y, 8), 92)
    });
  }

  if (!outputPreview) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-ink/45">
        EPNet output will appear here
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full overflow-hidden"
      onMouseMove={updateLens}
      onMouseEnter={() => setShowLens(true)}
      onMouseLeave={() => setShowLens(false)}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={outputPreview}
        alt="Super-resolved output"
        data-testid="output-image"
        className="absolute inset-0 h-full w-full object-cover transition duration-500 hover:scale-[1.04]"
      />
      {inputPreview ? (
        <>
          <div
            className="absolute inset-y-0 left-0 overflow-hidden"
            style={{ width: `${compareValue}%` }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={inputPreview}
              alt="Input comparison layer"
              className="absolute inset-0 h-full w-full object-cover"
              style={{ width: "100%", minWidth: "100%" }}
            />
          </div>
          <div
            className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(19,32,47,0.15)]"
            style={{ left: `${compareValue}%` }}
          />
          <div className="absolute inset-x-4 bottom-4">
            <input
              type="range"
              min={0}
              max={100}
              value={compareValue}
              onChange={(event) => setCompareValue(Number(event.target.value))}
              className="w-full"
              aria-label="Compare input and output"
              data-testid="compare-slider"
            />
          </div>
        </>
      ) : null}

      {showLens ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="pointer-events-none absolute rounded-full border-2 border-white shadow-[0_18px_45px_rgba(19,32,47,0.22)]"
          style={{
            width: lensSize,
            height: lensSize,
            left: `calc(${lensPosition.x}% - ${lensSize / 2}px)`,
            top: `calc(${lensPosition.y}% - ${lensSize / 2}px)`,
            backgroundImage: `url(${outputPreview})`,
            backgroundPosition: `${lensPosition.x}% ${lensPosition.y}%`,
            backgroundRepeat: "no-repeat",
            backgroundSize: "240%"
          }}
        />
      ) : null}
    </div>
  );
}
