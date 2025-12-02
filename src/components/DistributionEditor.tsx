import { useCallback, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';

type Point = {
  x: number;
  y: number;
};

type DragTarget =
  | { type: 'point'; index: number }
  | { type: 'bound'; side: 'll' | 'ul' }
  | null;

const WIDTH = 780;
const HEIGHT = 420;
const MARGIN = { top: 32, right: 28, bottom: 48, left: 64 };
const POINT_RADIUS = 7;
const BOUND_HANDLE_WIDTH = 12;
const MIN_GAP = 0.05;

export interface DistributionEditorProps {
  points: Point[];
  onChange(points: Point[]): void;
  lowerBound: number;
  upperBound: number;
  onChangeBounds(bounds: { lower: number; upper: number }): void;
}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

export function DistributionEditor({
  points,
  onChange,
  lowerBound,
  upperBound,
  onChangeBounds
}: DistributionEditorProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [dragTarget, setDragTarget] = useState<DragTarget>(null);
  const activePointerId = useRef<number | null>(null);

  const xValues = points.map((point) => point.x).concat([lowerBound, upperBound, 0]);
  const yValues = points.map((point) => point.y);

  const xExtent = useMemo(() => {
    const min = Math.min(...xValues);
    const max = Math.max(...xValues);
    const padding = (max - min || 1) * 0.1;
    return { min: min - padding, max: max + padding };
  }, [xValues]);

  const yExtent = useMemo(() => {
    const min = Math.min(0, ...yValues);
    const max = Math.max(1, ...yValues);
    const padding = (max - min || 1) * 0.1;
    return { min: min - padding, max: max + padding };
  }, [yValues]);

  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;

  const projectX = useCallback(
    (value: number) =>
      MARGIN.left + ((value - xExtent.min) / (xExtent.max - xExtent.min)) * innerWidth,
    [innerWidth, xExtent.min, xExtent.max]
  );

  const projectY = useCallback(
    (value: number) =>
      HEIGHT - MARGIN.bottom - ((value - yExtent.min) / (yExtent.max - yExtent.min)) * innerHeight,
    [innerHeight, yExtent.min, yExtent.max]
  );

  const coordsFromPointer = useCallback(
    (clientX: number, clientY: number) => {
      const svg = svgRef.current;
      if (!svg) {
        return null;
      }

      const rect = svg.getBoundingClientRect();
      const scaleX = WIDTH / rect.width;
      const scaleY = HEIGHT / rect.height;

      const rawX = ((clientX - rect.left) * scaleX - MARGIN.left) / innerWidth;
      const rawY = ((rect.bottom - clientY) * scaleY - MARGIN.bottom) / innerHeight;
      const ratioX = clamp(rawX, 0, 1);
      const ratioY = clamp(rawY, 0, 1);

      return {
        x: xExtent.min + ratioX * (xExtent.max - xExtent.min),
        y: yExtent.min + ratioY * (yExtent.max - yExtent.min)
      };
    },
    [innerHeight, innerWidth, xExtent.max, xExtent.min, yExtent.max, yExtent.min]
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (!dragTarget || activePointerId.current !== event.pointerId) {
        return;
      }

      const coords = coordsFromPointer(event.clientX, event.clientY);
      if (!coords) {
        return;
      }

      if (dragTarget.type === 'point') {
        const { index } = dragTarget;
        const nextPoints = points.map((p) => ({ ...p }));
        const minX = index === 0 ? xExtent.min : nextPoints[index - 1].x + MIN_GAP;
        const maxX = index === nextPoints.length - 1 ? xExtent.max : nextPoints[index + 1].x - MIN_GAP;
        nextPoints[index] = {
          x: clamp(coords.x, minX, maxX),
          y: clamp(coords.y, 0, yExtent.max)
        };
        onChange(nextPoints);
      } else if (dragTarget.type === 'bound') {
        if (dragTarget.side === 'll') {
          const clamped = Math.min(clamp(coords.x, xExtent.min, xExtent.max), upperBound - MIN_GAP);
          onChangeBounds({ lower: clamped, upper: upperBound });
        } else {
          const clamped = Math.max(clamp(coords.x, xExtent.min, xExtent.max), lowerBound + MIN_GAP);
          onChangeBounds({ lower: lowerBound, upper: clamped });
        }
      }
    },
    [coordsFromPointer, dragTarget, lowerBound, onChange, onChangeBounds, points, upperBound, xExtent.max, xExtent.min, yExtent.max, yExtent.min]
  );

  const handlePointerUp = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (activePointerId.current !== event.pointerId) {
        return;
      }

      const svg = svgRef.current;
      if (svg && svg.hasPointerCapture(event.pointerId)) {
        try {
          svg.releasePointerCapture(event.pointerId);
        } catch {
          // ignore release errors
        }
      }

      activePointerId.current = null;
      setDragTarget(null);
    },
    []
  );

  const startDrag = useCallback(
    (target: DragTarget) => (event: ReactPointerEvent<SVGElement>) => {
      event.preventDefault();
      event.stopPropagation();
      setDragTarget(target);
      activePointerId.current = event.pointerId;

      const svg = svgRef.current ?? event.currentTarget.ownerSVGElement;
      if (svg) {
        try {
          svg.setPointerCapture(event.pointerId);
        } catch {
          // ignore capture errors
        }
      }
    },
    []
  );

  const pathD = useMemo(() => {
    return points
      .map((point, index) => `${index === 0 ? 'M' : 'L'} ${projectX(point.x)} ${projectY(point.y)}`)
      .join(' ');
  }, [points, projectX, projectY]);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width="100%"
      height="auto"
      role="presentation"
      aria-label="Interactive distribution editor"
      className="distribution-editor"
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <defs>
        <clipPath id="plot-area">
          <rect
            x={MARGIN.left}
            y={MARGIN.top}
            width={innerWidth}
            height={innerHeight}
            rx={12}
          />
        </clipPath>
      </defs>

      <rect
        x={MARGIN.left}
        y={MARGIN.top}
        width={innerWidth}
        height={innerHeight}
        fill="rgba(255, 77, 0, 0.05)"
        stroke="rgba(255, 77, 0, 0.2)"
        rx={12}
      />

      <rect
        x={projectX(lowerBound)}
        y={MARGIN.top}
        width={projectX(upperBound) - projectX(lowerBound)}
        height={innerHeight}
        fill="rgba(255, 77, 0, 0.12)"
        clipPath="url(#plot-area)"
      />

      <line
        x1={projectX(0)}
        x2={projectX(0)}
        y1={MARGIN.top}
        y2={HEIGHT - MARGIN.bottom}
        stroke="rgba(255, 77, 0, 0.4)"
        strokeDasharray="6 4"
        strokeWidth={2}
        clipPath="url(#plot-area)"
      />

      <path
        d={pathD}
        fill="none"
        stroke="url(#gradient-line)"
        strokeWidth={3}
        clipPath="url(#plot-area)"
      />

      <defs>
        <linearGradient id="gradient-line" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#ff4d00" />
          <stop offset="100%" stopColor="#ff9100" />
        </linearGradient>
      </defs>

      {points.map((point, index) => (
        <g key={`control-${index}`}>
          <circle
            cx={projectX(point.x)}
            cy={projectY(point.y)}
            r={POINT_RADIUS}
            className="distribution-point"
            onPointerDown={startDrag({ type: 'point', index })}
          />
          <text
            x={projectX(point.x)}
            y={projectY(point.y) - 12}
            textAnchor="middle"
            className="distribution-label"
          >
            {`${point.x.toFixed(2)}, ${point.y.toFixed(2)}`}
          </text>
        </g>
      ))}

      {[{ value: lowerBound, label: 'LL', side: 'll' }, { value: upperBound, label: 'UL', side: 'ul' }].map(
        ({ value, label, side }) => (
          <g key={side}>
            <line
              x1={projectX(value)}
              x2={projectX(value)}
              y1={MARGIN.top}
              y2={HEIGHT - MARGIN.bottom}
              stroke="rgba(255, 77, 0, 0.8)"
              strokeDasharray="6 4"
              strokeWidth={2}
            />
            <rect
              x={projectX(value) - BOUND_HANDLE_WIDTH / 2}
              y={MARGIN.top}
              width={BOUND_HANDLE_WIDTH}
              height={innerHeight}
              rx={6}
              className="bound-handle"
              fill="transparent"
              onPointerDown={startDrag({ type: 'bound', side: side as 'll' | 'ul' })}
            />
            <text
              x={projectX(value)}
              y={MARGIN.top - 8}
              textAnchor="middle"
              className="distribution-label"
            >
              {`${label} = ${value.toFixed(2)}`}
            </text>
          </g>
        )
      )}

      <g className="axis-labels">
        <text x={WIDTH / 2} y={HEIGHT - 4} textAnchor="middle" className="axis-label">
          Work value
        </text>
        <text
          x={16}
          y={HEIGHT / 2}
          textAnchor="middle"
          transform={`rotate(-90 16 ${HEIGHT / 2})`}
          className="axis-label"
        >
          Density (relative)
        </text>
      </g>
    </svg>
  );
}

export default DistributionEditor;
