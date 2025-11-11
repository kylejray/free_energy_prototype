import { ChangeEvent, Fragment, useEffect, useState } from 'react';
import { fetchHealth, runNotebookAnalysis, NotebookRequestPayload } from './client';
import DistributionEditor from './components/DistributionEditor';

type DistributionPoint = {
  x: number;
  y: number;
};

type AnalysisSection = 'sampling' | 'free_energy' | 'standard';

type NotebookResultState = {
  samplingTopPlot: string | null;
  samplingBottomPlot: string | null;
  freeEnergyTopPlot: string | null;
  freeEnergyBottomPlot: string | null;
  freeEnergyStandardPlot: string | null;
  metadata: Record<string, number> | null;
};

type Bounds = {
  lower: number;
  upper: number;
};

type AnalysisPreset = {
  id: string;
  label: string;
  points: DistributionPoint[];
  bounds: Bounds;
};

type NumericRange = {
  min: number;
  max: number;
  step: number;
};

const DEFAULT_ANALYSIS_POINTS: DistributionPoint[] = [
  { x: 3, y: 0 },
  { x: 3.5, y: 0.4 },
  { x: 4.5, y: 0.4 },
  { x: 5, y: 0 },
  { x: 6, y: 0 },
  { x: 6.5, y: 1 },
  { x: 7.5, y: 1 },
  { x: 8, y: 0 }
];

const DEFAULT_ANALYSIS_BOUNDS: Bounds = {
  lower: 0,
  upper: 5
};

const CUSTOM_PRESET_ID = 'custom';

const ANALYSIS_PRESETS: AnalysisPreset[] = [
  {
    id: 'baseline',
    label: 'Baseline',
    points: DEFAULT_ANALYSIS_POINTS,
    bounds: DEFAULT_ANALYSIS_BOUNDS
  },
  {
    id: 'bimodal',
    label: 'Equal Reverse',
    points: [
      { x: -2.4, y: 0 },
      { x: -2.35, y: 0.05 },
      { x: -2.25, y: 0.05 },
      { x: -2.2, y: 0 },
      { x: 0.54, y: 0 },
      { x: 0.59, y: 0.95 },
      { x: 0.69, y: 0.95 },
      { x: 0.74, y: 0 }
    ],
    bounds: { lower: 0, upper: 2.5 }
  }
];

const DEFAULT_ANALYSIS_SAMPLE_SIZE = 25;
const DEFAULT_ANALYSIS_TRIALS = 50;

const ANALYSIS_SAMPLE_SIZE_RANGE: NumericRange = {
  min: 5,
  max: 1_000,
  step: 5
};

const ANALYSIS_TRIALS_RANGE: NumericRange = {
  min: 10,
  max: 500,
  step: 10
};

const clampWithStep = (value: number, range: NumericRange): number => {
  const { min, max, step } = range;
  const bounded = Math.min(Math.max(value, min), max);
  if (!step) {
    return Math.round(bounded);
  }
  const steps = Math.round((bounded - min) / step);
  return min + steps * step;
};

function App() {
  const [apiStatus, setApiStatus] = useState<string>('Checking backend...');

  const [analysisPoints, setAnalysisPointsState] = useState<DistributionPoint[]>(() =>
    ANALYSIS_PRESETS[0].points.map((point) => ({ ...point }))
  );
  const [analysisBounds, setAnalysisBoundsState] = useState<Bounds>(() => ({
    ...ANALYSIS_PRESETS[0].bounds
  }));
  const [analysisSampleSize, setAnalysisSampleSize] = useState<number>(
    DEFAULT_ANALYSIS_SAMPLE_SIZE
  );
  const [analysisTrials, setAnalysisTrials] = useState<number>(DEFAULT_ANALYSIS_TRIALS);
  const [analysisLoading, setAnalysisLoading] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<NotebookResultState | null>(null);
  const [activeAnalysisSection, setActiveAnalysisSection] = useState<AnalysisSection | null>(null);
  const [activePresetId, setActivePresetId] = useState<string>(
    ANALYSIS_PRESETS[0]?.id ?? CUSTOM_PRESET_ID
  );

  const setAnalysisPoints = (nextPoints: DistributionPoint[]) => {
    setActivePresetId(CUSTOM_PRESET_ID);
    setAnalysisPointsState(nextPoints);
  };

  const setAnalysisBounds = (nextBounds: Bounds) => {
    setActivePresetId(CUSTOM_PRESET_ID);
    setAnalysisBoundsState(nextBounds);
  };

  const applyPreset = (presetId: string) => {
    const preset = ANALYSIS_PRESETS.find((item) => item.id === presetId);
    if (!preset) {
      return;
    }
    setActivePresetId(presetId);
    setAnalysisPointsState(preset.points.map((point) => ({ ...point })));
    setAnalysisBoundsState({ ...preset.bounds });
    setAnalysisError('');
    setAnalysisResult(null);
    setActiveAnalysisSection(null);
  };

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      try {
        const payload = await fetchHealth();
        if (cancelled) {
          return;
        }
        setApiStatus(`Backend is up. Message: ${payload.message}`);
      } catch (error) {
        if (!cancelled) {
          setApiStatus('Backend check failed.');
        }
      }
    };

    bootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleAnalysisSampleSizeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const next = clampWithStep(Number(event.target.value), ANALYSIS_SAMPLE_SIZE_RANGE);
    setAnalysisSampleSize(next);
  };

  const handleAnalysisTrialsChange = (event: ChangeEvent<HTMLInputElement>) => {
    const next = clampWithStep(Number(event.target.value), ANALYSIS_TRIALS_RANGE);
    setAnalysisTrials(next);
  };

  const handleAnalysisSubmit = async (section: AnalysisSection) => {
    const xpList = analysisPoints.map((point) => point.x);
    const fpList = analysisPoints.map((point) => point.y);
    const { lower, upper } = analysisBounds;

    if (xpList.length < 2) {
      setAnalysisError('At least two control points are required.');
      return;
    }

    if (upper <= lower) {
      setAnalysisError('Ensure UL is greater than LL.');
      return;
    }

    const sanitizedSampleSize = clampWithStep(analysisSampleSize, ANALYSIS_SAMPLE_SIZE_RANGE);
    const sanitizedTrials = clampWithStep(analysisTrials, ANALYSIS_TRIALS_RANGE);

    if (sanitizedSampleSize !== analysisSampleSize) {
      setAnalysisSampleSize(sanitizedSampleSize);
    }
    if (sanitizedTrials !== analysisTrials) {
      setAnalysisTrials(sanitizedTrials);
    }

    setAnalysisLoading(true);
    setAnalysisError('');
    setActiveAnalysisSection(section);
    try {
      const payload: NotebookRequestPayload = {
        xp: xpList,
        fp: fpList,
        ll: lower,
        ul: upper,
        section,
        sampleSize: sanitizedSampleSize,
        trials: sanitizedTrials
      };
      const result = await runNotebookAnalysis(payload);
      setAnalysisResult((previous) => {
        const base: NotebookResultState = previous ?? {
          samplingTopPlot: null,
          samplingBottomPlot: null,
          freeEnergyTopPlot: null,
          freeEnergyBottomPlot: null,
          freeEnergyStandardPlot: null,
          metadata: null
        };

        const next: NotebookResultState = { ...base };

        if (typeof result.samplingTopPlot === 'string') {
          next.samplingTopPlot = result.samplingTopPlot;
        }
        if (typeof result.samplingBottomPlot === 'string') {
          next.samplingBottomPlot = result.samplingBottomPlot;
        }
        if (typeof result.freeEnergyTopPlot === 'string') {
          next.freeEnergyTopPlot = result.freeEnergyTopPlot;
        }
        if (typeof result.freeEnergyBottomPlot === 'string') {
          next.freeEnergyBottomPlot = result.freeEnergyBottomPlot;
        }
        if (typeof result.freeEnergyStandardPlot === 'string') {
          next.freeEnergyStandardPlot = result.freeEnergyStandardPlot;
        }
        if (result.metadata && Object.keys(result.metadata).length > 0) {
          next.metadata = result.metadata;
        }

        return next;
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error.';
      setAnalysisError(message);
    } finally {
      setAnalysisLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="panel">
        <header className="panel-header">
          <h1>Free Energy Prototype Suite</h1>
        </header>

        <p className="status">{apiStatus}</p>
        <div className="analysis-container">
          <div className="analysis-editor">
            <DistributionEditor
              points={analysisPoints}
              onChange={setAnalysisPoints}
              lowerBound={analysisBounds.lower}
              upperBound={analysisBounds.upper}
              onChangeBounds={setAnalysisBounds}
            />

            <div className="analysis-toolbar">
              <div className="preset-switcher" role="group" aria-label="distribution presets">
                {ANALYSIS_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    type="button"
                    className={`preset-btn${activePresetId === preset.id ? ' active' : ''}`}
                    onClick={() => applyPreset(preset.id)}
                    disabled={analysisLoading}
                    aria-pressed={activePresetId === preset.id}
                  >
                    {preset.label}
                  </button>
                ))}
                {activePresetId === CUSTOM_PRESET_ID ? (
                  <span className="preset-indicator">Custom</span>
                ) : null}
              </div>
              <div className="bounds-display">
                <span>LL: {analysisBounds.lower.toFixed(2)}</span>
                <span>UL: {analysisBounds.upper.toFixed(2)}</span>
              </div>
              <div className="analysis-parameters" role="group" aria-label="analysis parameters">
                <label className="analysis-parameter" htmlFor="analysis-sample-size">
                  <div className="parameter-header">
                    <span>Samples</span>
                    <span>{analysisSampleSize.toLocaleString()}</span>
                  </div>
                  <input
                    id="analysis-sample-size"
                    type="range"
                    min={ANALYSIS_SAMPLE_SIZE_RANGE.min}
                    max={ANALYSIS_SAMPLE_SIZE_RANGE.max}
                    step={ANALYSIS_SAMPLE_SIZE_RANGE.step}
                    value={analysisSampleSize}
                    onChange={handleAnalysisSampleSizeChange}
                    disabled={analysisLoading}
                  />
                </label>
                <label className="analysis-parameter" htmlFor="analysis-trials">
                  <div className="parameter-header">
                    <span>Trials</span>
                    <span>{analysisTrials}</span>
                  </div>
                  <input
                    id="analysis-trials"
                    type="range"
                    min={ANALYSIS_TRIALS_RANGE.min}
                    max={ANALYSIS_TRIALS_RANGE.max}
                    step={ANALYSIS_TRIALS_RANGE.step}
                    value={analysisTrials}
                    onChange={handleAnalysisTrialsChange}
                    disabled={analysisLoading}
                  />
                </label>
              </div>
              <div className="analysis-actions">
                <button
                  type="button"
                  className="cta"
                  onClick={() => handleAnalysisSubmit('sampling')}
                  disabled={analysisLoading}
                  aria-pressed={activeAnalysisSection === 'sampling'}
                >
                  {analysisLoading && activeAnalysisSection === 'sampling'
                    ? 'Generating…'
                    : 'Generate Sampling Diagnostics'}
                </button>
                <button
                  type="button"
                  className="cta"
                  onClick={() => handleAnalysisSubmit('free_energy')}
                  disabled={analysisLoading}
                  aria-pressed={activeAnalysisSection === 'free_energy'}
                >
                  {analysisLoading && activeAnalysisSection === 'free_energy'
                    ? 'Generating…'
                    : 'Generate Trajectory Class Estimates'}
                </button>
                <button
                  type="button"
                  className="cta"
                  onClick={() => handleAnalysisSubmit('standard')}
                  disabled={analysisLoading}
                  aria-pressed={activeAnalysisSection === 'standard'}
                >
                  {analysisLoading && activeAnalysisSection === 'standard'
                    ? 'Generating…'
                    : 'Generate Variance Diagnostics'}
                </button>
              </div>
            </div>
          </div>

          {analysisError ? <p className="error">{analysisError}</p> : null}

          <div className="analysis-groups">
            <section className="analysis-group">
              <header className="analysis-group-header">
                <h2>Sampling Diagnostics</h2>
              </header>
              {analysisResult?.samplingTopPlot || analysisResult?.samplingBottomPlot ? (
                <div className="analysis-group-images">
                  {analysisResult?.samplingTopPlot ? (
                    <figure className="analysis-image-card">
                      <img
                        src={`data:image/png;base64,${analysisResult.samplingTopPlot}`}
                        alt="Histograms of forward and reverse samples"
                      />
                      <figcaption className="image-caption">Histograms & joint PDFs</figcaption>
                    </figure>
                  ) : null}
                  {analysisResult?.samplingBottomPlot ? (
                    <figure className="analysis-image-card">
                      <img
                        src={`data:image/png;base64,${analysisResult.samplingBottomPlot}`}
                        alt="PDF overlays comparing F(x) and R(-x)"
                      />
                      <figcaption className="image-caption">PDF overlays (linear & log)</figcaption>
                    </figure>
                  ) : null}
                </div>
              ) : (
                <p className="placeholder">
                  Use “Generate Sampling Diagnostics” to render the histogram and PDF comparisons.
                </p>
              )}
            </section>

            <section className="analysis-group">
              <header className="analysis-group-header">
                <h2>Free Energy Estimation</h2>
              </header>
              {analysisResult?.freeEnergyTopPlot ||
                analysisResult?.freeEnergyBottomPlot ||
                analysisResult?.freeEnergyStandardPlot ? (
                <div className="analysis-group-images">
                  {analysisResult?.freeEnergyTopPlot ? (
                    <figure className="analysis-image-card">
                      <img
                        src={`data:image/png;base64,${analysisResult.freeEnergyTopPlot}`}
                        alt="BAR variance diagnostics"
                      />
                      <figcaption className="image-caption">BAR variance diagnostics</figcaption>
                    </figure>
                  ) : null}
                  {analysisResult?.freeEnergyBottomPlot ? (
                    <figure className="analysis-image-card">
                      <img
                        src={`data:image/png;base64,${analysisResult.freeEnergyBottomPlot}`}
                        alt="JAR variance diagnostics"
                      />
                      <figcaption className="image-caption">JAR variance diagnostics</figcaption>
                    </figure>
                  ) : null}
                  {analysisResult?.freeEnergyStandardPlot ? (
                    <figure className="analysis-image-card">
                      <img
                        src={`data:image/png;base64,${analysisResult.freeEnergyStandardPlot}`}
                        alt="Variance diagnostics"
                      />
                      <figcaption className="image-caption">Variance diagnostics</figcaption>
                    </figure>
                  ) : null}
                </div>
              ) : (
                <p className="placeholder">
                  Use “Generate Trajectory Class Estimates” or “Generate Variance Diagnostics” to
                  compute the variance panels.
                </p>
              )}

              {analysisResult?.metadata ? (
                <dl className="analysis-metadata">
                  {Object.entries(analysisResult.metadata).map(([key, value]) => (
                    <Fragment key={key}>
                      <dt>{key}</dt>
                      <dd>{Number(value).toFixed(3)}</dd>
                    </Fragment>
                  ))}
                </dl>
              ) : null}
            </section>
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
