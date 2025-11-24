import { ChangeEvent, Fragment, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { fetchHealth, runNotebookAnalysis, NotebookRequestPayload } from './client';
import DistributionEditor from './components/DistributionEditor';
import LandingPage from './components/LandingPage';

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

const SAMPLING_DESCRIPTION = `
### Sampling Diagnostics

Here we visualize the forward and reverse work distributions. The forward distribution $P_F(W)$ is defined by the control points above. The reverse distribution $P_R(W)$ is derived using the Crooks Fluctuation Theorem:

$$
\\frac{P_F(W)}{P_R(-W)} = e^{\\beta(W - \\Delta F)}
$$

The trajectory class $C$ is defined by work values that are in the bounds (LL, UL) specified above. 
Generally, the reverse class $C^\\dagger$ is defined as all trajectories that are time reversals of those in $C$.
In this simple case, that reduces simply to those in the negated bounds (-UL, -LL).
The plots show:
1.  **Histograms**: Samples drawn from both distributions using rejection sampling with a piecewise linear envelope.
2.  **Trajectory Classes**: Work values in the forward class defined by the bounds (LL, UL) is highlighted in blue
those in the reverse class are highlighted in orange. Overlaps are given by a grey shade.
3.  **PDF Overlays**: The theoretical probability density functions.
4.  **Log Scale**: Useful for inspecting the tails where rare events occur.
`;

const FREE_ENERGY_DESCRIPTION = `
### Free Energy Estimation

This section compares different estimators for the free energy difference $\\Delta F$. We compare the standard Jarzynski and BAR estimators against their "Trajectory Class" counterparts.

The Trajectory Class estimators use only a subset of trajectories $C$ (defined by the bounds above) and apply the correction:

$$
\\beta \\Delta F = \\widehat{\\Delta F}(C) - \\ln \\frac{R(C)}{P(C)}
$$

*   **Top Row**: BAR-based estimators. Variance estimates use the MBAR method
*   **Bottom Row**: Jarzynski-based estimators. Variance estimators naively take the sample's variance / sqrt(N)
 and propagate it through the logarithm. This is generically not a principled way to estimate the variance of an exponential estimator.
 We use this heuristic to see if/when it can be helpful for certain trajectory classes.
*   **Left**: Uncorrected estimator on class $C$. Note that these methods only have access to work values within the class.
* The Jarzynski method is only given the subset of samples that are within the class in the forward process.
* The BAR method is given also the subset of samples that are within $C^{\\dagger}$ in the reverse process.
*   **Middle**: Corrected estimator (TCFT). 
The probabilities are estimated directly from a binary coarse graining on the sampled data based on if the values fall within the class or not. This introduces the additional variance in according to standard methods of estimating sample proportions. 
This is the worse case scenario, since analytics or others methods might often be able to calculate the exact probabilities.
*   **Right**: Full estimator using all data.
`;

const STANDARD_DESCRIPTION = `
### Variance Diagnostics

Standard variance estimates for the Bennett Acceptance Ratio (BAR) method. These plots help diagnose the reliability of the BAR estimator itself, independent of the trajectory class modifications.

We compare different methods for estimating the asymptotic variance of the BAR estimator, including the standard method and the MBAR method.
`;

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
const DEFAULT_SAMPLING_SAMPLE_SIZE = 5000;

const ANALYSIS_SAMPLE_SIZE_RANGE: NumericRange = {
  min: 5,
  max: 1_000,
  step: 5
};

const SAMPLING_SAMPLE_SIZE_RANGE: NumericRange = {
  min: 100,
  max: 10_000,
  step: 100
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
  const [showLanding, setShowLanding] = useState(true);
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
  const [samplingSampleSize, setSamplingSampleSize] = useState<number>(
    DEFAULT_SAMPLING_SAMPLE_SIZE
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

  const handleSamplingSampleSizeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const next = clampWithStep(Number(event.target.value), SAMPLING_SAMPLE_SIZE_RANGE);
    setSamplingSampleSize(next);
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

    const currentSampleSize = section === 'sampling' ? samplingSampleSize : analysisSampleSize;
    const currentRange = section === 'sampling' ? SAMPLING_SAMPLE_SIZE_RANGE : ANALYSIS_SAMPLE_SIZE_RANGE;
    const sanitizedCurrentSampleSize = clampWithStep(currentSampleSize, currentRange);

    if (section === 'sampling' && sanitizedCurrentSampleSize !== samplingSampleSize) {
      setSamplingSampleSize(sanitizedCurrentSampleSize);
    } else if (section !== 'sampling' && sanitizedCurrentSampleSize !== analysisSampleSize) {
      setAnalysisSampleSize(sanitizedCurrentSampleSize);
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
        sampleSize: sanitizedCurrentSampleSize,
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

  if (showLanding) {
    return <LandingPage onEnterApp={() => setShowLanding(false)} />;
  }

  return (
    <div className="container">
      <div className="panel">
        <div className="panel-header">
          <h1>Free Energy Prototype Suite</h1>
        </div>

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
                <label className="analysis-parameter" htmlFor="sampling-sample-size">
                  <div className="parameter-header">
                    <span>Histogram Samples</span>
                    <span>{samplingSampleSize.toLocaleString()}</span>
                  </div>
                  <input
                    id="sampling-sample-size"
                    type="range"
                    min={SAMPLING_SAMPLE_SIZE_RANGE.min}
                    max={SAMPLING_SAMPLE_SIZE_RANGE.max}
                    step={SAMPLING_SAMPLE_SIZE_RANGE.step}
                    value={samplingSampleSize}
                    onChange={handleSamplingSampleSizeChange}
                    disabled={analysisLoading}
                  />
                </label>
                <label className="analysis-parameter" htmlFor="analysis-sample-size">
                  <div className="parameter-header">
                    <span>Estimator Samples</span>
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
              <div className="analysis-group-content">
                <div className="analysis-description markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                    {SAMPLING_DESCRIPTION}
                  </ReactMarkdown>
                </div>
                <div className="analysis-visuals">
                  {analysisResult?.metadata && (analysisResult.metadata.p_c !== undefined) ? (
                    <p style={{ marginTop: '0rem', marginBottom: '1rem', color: '#94a3b8', fontSize: '0.95rem', lineHeight: '1.6' }}>
                      The forward class represents <strong style={{ color: '#e2e8f0' }}>{(Number(analysisResult.metadata.p_c) * 100).toFixed(2)}%</strong> of 
                      trajectories in the forward process, and the reverse class represents <strong style={{ color: '#e2e8f0' }}>{(Number(analysisResult.metadata.r_c_rev) * 100).toFixed(2)}%</strong> of 
                      the trajectories in the reverse process.
                    </p>
                  ) : null}
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
                </div>
              </div>
            </section>

            <section className="analysis-group">
              <div className="analysis-group-content">
                <div className="analysis-description markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                    {FREE_ENERGY_DESCRIPTION}
                  </ReactMarkdown>
                </div>
                <div className="analysis-visuals">
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
                      {Object.entries(analysisResult.metadata)
                        .filter(([key]) => !['p_c', 'r_c_rev'].includes(key))
                        .map(([key, value]) => (
                        <Fragment key={key}>
                          <dt>{key}</dt>
                          <dd>{Number(value).toFixed(3)}</dd>
                        </Fragment>
                      ))}
                    </dl>
                  ) : null}
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
