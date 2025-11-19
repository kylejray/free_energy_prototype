import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

interface LandingPageProps {
  onEnterApp: () => void;
}

const LANDING_CONTENT_MARKDOWN = `
## Webpage Overview
This project is a playground for messing around with different sampling methods to estimate 
free energy difference from work measurements. The main focus is on understanding how very biased subclasses of trajectories 
can be used to recover free energy estimates that are remarkably unbiased.

This idea stems from something called the Trajectory Class Fluctuation Theorem (TCFT). [As early as 2005](doi.org/10.1103/PhysRevE.73.046105), Jarzynski proposed
the probabilities of specific subclasses of trajectories in the forward and reverse process were related to their usefulness for estimating free energy differences from exponential averages.
More recently, this idea has been formalized into the [TCFT](https://link.springer.com/article/10.1007/s10955-025-03422-z), which states that any measureable subclass of
trajectories $C$ satisfies:

$$
\\langle e^{-\\Sigma} \\rangle_C = \\frac{R(C)}{P(C)}
$$
Where the average is a conditional average among the trajecotries in $C$, $\\Sigma$ is the entropy production associted with the trajecotry,
$P(C)$ is the probability of observing  a trajectory in class $C$ during the forward process, and $R(C)$ is the
probability of observing the a trajectory in  the "reversed class" $C^\\dagger$ during the reverse process.
Under some typical asssumptions, trajecotries in all classes will have an entropy production that can be written as $\\Sigma = \\beta (W - \\Delta F)$, leading
to a somewhat familiar form of the equation:

$$
\\ln \\langle e^{-\\beta W} \\rangle_C = -\\beta \\Delta F + \\ln \\frac{R(C)}{P(C)}
$$
And, it is from this equation that we find a remarkable insight into free energy estimation.
The equation above is true for *any* class of trajectories $C$. This means that, for example-- we could use the work value of only a single trajecotry
to come up with an estimate of the free energy difference, provided we correctly conpensate via the term $\\ln \\frac{R(C)}{P(C)}$.
A more practical solution would be to find classes of trajectories that are particularly well-suited
for estimating free energy differences-- for example, those that have a high probability of observing
trajectories with work values near the free energy difference, or even just a batch that has a particularly small variance in work values.
Of course, with the caveat that we need to be able to estimate the term $\\ln \\frac{R(C)}{P(C)}$ accurately for the chosen class.

Even more promising, is the fact that any asymptotically unbiased estimator of the free energy (say, for example the Bennett Acceptance Ratio method) can be plugged into the TCFT framework because
it will agree with the log of the exponential average in the infinite sample limit. In this light, we can write the above into a more general form.
 $$
 \\beta \\Delta F = \\widehat{\\Delta F}(C) -  \\ln \\frac{R(C)}{P(C)}
$$
Here the $\\widehat{\\Delta F}(C)$ is any unbiased estimator of the free energy difference computed from observations in class $C$, 
and the term $\\ln \\frac{R(C)}{P(C)}$ is a correction factor that accounts for the bias introduced by selecting trajectories in class $C$.

This approach opens up a wide range of possibilities for designing sampling protocols and analyzing statistical simulations, in which the two estimation 
problems need to be balanced. If, for example, we can find a particularly good way of estimating one of the two terms, we can focus on optimizing the other.

Here, I don't provide a sampling strategy, but rather a simple interactive tool to explore how different classes of trajectories can be used to estimate free energy differences.

-Kyle Ray

## Methodology
In the tool provided here, the following steps are taken to explore free energy estimation using trajectory classes:

1. **Define a Work Distribution and trajectory Class**: 
Start by specifying a piecewise continuous work distribution using a set of control points. 
These points define the shape of the distribution from which work values will be sampled.
Specify lower and upper bounds for the work values to be included in the trajectory class of interest.

2. **Visualize the Distribution**: 
The defined work distribution is visualized using a distribution editor, allowing for interactive adjustment of control points and bounds.
When this button is pressed, the distribution of works in the reverse process and the free energy change are computed, 
as well as the work values associated with the reverse class.  Also, this is an opportunity to verify that the
built in sampler for the distribution and the reverse are working as expected. 

3. **Compare Free Energy Estimators**: 
Using the work values sampled from the defined distribution, compute free energy estimates using both the Jarzynski equality and the Bennett Acceptance Ratio (BAR) method.
These estimates are compared to understand how well each method performs given the sampled data. Free energy estimates are generated using the free energy functions from Gavin Crooks'
[thermoflow](https://github.com/gecrooks/thermoflow) library, ported from jax to numpy (surprisngly, for performance reasons on the "free tier" hardware this is running on).

4. **Variance Diagnostics**:
Analyze previous standard ways of estimating the variance in the BAR method for free energy estimation. These are somewhat unrelated to the TCFT,
but provide useful context for understanding the reliability of free energy estimates when work data from the full distribution is accesible.

## Particualarly Relevant Literature

*   [Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences](https://doi.org/10.1103/PhysRevE.60.2721)
*   [Comparison of efficiency and bias of free energies computed by exponential averaging, the Bennett acceptance ratio, and thermodynamic integration](https://doi.org/10.1063/1.1873592)
*   [Statistically optimal analysis of samples from multiple equilibrium states](https://doi.org/10.1063/1.2978177)
*   [Measuring Thermodynamic Length](https://doi.org/10.1103/PhysRevLett.99.100602)
`;

const LandingPage: React.FC<LandingPageProps> = ({ onEnterApp }) => {
  return (
    <div style={{
      maxWidth: '800px',
      margin: '0 auto',
      padding: '4rem 2rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '2rem',
      alignItems: 'flex-start'
    }}>
      <div style={{ marginBottom: '1rem' }}>
        <h1 style={{ fontSize: '3rem', lineHeight: '1.1', marginBottom: '1rem' }}>
          Free Energy Estimation Prototype
        </h1>
        <p style={{ fontSize: '1.25rem', color: '#94a3b8', maxWidth: '60ch' }}>
          An interactive tool for exploring free energy estimation methods and sampling distributions for a simple toy model.
        </p>
      </div>

      <button 
        onClick={onEnterApp} 
        className="toggle-btn"
        style={{
          background: '#3b82f6',
          color: 'white',
          fontSize: '1.1rem',
          padding: '0.75rem 2rem',
          marginTop: '0rem',
          marginBottom: '1rem'
        }}
      >
        Start Analysis
      </button>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        background: 'rgba(30, 41, 59, 0.3)',
        padding: '2rem',
        borderRadius: '1rem',
        width: '100%',
        border: '1px solid rgba(148, 163, 184, 0.1)'
      }}>
        <div className="markdown-content" style={{ color: '#cbd5e1', lineHeight: '1.7' }}>
          <ReactMarkdown 
            remarkPlugins={[remarkMath]} 
            rehypePlugins={[rehypeKatex]}
          >
            {LANDING_CONTENT_MARKDOWN}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
