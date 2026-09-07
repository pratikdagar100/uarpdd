# Product

## Register

product

## Users

**District disaster-management officers and IMD duty forecasters.** They open
this early in the morning during monsoon season, in a fluorescent-lit control
room, on a desk monitor. The job: decide, for each station they are
responsible for, whether today warrants activating a response — moving
equipment, staffing shelters, issuing a public advisory. They are reading to
make a decision under time pressure, not browsing.

**Evaluators (SIH panel, MoES reviewers).** Same screens, different context: a
bright hall, a projector, five to ten minutes. They are judging whether the
system is scientifically sound and whether it looks like something that could
actually be deployed.

The two audiences do not want different products. An evaluator is convinced by
exactly what an officer needs — a forecast that states its own uncertainty
honestly and a screen that can be read at a glance.

## Product Purpose

Predict whether significant rainfall (≥ 2.5 mm) will fall tomorrow at Indian
weather stations, quantify how much the model trusts each prediction
(calibrated probability + split conformal prediction set), and translate both
into an IMD-standard colour-coded warning and an inundation risk band.

A fourth job runs alongside: a controlled experiment testing whether showing
uncertainty actually improves human decisions. The product is therefore its own
evidence — it must communicate uncertainty well enough that the study measures
the idea rather than a bad interface.

Success: an officer reads a station's warning and knows, without clicking,
what is predicted, how sure the model is, and what to do. Failure: the
interface projects more confidence than the model has.

## Brand Personality

**Instrumented, candid, unhurried.**

Voice is that of a duty forecaster writing a log entry: precise, plain, never
dramatic. It states what is known, what is not, and what follows. It never
sells the model. "The model cannot rule out either outcome" is the register —
an admission stated as calmly as a fact, because operationally it *is* one.

Numbers carry the emotion; the prose stays flat. No exclamation marks, no
"AI-powered", no urgency the data does not justify.

## Anti-references

- **The generic blue SaaS dashboard.** A blue accent on every chrome element,
  cards in a uniform grid, colour used decoratively. This is where the product
  started and it is precisely wrong for a warning system: it spends colour on
  furniture, leaving none for severity.
- **Consumer weather apps.** Illustrated clouds, gradients, big friendly
  temperature. This is not a forecast to enjoy; it is a decision aid.
- **Crisis-theatre dashboards.** Dark screens with pulsing red, map glow,
  "ALERT" in wide tracking. Ambient urgency is a lie when the data is green,
  and it destroys the contrast the real red needs.
- **Terminal-dark ops console.** The reflexive answer to "not blue SaaS", and
  wrong here for a physical reason: these screens are read under fluorescent
  light and thrown on projectors in bright halls.

## Design Principles

1. **Colour is data.** The IMD scale (Green / Yellow / Orange / Red) is a
   regulated vocabulary, and every decorative use of colour elsewhere steals
   signal from it. Chrome is neutral. Saturation on screen means severity, and
   nothing else.
2. **Never encode meaning in colour alone.** A red–green scale fails the most
   common form of colour blindness outright. Every warning level carries a
   text label, a distinct glyph, and a severity rank in addition to its colour.
3. **The forecast must survive being wrong.** Uncertainty is not a footnote
   appended to a prediction; it is part of the prediction. An ambiguous
   forecast should look visibly different from a confident one at a glance,
   not on inspection.
4. **Legible at a glance, complete on inspection.** The decision (level,
   probability, advice) reads from across a desk. The evidence (conformal set,
   feature influence, climatology) is one level down for the person who needs
   to defend the call.
5. **Operational calm.** Density without noise, motion only where it reports a
   state change. The interface should be boring to use for the hundredth time.

## Accessibility & Inclusion

- **WCAG 2.1 AA.** Body text ≥ 4.5:1, large text ≥ 3:1, verified rather than
  assumed. The previous muted grey (`#8494a7`) failed this at every size it
  was used, which is why readability is a correctness issue here and not a
  preference.
- **Colour-blind safe.** Deuteranopia and protanopia make the Green/Red
  extremes of the IMD scale indistinguishable. Redundant encoding (label +
  glyph + rank) is mandatory on every warning surface, per principle 2.
- **Reduced motion.** Every transition has a `prefers-reduced-motion`
  fallback.
- **Keyboard and screen reader.** Visible focus rings on every control;
  semantic structure and text alternatives wherever Streamlit permits them.
  Charts always accompanied by the numbers in text form.
