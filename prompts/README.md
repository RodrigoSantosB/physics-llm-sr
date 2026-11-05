# Prompts (v1.1-reconstructed)

Versioned plain-text templates for the LLM evaluator (no code). Version: `v1.1-reconstructed`.

## Origin
- The base prompt is NOT published in the paper (Taskin et al., Sci Rep 2026, 16:1614, p.5, blank after "we used the following prompt") and the supplement (MOESM1) contains only runtime/performance tables, no prompt text. Everything here is `reconstructed: true` (GAP-13).
- `base/system.txt`: structure from p.5 (`### ROLE`, `### METRICS` with three criteria on numeric scales, three anchor examples: kinetic energy, a dimensional mismatch, an unnecessarily complex form, strict "ONLY" output). Scales, example scores and wording are invented.
- Deviation: the paper uses a Python-style list `[dim corr, simp, sim, feedback]`; here the output is a JSON object (PRD 6.3 contract): `{"dimensional_consistency", "simplicity", "physical_realism", "feedback"}`. Parsing must validate keys and the [0,1] range and never execute the response.
- `base/evaluation.txt`: per-call user message.
- `variants/A..H.yaml`: definitions from p.7. Placement of the GT ("Formula at end") is unspecified; here the context block is inserted after the expression.

## Changelog
- v1.1-reconstructed: the examples no longer include `sqrt(2 * g * h)` (the free-fall ground truth, which leaked the answer into every variant and was copied by Mistral). Examples are now kinetic energy `m * v ** 2 / 2`, `m + v` (dimensional mismatch) and a convoluted form; none is the GT or structurally equivalent to the GT of any scenario (`tests/unit/test_prompt_leakage.py` enforces this). The prompt hash changes (it is content-based), so cached v1.0 evaluations are not reused.
- v1.0-reconstructed: initial version.

## Variants
| id | includes | gt_disclosed |
|---|---|---|
| A | - | false |
| B | variables | false |
| C | experiment | false |
| D | ground_truth | true |
| E | variables, experiment | false |
| F | variables, ground_truth | true |
| G | experiment, ground_truth | true |
| H | variables, experiment, ground_truth | true |

## Scientific control rule
D, F, G, H disclose the ground-truth equation to the LLM (`gt_disclosed: true`) and must be analyzed separately as ground-truth-disclosure experiments. E is the priority configuration: equation recovery without handing the answer to the model.

## Placeholder contract (syntax `{{name}}`, exact names)
- `{{expression}}` (evaluation.txt): candidate expression, infix text in the AST syntax.
- `{{context_block}}` (evaluation.txt): concatenation, in order variables, experiment, ground_truth, of the `sections` blocks of the chosen variant (only those in `includes`); empty string for A.
- `{{variable_descriptions}}`, `{{experiment_description}}`, `{{ground_truth}}` (inside variant sections): from `PhysicalContext.variable_descriptions` (formatted, one per line), `.experiment_description`, `.ground_truth`.
`system.txt` has no placeholders. The renderer must fail on unresolved `{{...}}` and must substitute only listed names. A section appears in `sections` iff its name is in `includes`; `gt_disclosed` is true iff `ground_truth` is in `includes`.
