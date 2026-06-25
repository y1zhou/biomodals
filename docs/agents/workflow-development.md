# Biomodals Workflow Development

Detailed workflow-development instructions for `src/biomodals/workflow/` and
shared workflow schemas under `src/biomodals/schema/` live in the repo-local
skill:

- `.agents/skills/biomodals-workflow-development/SKILL.md`
- `.agents/skills/biomodals-workflow-development/references/workflow-development.md`

## How Agents Should Use It

- Invoke or read the `biomodals-workflow-development` skill before creating,
  editing, or reviewing Biomodals workflow code.
- When adding workflow-compatible app functions under `src/biomodals/app/`, also
  follow `docs/agents/app-development.md`.

## Maintenance

- Update the workflow skill when workflow standards change.
- Keep this document as a pointer and coordination note, not a duplicate copy of
  the skill.
