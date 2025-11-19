Paper files moved to separate repo

The LaTeX paper source (paper.tex, references.bib, and related files) have been moved to the `LTC_CFC_ContinualLearning` repository. That repo is linked to Overleaf for collaborative editing.

Guidelines

- Workflow:
  1. Before editing paper files locally, always pull the latest changes in the `LTC_CFC_ContinualLearning` repo:
     ```bash
     cd LTC_CFC_ContinualLearning
     git pull origin main
     ```
  2. If you edit on Overleaf, pull again before making local changes to avoid merge conflicts.
  3. Make small, focused commits and push to the `LTC_CFC_ContinualLearning` repo. If you update the bibliography, make sure `references.bib` is kept in sync.

- Where files live:
  - The paper repository is in `LTC_CFC_ContinualLearning/` in this workspace (and also on GitHub: `FNeubuerger/LTC_CFC_ContinualLearning`).

- Overleaf:
  - The Overleaf project is synchronized with the `LTC_CFC_ContinualLearning` GitHub repository. When collaborating via Overleaf, be careful to pull the latest GitHub changes before editing locally.

Contact

If you're unsure about the workflow, ping the repository owner before making large edits.
