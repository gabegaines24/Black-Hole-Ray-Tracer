# GitHub contributions graph (heatmap)

Commits only affect your profile activity graph **after** they appear on GitHub.

## Checklist

1. **`git push`** — Local commits never count until they reach GitHub (`git push origin main`).
2. **Author email on the commit** must match one of these (see GitHub Settings → Email):
   - A **verified** address on your account, or
   - GitHub’s `noreply` address (recommended for privacy).
3. **`git config user.email`** on this machine should match that verified / noreply value for new commits (`git commit` inherits it).
4. **Repository visibility** — If the repo is private, enable “Include private contributions on my profile” under GitHub profile contribution settings.

If a commit locally shows correct author (`git log -1 --format=%ae`) but the graph stays empty after a refresh, usual causes are: not pushed yet, push went to another remote/branch without a merged PR to default branch, or the email used is not verified on GitHub.
