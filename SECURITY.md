# Security and data handling

- Never commit API keys, `.env` files, model credentials, or private endpoints.
- Keep licensed Pistachio data, unpublished experimental records, manuscripts,
  and model checkpoints outside the repository.
- Before publishing a branch, review `git diff --cached` and run a secret
  scanner if one is available.
- Report accidental disclosure privately to the repository owner; rotate any
  exposed credential before removing it from Git history.
