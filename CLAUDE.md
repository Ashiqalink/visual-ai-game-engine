# CLAUDE.md

## Hard rules

- **Never use the WebSearch tool.** Hand the user the exact query to run instead.
- **Never call an external API.** No WebFetch, no `curl` / `Invoke-WebRequest` to a
  remote host, no provider SDK call. Work from local files, the repos, and `brain`;
  hand the user the exact command to run instead.
