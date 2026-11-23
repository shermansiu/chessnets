"""Model card generators for first-party ChessNets checkpoint repositories.

``common`` holds the shared repository and preview/write helpers. Each family
module provides ``update(root, *, write, repository_id) -> bool``:

- ``lc0``: Lc0 and Maia-1 cards. Only the generated tail starting at
  ``## Use with ChessNets`` is managed; the hand-written frontmatter and
  source notes above it are preserved.
- ``maia2``: Maia 2 cards, generated in full from ``config.json``.
"""
