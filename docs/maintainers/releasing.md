# Releasing

`chessnets` and `chessnets-lc0-protobuf` are released together with one
version. Pushing a tag `v<version>` runs
[`.github/workflows/release.yml`](https://github.com/shermansiu/chessnets/blob/main/.github/workflows/release.yml), which:

1. runs the full CI suite;
2. checks the release with `scripts/release_check.py`: versions in sync, tag
   matching the version, no `Private ::` classifier, matching versions in
   `CITATION.cff` and the README citation, and a dated `CHANGELOG.md` section;
3. builds both packages and runs the `twine check` and license checks in
   `scripts/publish_pypi.sh`;
4. uploads the plugin, then the core package, to PyPI through Trusted
   Publishing;
5. creates the GitHub Release, using the changelog section as its notes and
   attaching the wheels and sdists.

If any step fails, nothing after it runs. A version cannot be re-uploaded to
PyPI, so if the core upload fails after the plugin succeeded, fix the cause
and re-run the failed jobs rather than tagging again.

## One-time setup

1. On GitHub, create two environments (*Settings → Environments*): `pypi`
   for `chessnets` and `pypi-lc0-protobuf` for the plugin. They need no
   secrets or variables. For each one, it is worth setting:
   - *Deployment branches and tags* → *Selected branches and tags*, with a
     tag rule `v*`, so only release tags can publish;
   - *Required reviewers* → yourself, so every PyPI upload waits for your
     approval.
2. On PyPI, add a trusted publisher to each project (*Manage → Publishing*),
   with owner `shermansiu`, repository `chessnets` and workflow
   `release.yml`:

   | Project | Environment |
   |---|---|
   | `chessnets` | `pypi` |
   | `chessnets-lc0-protobuf` | `pypi-lc0-protobuf` |

   The environments must differ: PyPI refuses a second publisher with the same
   repository, workflow and environment. For a project that does not exist on
   PyPI yet, add it as a *pending publisher* from your account's *Publishing*
   page; the first release creates the project.

No API token is stored in the repository.

## Making a release

1. Set the new version everywhere:

   ```shell
   scripts/sync_versions.py --bump minor   # or --bump patch, or --set 0.2.0rc1
   ```

2. In `CHANGELOG.md`, rename `## [Unreleased]` to
   `## [<version>] - YYYY-MM-DD`, start a new empty `## [Unreleased]` above
   it, and update the link references at the bottom:

   ```markdown
   [Unreleased]: https://github.com/shermansiu/chessnets/compare/v<version>...HEAD
   [<version>]: https://github.com/shermansiu/chessnets/releases/tag/v<version>
   ```

   Update `version` and `date-released` in `CITATION.cff`, and `version` in
   the README's BibTeX entry; the release check fails if they disagree.

3. Check locally, then optionally rehearse on TestPyPI:

   ```shell
   scripts/release_check.py --tag v<version>
   scripts/publish_pypi.sh                              # build and check only
   scripts/publish_pypi.sh --upload --test-pypi         # optional rehearsal
   ```

4. Commit, tag and push:

   ```shell
   git commit -am "Release <version>"
   git tag -a v<version> -m "Release <version>"
   git push origin main v<version>
   ```

Versions with an `a`, `b`, `rc` or `.dev` suffix are marked as prereleases on
GitHub. PyPI also treats them as prereleases, so `pip install chessnets` does
not pick them up unless asked.
