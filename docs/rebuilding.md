# Rebuilding or refitting the bundle

Related: [README](../README.md) · [Methods](methods.md) · [Commands](commands.md) · [Scenarios](scenarios.md) · [Working with the data](working-with-data.md)

The shipped bundle is ready to use and most users never need this page. It
matters if you hold authorized access to the VA CAN extract and want to rebuild
the bundle against your own extract, refit for a different cohort year, or change
the model specification.

## What you need

Authorized access to the source extract, specifically
`all_can_2014_v2_ordinalver.parquet`, plus whatever approvals your data use
agreement requires. The extract is never distributed with this repository.

## Point the fitter at your extract

```bash
export VA_DIR=/path/to/can/extract
uv run --extra fit python -c "from tosw_synth.fit import rebuild; print(rebuild())"
```

The rebuild:

1. reads `all_can_2014_v2_ordinalver.parquet` from `VA_DIR`
2. drops `patienticn`, `yr`, `Death1y`, and `intercept`
3. draws the 50 percent fitting sample with seed `343`
4. fits 32 covariate classifiers, the age summary, and the two outcome
   coefficient sets
5. rewrites `bundles/can2014/` and records the fit in `FIT_INFO.txt`

Expect about 10 minutes and roughly 4 GB of memory.

## Refitting for a different cohort or specification

The fit lives in `src/tosw_synth/fit.py`.

| Goal | Change |
| --- | --- |
| A different cohort year | `SOURCE_FILENAME` |
| A different model | `build_spec` |
| A different covariate order | the order produced by `covariate_columns` |

After any refit, regenerate your data. The bundle now carries different fitted
parameters, so previously generated files no longer correspond to it. Bump the
bundle version and regenerate rather than mixing old tables with a new bundle.

## Keeping a rebuild private

A rebuilt bundle still contains fitted parameters rather than patient rows, but
those parameters come from your extract. Publish it only if your data use
agreement permits it.

Run the leak check before sharing:

```bash
grep -rlE "/Volumes/|/Users/|patienticn" bundles/can2014/ || echo "no private paths found"
```

The committed bundle passes this check. Verify the same for any rebuild, and
review `FIT_INFO.txt` to confirm it records no absolute path.
