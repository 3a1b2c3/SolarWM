# Dataset access

SolarWM-Data is distributed through several repositories because the complete
video and latent payloads are much larger than the release controls. Start with
the main dataset repository, then add the payload required by your example.

For released recipes that use preencoded data, the simplest route is to
download the matching latent generation. Raw-WDS is not required for those
training runs; it is needed only for full raw-video access or workflows that
encode directly from source videos.

## Start here

| Goal | Download |
|---|---|
| Train a released preencoded recipe | Main repository plus its matching latent generation |
| Inspect schemas or test the reader | Main SolarWM-Data repository only |
| Access the full video corpus, train online, or create new latents | Main repository plus `raw-wds/` |
| Run periodic validation or standalone inference | The payload referenced by the selected validation/inference index |

The main release repository is available on
[Hugging Face](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data) and
[ModelScope](https://modelscope.cn/datasets/junchao2003/SolarWM-Data).
It contains the release metadata, licenses, recipe indexes, test-set controls,
the small self-contained `releases-v1/example/` catalog, and the optional
`SolarWM-Data-Annotation/` reconstruction package. It does **not** contain the
full `releases-v1/raw-wds/` or `releases-v1/latent-wds/` payloads.

```bash
python -m pip install --upgrade huggingface_hub
export SOLAR_DATA_HOME=/path/to/SolarWM-Data

hf download junchaoh-cs/SolarWM-Data \
  --repo-type dataset \
  --exclude "SolarWM-Data-Annotation/**" \
  --local-dir "$SOLAR_DATA_HOME"

export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"
```

The example archives are enough for schema inspection and reader/preencoding
smokes. They deliberately contain only one record per format and are not a
training corpus.

## Getting preencoded latents

Each latent generation is published in a separate dataset repository. Links
are maintained in the [latent-WDS release list](latent-wds.md), including the
Wan2.2-5B 153f generation used by the quickstart.

Preserve the downloaded generation at
`$SOLAR_DATA_ROOT/latent-wds/<generation>/`. The main SolarWM-Data repository
contains the matching recipe indexes; no raw-WDS download is needed for a
latent-only training run.

## Getting raw-WDS

Raw-WDS provides the full processed video corpus. Download or reconstruct it
only when you need the raw videos, online encoding, your own latent generation,
or an example whose selected index points to `raw-wds/`. Choose either
reconstruction from the annotation release or assisted access.

### Reconstruct from annotations

The annotation package is the
[`SolarWM-Data-Annotation/`](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data/tree/main/SolarWM-Data-Annotation)
directory in the same SolarWM-Data repository. Download it without fetching
the rest of the data repository:

```bash
hf download junchaoh-cs/SolarWM-Data \
  --repo-type dataset \
  --include "SolarWM-Data-Annotation/**" \
  --local-dir "$SOLAR_DATA_HOME"

export ANNOTATION_REPO="$SOLAR_DATA_HOME/SolarWM-Data-Annotation"
```

This is an annotation-only release and contains no videos. It provides the
released camera trajectories, captions, metadata, public source identities,
and reconstruction tools for the complete corpus. Follow
`$ANNOTATION_REPO/README.md` to obtain the original videos from their source
publishers under the applicable terms and process them into the `raw-wds/`
layout expected by SolarWM-Data.

Use the recipe indexes emitted by the reconstruction tools. Do not pair a
newly packed tar with an index for a separately distributed shard: local reads
check the declared byte size as well as the relative path.

The companion package includes download, corpus-restoration, and raw-WebDataset
build tools.

### Request access assistance

If reconstructing the source media is impractical, submit the
[Dataset Access Form](https://docs.google.com/forms/d/e/1FAIpQLSfS-SLOiSRVDWwZ2kPl9ywN27aB6QplN0jpdKaBu-gG8aNsvQ/viewform).

The form requests the reason for assistance, institution, organizational
email, requested subset, and confirmation that the applicant accepts the
applicable dataset terms. Approved applicants will receive download
instructions by email. Submission does not replace any upstream license or
usage restriction.

## Portable local layout

After combining the controls and the payloads you need, the relevant portion
of the tree looks like this:

```text
SolarWM-Data/
|-- SolarWM-Data-Annotation/  # optional raw-WDS reconstruction package
`-- releases-v1/
    |-- release.json
    |-- recipes/
    |-- test-set/
    |-- example/
    |-- licenses/
    |-- raw-wds/               # obtained through one raw-data route, when needed
    `-- latent-wds/            # one or more separately downloaded generations
```

Index rows use paths relative to `releases-v1/`; the same assembled tree can be
mounted at any local path. See the [data contract](data-contract.md) for the
runtime storage and integrity rules.
