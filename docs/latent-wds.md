# Latent-WDS releases

Each preencoded latent generation is distributed in a separate dataset
repository. After downloading a generation, preserve its directory name and
place it at:

```text
SolarWM-Data/releases-v1/latent-wds/<generation>/
```

For a released preencoded training recipe, the main SolarWM-Data repository
and its matching latent generation are sufficient. You do not need raw-WDS
unless you want the full raw-video corpus or a workflow that performs encoding
from source videos.

The following list mirrors the latent generation directories defined by the
SolarWM-Data release. Repository links will be added as uploads complete.

| Generation | Repository |
|---|---|
| `wan22-ti2v5b-81f-480p-v1` | Coming soon |
| `wan22-ti2v5b-81f-720p-v1` | Coming soon |
| `wan22-ti2v5b-153f-480p-v1` | [ModelScope](https://modelscope.ai/datasets/Junchao-cs/SolarWM-Data_Latent-WDS_wan22-ti2v5b-153f-480p-v1) |
| `wan22-ti2v5b-153f-720p-v1` | Coming soon |
| `wan22-ti2v5b-957f-480p-v1` | Coming soon |
| `wan22-ti2v5b-957f-720p-v1` | Coming soon |
| `wan22-i2v-a14b-81f-480p-v1` | Coming soon |
| `wan22-i2v-a14b-81f-720p-v1` | Coming soon |
| `wan22-i2v-a14b-153f-480p-v1` | Coming soon |
| `wan22-i2v-a14b-153f-720p-v1` | Coming soon |
| `wan22-i2v-a14b-957f-480p-v1` | Coming soon |
| `wan22-i2v-a14b-957f-720p-v1` | Coming soon |
| `ltx-153f-h512-w768` | Coming soon |
| `ltx-953f-h512-w768` | Coming soon |
| `minimax-h3-158f-768p-nomind-v1` | Coming soon |

Download only the generation required by the selected training example. The
main SolarWM-Data repository contains the matching recipe indexes, but it does
not contain these latent payloads.
