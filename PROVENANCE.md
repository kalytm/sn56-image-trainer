# Provenance

This training repository builds on the published SN56 image tournament winner:

- Source: https://github.com/gradients-opensource/god-image-tourn-f16ed79079376136-20261102-position-1
- Commit: 44cc7f5454519016aaf0de15e5fa12d74c6b2fdf

LICENSE.md and NOTICE are kept unchanged. CHANGES-vs-champion.md lists every change from the source commit.

New image families (image-v2, after G.O.D #1420): `scripts/toolkit_entry.py` and `templates/*.yaml` are taken from the
Gradients-on-Demand repository (Apache-2.0, same LICENSE.md and NOTICE as above):

- Source: https://github.com/gradients-ai/G.O.D (`ops/docker/scripts/image_toolkit_entrypoint.py`, `core/training_templates/`)
- Commit: 14c3208 (#1420, "Add next image families and native-loss evaluation"); the only change is the template directory.
- The base image and the AI Toolkit / Musubi pins in `ops/docker/standalone-image-toolkit-trainer.dockerfile` are the
  same as in that commit's reference toolkit trainer, with the base pinned by digest.
