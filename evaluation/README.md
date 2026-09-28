# RemoteSAM competition data preparation

This directory contains data-preparation and frozen evaluation-protocol files.
Keep downloaded archives, original images, masks, tokens, and machine-local
paths out of Git. The captain delegated the evaluation-set decision to teammate
B; the final 20 image IDs and relative paths are now fixed in
[`dataset_manifest.csv`](dataset_manifest.csv). The measurement rules are in
[`REMOTESAM_EVALUATION_PROTOCOL.md`](REMOTESAM_EVALUATION_PROTOCOL.md).

## Buildings: WHU Building Dataset

Source: Wuhan University GPCV's [WHU Building Dataset page](https://gpcv.whu.edu.cn/data/building_dataset.html),
**Satellite Dataset I (global cities)**, an archive of 204 paired 512 × 512
image/label TIFFs. The selected archive has SHA-256
`7208f419d3f900e2b00fe76f47e8635ad5a3265ecbc68a0f48620df43fed3681`.

Run `prepare_whu_samples.py --archive ARCHIVE.zip --output-dir OUTSIDE_GIT`
to reproduce ten candidates. The fixed default seed is
`geoscope-competition-20260927`. The script CRC-checks the archive, confirms
one-to-one filenames, equal image/label sizes, black/white-only building labels,
then selects positive images by a fixed SHA-256 ordering while preferring
distinct filename prefixes. It converts each original TIFF and its raster label
to lossless PNG, without resizing. The output includes SHA-256 hashes and a CSV
manifest so every chosen pair can be checked again.

Frozen source filenames selected from the official archive:

`18_0.tif`, `25_1.tif`, `6_2.tif`, `30_2.tif`, `42_1.tif`,
`41_1.tif`, `9_1.tif`, `51_2.tif`, `4_3.tif`, `5_3.tif`.

These ten cases are a fixed diagnostic subset, **not** a claim about accuracy on
the full WHU dataset. Model outputs were not inspected to choose them.

## Airplanes: iSAID

The [iSAID dataset page](https://captain-whu.github.io/iSAID/dataset.html)
provides pixel annotations, while its authors' [DOTA v1.0 page](https://captain-whu.github.io/DOTA/dataset.html)
provides the corresponding original imagery. Use matching **validation** splits;
the iSAID test masks are withheld. Do not use DOTA's oriented bounding boxes as
pixel ground truth. The downloaded iSAID validation `Semantic_masks/images.zip`
has SHA-256
`F3B333B4B0469EDB1079C49C163C915AE8DD39D116191798633564BAC1998D69`.
The official [development-kit label map](https://github.com/CAPTAIN-WHU/iSAID_Devkit/blob/master/preprocess/cityscapesScripts/cityscapesscripts/helpers/labels.py)
defines `plane` as RGB `(0, 127, 255)`. Convert only exactly matching pixels to
white and all other classes/background to black. This makes a **binary plane
class mask**, not instance-separated ground truth or an instance-AP benchmark.

Run `prepare_isaid_samples.py --masks-archive MASKS.zip --dota-archive DOTA.zip
--output-dir OUTSIDE_GIT` if the full DOTA validation archive is local. To avoid
waiting for the 3.34 GB archive, the script can instead read only required ZIP
byte ranges from the official Google Drive file using
`--dota-drive-file-id 1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP`. This requires
network access and the public sharing link to remain available. Each extracted
PNG is CRC-checked by Python's ZIP reader, matched by `Pxxxx` ID, checked for
equal image/mask dimensions, and recorded with source/output hashes. The fixed
default seed is `geoscope-competition-20260927`, and the minimum is 1,000 plane
pixels. Among 458 validation masks, 72 contained at least one plane and 67
passed the 1,000-pixel threshold; the deterministic ordering yielded these 10
paired cases:

`P1492`, `P1738`, `P1143`, `P1184`, `P1483`, `P0179`, `P0257`,
`P1529`, `P1122`, `P0259`.

The separately downloaded official DOTA validation `images/part1.zip` has
SHA-256
`E8F5602DC22C5142B3D2F26982C54DF4E9E5E9B6B92015BC20A434E12B45CA9D`.
It contains 458 PNG images; its complete ZIP CRC check passed. All 10 selected
source-image hashes match the copies previously obtained by range requests.

These include large original images (up to 6912 × 3072 or 5838 × 5338). Keep
the original resolutions for the source-of-truth set; if the service requires
tiling or downscaling, treat that as a separately versioned experiment before
comparing predictions with masks. The ten cases are a fixed diagnostic subset, not a
claim about the complete iSAID validation split. Model outputs were not used to
choose them.

iSAID and DOTA restrict images and annotations to academic use and prohibit
commercial use. Check redistribution terms before placing any raw examples in
the Git repository or public competition materials.

## Evaluation handoff

Use the frozen IDs in the shared manifest for Agent tasks, RemoteSAM metrics,
visualizations, and presentation figures. Run the model on the complete fixed
set, retain failures, and publish per-image records before reporting averages.
Before each run, `verify_samples.py WHU_OUTPUT_DIR ISAID_OUTPUT_DIR` checks all
manifest hashes, dimensions, binary masks, and positive foregrounds. Add
`--dota-archive DOTA.zip` to compare the selected source-image hashes with the
complete local DOTA validation archive.

The portable team manifest can be regenerated with `build_dataset_manifest.py`
from the two local `selected_manifest.csv` files. Its current SHA-256 is
`5b5339814a2c83c4b87ec026fa46fe8ff67cf6b534554c24ac8d21e421456241`.
Before inference, run `verify_dataset_manifest.py --manifest
evaluation/dataset_manifest.csv --data-root DATASETS_ROOT --expected-sha256
5b5339814a2c83c4b87ec026fa46fe8ff67cf6b534554c24ac8d21e421456241`.
