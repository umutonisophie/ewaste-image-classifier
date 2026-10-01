# E-Waste Image Classifier

A MobileNetV3-Large image classifier that identifies 29 categories of electronic
device, built for an e-waste recycling dataset scraped from Nyabugogo market,
Kigali.

**The trained model works: 97.2% top-1 accuracy (69/71) on the master dataset.**
It did not load at all when this project was picked up — see below.

## Usage

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python repair_checkpoint.py                    # once, see below
python predict.py path/to/device.jpg           # single image
python predict.py --dir MASTER_e_waste_visual_dataset   # evaluate a folder
```

The CPU index URL matters. A default `pip install torch` pulls CUDA wheels that
are several gigabytes and will not work on a machine without a GPU.

## What was broken

**The checkpoint would not load.** `ewaste_mobilenet_master_model.pth` is a
PyTorch zip archive that also contained a loose `split_dataset.py` at the
archive root. torch validates archive layout and rejects it, so the single most
important file in the project was unusable — and it fails with a confusing error
about a file *in an archive* not being in a subdirectory, which does not
obviously point at a stray source file.

The weights inside were intact. `repair_checkpoint.py` rebuilds the archive with
only the entries torch expects, dropping the one stray file. The repaired model
loads cleanly, fits a 29-class MobileNetV3-Large with zero missing and zero
unexpected keys, and scores 97.2%.

The broken original is kept so the failure stays reproducible; the test suite
asserts it is still unloadable.

**Three scripts pointed at a spreadsheet that was not there.** `ai_merged.py`,
`updater.py` and `scraper.py` all referenced `e-waste datasets.xlsx` in the
project root. It actually lives in `emadini/`, so all three exited immediately
on a missing-file check. Paths are now resolved relative to the script file, so
they also work regardless of the current directory.

**`scraper.py` used `Path` without importing it**, which would have raised
`NameError` even after the path was fixed.

## What the model does

`ai_merged.py` merges scraped e-waste records and filters image folders using
ImageNet-pretrained MobileNetV3 to keep only images that plausibly contain an
electronic device. `split_dataset.py` splits the result 80/20. `train_model.py`
trains a 29-class MobileNetV3-Large. `predict.py` runs inference.

`labels.txt` defines the 29 classes, and its order *is* the model's class index —
reordering it silently breaks every prediction. The test suite checks it against
the dataset folders for exactly that reason.

## Honest limitations

**The dataset is very small.** 71 images across 29 classes — roughly 2.4 images
per class. 97.2% on 71 images is not a meaningful generalisation figure; it
mostly confirms the model loaded and learned something. A real evaluation needs
hundreds of images per class and a held-out test set the model never saw.

**The 97.2% is measured on the same data the model was built from.** It is a
sanity check, not a benchmark.

**Two classes are confusable by design.** `dataset_split` holds 42 training and
29 validation images. The two misclassifications are a Samsung Blu-ray Player
read as one and an iPhone 12 read as a Surface Pro 7 — black rectangular
consumer devices, genuinely hard to tell apart at this resolution and dataset
size.

**The filter is an ImageNet model, not an e-waste model.** `ai_merged.py` keeps
images whose top-3 ImageNet predictions fall in a hand-listed set of 24 class
IDs. That set is a guess, and it is what determines which images enter the
dataset in the first place, so filtering errors become training errors.

## Tests

```bash
python -m unittest test_pipeline -v
```

11 tests. They deliberately avoid asserting specific logits, which would break
on every retrain and prove nothing. Instead they check the machinery around the
model: the checkpoint loads, the architecture matches the label count, the
repaired one loads while the original still does not, probabilities form a
distribution, predicted indices stay inside the label range, and accuracy clears
a loose floor that would catch an untrained or mismatched model.

## Layout

```
predict.py            inference
repair_checkpoint.py  rebuilds the corrupt archive
test_pipeline.py      test suite
train_model.py        training
split_dataset.py      80/20 split
ai_merged.py          dataset merge and device filtering
scraper.py            image scraping (needs icrawler)
updater.py            dataset top-up (needs icrawler)
labels.txt            the 29 classes, in class-index order
emadini/              source spreadsheet
```
