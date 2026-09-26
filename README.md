# Cuneiform Tablet Reader

This is my final year project for CM3070 at the University of London. I built a
system that takes a photograph of a Neo-Assyrian clay tablet and works out
which cuneiform signs are on it. It then fills in the signs it could not read
with confidence, and ends with a short plain-English description of what the
tablet is probably about.

To be clear about what it does not do: it is not a translator. The final step
gives a rough, genre-level reading (for example "this looks like a legal
record of a sale"), not a line-by-line translation. I explain why in the
report.

## How it works

The system chains three pre-trained models, and each one works on a different
kind of data.

| Stage | Model | Input | What I did with it |
|---|---|---|---|
| 1. Vision | YOLOv8s (pre-trained on COCO) | Pixels | Fine-tuned it to find and classify cuneiform signs in 512 px tiles cut from the tablet photo |
| 2. Sequence | distilroberta-base | Sequences of sign tokens | Kept the pre-trained body, swapped the vocabulary for 288 sign tokens and trained it as a masked model on transliterated tablets |
| 3. Generative | LLaMA 3.1 8B, run locally with Ollama | English | Prompt only, no training. It turns the recovered sign names and catalogue details into a short summary |

In between, my own orchestration code groups the detected signs into lines,
marks the gaps where a sign should be but was not found, and fuses the vision
model's guess with the language model's guess for each position. If neither
is confident enough, the system shows `[x]` and declines to guess instead of
making something up.

Everything runs offline once it is set up. The only network connection it
makes is to Ollama on the same machine.

## What is in this repository

```
cuneiform/
  config.py            every path and shared constant in one place
  domain1_vision/      tiling, YOLO training and inference, detector evaluation
  domain2_sequence/    vocabulary, corpus, masked language model training
  domain3_generative/  sign list, prompt, generation, Domain 3 evaluation
  orchestrator/        line grouping, gap insertion, fusion, live evaluation
  api/                 FastAPI server that the dashboard talks to
web/                   React + TypeScript dashboard (Vite)
tests/                 pytest suite
artifacts/             trained detector, vocab, sign lists, detections and results
cuneiform_data/        sign annotations and transliterations (CompVis dataset)
yolo_dataset/          YOLO labels for the tiled training set (the tiles themselves are not included)
runs/domain1_local_v1/ a second detector I trained locally, used as a comparison
```

A few files in `cuneiform/` are there on purpose even though they do not work.
`inspect_box_pitch.py` and `inspect_segment_choice.py` both failed their own
validation checks, and I kept them as a record of what I tried and why I
dropped it. The same goes for `prepare_cuneiform_data.py`, which belongs to an
earlier YOLOS-based design I abandoned. The various `inspect_*` and `verify_*`
scripts are one-off checks I ran to confirm each stage worked before building
on it.

## Setting it up

I developed this on Windows with Python 3.12.10 and Node 22.

1. Create a virtual environment and install the Python packages:
   ```
   python -m venv fyp_env
   fyp_env\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
   `requirements.txt` is a straight `pip freeze` of my environment (CPU build
   of PyTorch). `requirements-dev.txt` adds FastAPI, uvicorn and pytest.
2. Install [Ollama](https://ollama.com) and pull the model:
   ```
   ollama pull llama3.1:8b
   ```
   The Ollama app has to be running for the summary step. The first request
   after it has been idle is slow (over a minute on my machine) because the
   model is loading into memory.
3. Install the dashboard's packages:
   ```
   cd web
   npm install
   ```
4. Model weights. The fine-tuned detector is included at
   `artifacts/domain1/weights/domain1_best.pt`. The Domain 2 checkpoint is
   167 MB, which is over GitHub's file size limit, so it is not stored in the
   repository itself. It belongs in
   `artifacts/domain2/checkpoints/domain2_distilroberta_pretrained/`.
   **[How to get it: to be added.]**

5. Tablet photographs. These are **not** in the repository. The 75 photos I
   worked from come from the Cuneiform Digital Library Initiative (CDLI)
   website, and I don't have permission to republish them. That also rules out
   anything made from them: the 512 px training tiles, the photos with boxes
   drawn on them, and the training-batch previews. To download the originals
   into `cuneiform_data/images/`, run this from the project root:
   ```
   python -m cuneiform.domain1_vision.scrape_cdli_images
   ```
   I haven't re-run this since I first collected the images, so I can't promise
   CDLI still serves them at the same addresses. Without the photos, the
   command-line pipeline and the evaluation still run, because they work from
   the saved detection files in `artifacts/domain1/live_detections/`. What
   breaks is the photograph pane in the dashboard, and the handful of tests in
   `tests/test_api.py` and `tests/test_offline.py` that load a tablet photo
   (the image route and upload tests). Uploading your own photo in the
   dashboard works either way.

`start.ps1` is the script I use to open a session. It activates the
environment and keeps the Hugging Face and pip caches on my D: drive. The paths
in it are specific to my machine, so on another machine just activate the
environment yourself.

## Running it

Every script is run as a module from the project root, never from inside a
package folder.

Read one tablet from the command line:
```
python -m cuneiform.orchestrator.live_pipeline --tablet P335940
```

Start the dashboard. This needs two terminals:
```
uvicorn cuneiform.api.main:app          # terminal 1, from the project root
cd web && npm run dev                   # terminal 2
```
Then open http://localhost:5173. Use `localhost` rather than `127.0.0.1`,
because the Vite dev server only listens on the IPv6 loopback address.

The dashboard lists the 30 test tablets and also lets you upload your own
photo (JPEG, PNG or TIFF; iPhone HEIC files are not supported). For each tablet it
shows the photo with the detected boxes, the recovered sign sequence (with
each sign marked by where it came from), a tentative reading from LLaMA along
with a check of whether every sign it cited really appears in the input, and a
provenance panel showing how many detections survived each step.

Re-run the main evaluation over the 30 test tablets:
```
python -m cuneiform.orchestrator.evaluate_live
python -m cuneiform.orchestrator.significance_live
```

## Tests

```
python -m pytest                    # the full Python suite
python -m pytest -m "not slow"      # skips tests that load the big checkpoint
cd web && npm run test              # dashboard component tests
```

A few of the Python tests only run when the Ollama app is on, and a few only
run when it is off (they check that the server fails cleanly), so every run
skips some tests. That is expected.

## Results

The main question I tested was whether combining the vision model with the
language model reads a tablet better than either design on its own. I
measured it on real detector output over 30 test tablets, and compared each
tablet with itself under each design:

- Against the vision model alone, fusion cut the word error rate by 0.035
  (25 tablets better, none worse) and the character error rate by 0.073 (all
  30 better).
- Against a simpler design that only asks the language model about
  low-confidence signs, fusion still cut word error rate by 0.029 (p = 0.0004).
  It did **not** improve character error rate against that design, and I
  report that as a limitation rather than hiding it.

The absolute error rates are high, partly because the detector finds more or
fewer boxes than there are real signs on a tablet. That is why I only compare
designs against each other on the same tablets. The report covers all of this
in detail, including the detector's own accuracy and the limits of the Domain 3
summaries.

## Data and credits

- **CompVis cuneiform sign detection dataset**: tablet images, sign bounding
  boxes and transliterations. Dencker, T., Klinkisch, P., Maul, S.M. and Ommer, 
  B. (2020) ‘Deep learning of cuneiform sign detection with weak supervision 
  using transliteration alignment’, PLOS ONE, 15(12), e0243039. Available at: 
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039.
- **Oracc State Archives of Assyria online (SAAo)**, volumes SAA 1 and SAA 6:
  catalogue metadata for the tablets. Released under CC0.
- **Oracc Global Sign List (OGSL)**: used to map sign numbers to sign names.
  Released under CC0. The full checkout is not in this repository;
  `python -m cuneiform.domain3_generative.fetch_signlist` downloads it again,
  and the files built from it are already in `artifacts/domain3/`.
- **Pre-trained models**: YOLOv8s by Ultralytics, distilroberta-base from
  Hugging Face, and LLaMA 3.1 8B by Meta through Ollama.

I did not use or redistribute the published English translations of these
tablets. They are scholarly work that is not licensed for bulk reuse.
