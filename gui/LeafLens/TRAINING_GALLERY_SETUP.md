# LeafLens training image gallery patch

Extract this ZIP inside your existing `LeafLens_Simple` folder (not a new project), then run `python apply_training_gallery.py`. The installer edits only the final page-rendering block of `app.py` and makes a backup. Models, CSS, data, config and prediction code are unchanged. It is safe to run the installer again.

The existing `data/train.csv` contains 8,000 labelled training rows. You need the actual photographs from the group's original DPD-Small dataset. Place them in `data/train_images/`, retaining their exact filenames, or create/update `config.json` with `"training_image_root": "/absolute/path/to/DPD_Small_dataset"`. If a config already exists, change only that field. A full DPD-Small image folder can be used: only filenames in the training CSV are opened, so test images are never retrieved as reference examples. Do not rename the photographs. No model training, feature indexing, checkpoint changes or new dependencies are required.

After a successful prediction the new gallery appears below the analysis. For a seen predicted class it shows up to six exact training-label matches. For an unseen predicted class it shows up to three same-disease/different-crop examples and three same-crop/different-disease examples, separately labelled. If exact files are unavailable it says so instead of claiming the combination is unseen. References are selected by class labels, not visual nearest-neighbour search. Missing or unreadable images are handled without blocking prediction.

Run `python -m unittest discover -s tests -p 'test_training_gallery.py'` for the new tests. Restart with `python -m streamlit run app.py`. To undo the installation, restore `app.py.before_training_gallery.bak` to `app.py`.
