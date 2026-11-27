# Upload remaining checkpoints to GitHub Releases

**Norman's five PT files are already on GitHub:** https://github.com/tylsee/aml-group2/releases/tag/new-approach-models

Seven verified source files remain in Google Drive. Use the links in [checkpoint_inventory.csv](checkpoint_inventory.csv) to download them under the authorized account. Rename them to the unique `recommended_asset_filename` values if convenient. The original binary contents must not be changed.

1. Open https://github.com/tylsee/aml-group2/releases and choose **Draft a new release**.
2. Create a new tag (e.g. `group2-other-checkpoints`) targeting `main`.
3. Name it **Baseline 1, Baseline 2 and CDC-CP checkpoints**.
4. Upload the seven files as **release attachments**, not repository commits. Each is smaller than GitHub Releases' 2 GiB per-asset limit.
5. Publish the release, check every file's name and size, and update this inventory with their actual GitHub Release download links.

**Why:** GitHub rejects ordinary Git blobs over 100 MiB. Several final weights exceed that threshold, including Baseline 2 ResNet and ConvNeXt. Releases keep the Git history lightweight; Git LFS is another option but needs an LFS-capable client and quota.

The connected GitHub integration can create Git commits and documentation but does **not** expose an action for uploading Release assets or LFS objects. No files have been removed from Google Drive and no upload to Releases is being claimed here.

Selection confirmation is still required for Baseline 1's C1/D1 pairing and Baseline 2's single approach-level final. Drive sharing needs to allow the intended assessor to download files.
