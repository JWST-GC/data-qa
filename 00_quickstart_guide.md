
# JWST-GC Quality Assessment Guide

This guide is for those getting started with Quality Assessment of the JWST-GC Treasury Program 10678.

## Important Links

- **QA starting point:**  https://github.com/orgs/JWST-GC/projects/2

- **Detailed explanation of all QA plots and numbers**: https://github.com/JWST-GC/data-qa/blob/main/docs/qa_methods.md


## Brief Description

The Quality Assessment process is managed through GitHub's Projects system:
https://github.com/orgs/JWST-GC/projects/2

This is an overview page and should be your starting point.

Each time a new field gets observed, a GitHub **issue** will be created automatically just for that field. The issue will appear on the **Projects** page, but can also be viewed in the list of issues in the [`data-qa` repository](https://github.com/JWST-GC/data-qa/issues?q=is%3Aissue%20state%3Aopen%20label%3ANIRCam).

Each issue has **stages** for QA review posted as individual comments. The stage comments are automatically updated as newer reduction products become available.

The very basic summary of each stage is:

- Stage 1 Do data exist?
- Stage 2 Do catalogs exist?
- Stage 3 How is the absolute photometry?
- Stage 4 How is the absolute astrometry?
- Stage 5 Are detectors well-aligned astrometrically?
- Stage 6: What is the astrometric precision?
- Stage 7: How do jicama pipeline products compare to MAST-delivered products?
- Stage 8: What is the relative spatial distortion between two filters?
- Stage 9: How do PSF and aperture photometry compare?
- Stage 10: Are source positions consistent across exposures?
- Stage 11: Did any exposures lose tracking?

## Brief Workflow

1. Go to https://github.com/orgs/JWST-GC/projects/2.
2. Click on the title of one of the issues.
3. Assign yourself to this issue using **Assignees** in the column view or in the top-right corner, and update the **Status** of the issue from **Todo** to **In progress**.
4. Review plots and numeric values in each **stage**.
5. Mark the corresponding boxes in the very first comment as completed.
6. Point out any anomalies by leaving a comment.
7. If everything is checked and resolved, mark the issue as **Done**.

For a more detailed explanation of each stage, as well as what to look for in the plots, see:
https://github.com/JWST-GC/data-qa/blob/main/docs/qa_methods.md.
The **"how this plot & its numbers are made"** link in each stage points to that page. The instructions are intentionally kept vague to not influence the review process.

## General Tips and Other Information

If you are unsure whether something is actually wrong or worth highlighting, leave a comment anyway.

The QA process uses catalogs from the jicama pipeline. You can learn more about this particular pipeline here:
https://github.com/keflavich/jwst-gc-pipeline/blob/main/PHOTOMETRY_PIPELINE_BRIEF.md.
In short, the catalogs are created by iteratively fitting and subtracting PSFs on individual frames rather than stacked mosaics.


You can download data and catalogs by selecting the relevant field here:
https://starformation.astro.ufl.edu/jwst-gc/index.html. The downloads are done through Globus, and the products are stored on UF's supercomputer.

There are about two dozen issues for data other than the JWST-GC 10678 program. These can generally be ignored, but might be useful as a reference.


## All Other Links

- **Underlying code that makes plots in the issues:**  
  https://github.com/JWST-GC/data-qa/blob/main/data_qa/diagnostics.py

- **Project homepage:**  
  https://sites.google.com/view/jwst-gc/

- **Project observations and data-reduction status:**  
  https://starformation.astro.ufl.edu/jwst-gc/monitor/

- **Project observations status by Ashley Barnes:**  
  https://ashleythomasbarnes.github.io/jwst_gc_dashboard/

- **QA-related GitHub issues:**  
  https://github.com/JWST-GC/data-qa/issues?q=is%3Aissue%20state%3Aopen%20label%3ANIRCam