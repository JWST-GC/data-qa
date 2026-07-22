# $\text{EMPSignature\_Report}$: Addressing Outlier Over-Rejection of PSF Signal in $\textit{crf}$ Products

## Executive Summary

The current implementation of the data quality flagging (specifically `OUTLIER` and `JUMP_DET`) steps within the Critical/Recovery Frame ($\textit{crf}$) product incorrectly flags genuine, smooth flux signal originating from the Point Spread Function (PSF) wings and diffraction spikes of bright stars. These pixels possess monotonically rising ramps characteristic of PSF structure but are mistakenly rejected as artifacts (Cosmic Rays or jumps).

This bounty proposes a **localized, conditional recovery mechanism** that selectively bypasses $\text{OUTLIER}$/$\text{JUMP\_DET}$ flagging only within a predefined geometric neighborhood of detected bright star cores. This ensures the recovery of real astrophysical signal without compromising the integrity of genuine far-field artifacts (e.g., CR showers or detector snowballs), thus preserving the global utility of the $\textit{crf}$ data product.

***

## I. Root Cause Analysis and Mechanism Diagnosis

### 1. The Conflict
The detection pipeline treats a sudden, sharp change in signal ($\text{JUMP\_DET}$) or any deviation from the dither-median reconstruction ($\text{OUTLIER}$) as an undesirable artifact (e.g., CR). While effective for transient events like high-energy particle hits, this methodology fails when applied to the predictable, steep gradients inherent to bright star PSFs and diffraction spikes.

### 2. Differential Failure Modes
| Flag Type | Condition Violated by PSF Signal | Why it Fails | Solution Required |
| :--- | :--- | :--- | :--- |
| **`JUMP_DET`** (Jump Detection) | The bright, smooth ramp ($45 \to 298 \to 799$) is interpreted as a non-continuous jump or sharp step between dither positions. | PSF flux gradients are steepest near the core/spikes, mimicking abrupt steps when undersampled. | Modify sensitivity in the immediate star neighborhood; allow for smooth, high-slope ramp transitions. |
| **`OUTLIER`** (Outlier Detection) | The distinct, narrow spike structure deviates significantly from the median signal profile built across dither positions. | Spikes are highly structured and often undersampled relative to the median background/PSF wing average, triggering rejection. | Mask or attenuate the $\text{OUTLIER}$ test in known PSF/spike regions based on star position. |

### 3. Scope Confirmation (The Differential)
Crucially, the mechanism is *star-local*. The two distinct populations of flagged pixels confirm this:

1.  **Target Signal:** Clustered near bright stars; exhibit monotonic ramps compatible with stellar emission profiles.
2.  **Genuine Artifacts:** Far-field and distributed; clustered but lacking local correlation to a dominant PSF source, consistent with CR showers.

The proposed solution must maintain this separation rigorously.

***

## II. Proposed Technical Solution Architecture (Code Implementation Strategy)

The remediation requires adding a specialized, star-centric masking layer *before* the final application of $\text{OUTLIER}$ and $\text{JUMP\_DET}$. This mechanism should operate on a per-star basis.

### 1. System Modification Target
Modify the pipeline segment responsible for calculating and applying quality flags $D_{QC} = \text{Apply}(\text{CRF}, \text{DQ}_{\text{Outlier}}, \text{DQ}_{\text{Jump}})$. We introduce a pre-flag mask, $\mathbf{M}_{PSF}$, that defines regions to *ignore* the flag tests.

### 2. Algorithm Flow: PSF-Aware Flagging ($D'_{QC}$)

The updated quality flag determination for any pixel $(x, y)$ becomes:
$$ \text{Flag}_{(x,y)} = \begin{cases} \text{Current Signal Quality Flags} & \text{if } (x, y) \notin \mathbf{M}_{PSF} \\ \text{PASS} & \text{if } (x, y) \in \mathbf{M}_{PSF} \text{ AND } \mathbf{M}_{\text{Gradient}}(x,y) \ge T_{\text{smooth}} \\ \text{Original Flag} & \text{otherwise} \end{cases} $$

Where:
*   $\mathbf{M}_{PSF}$: The Star-Local PSF Mask (a boolean mask defining the neighborhood).
*   $\mathbf{M}_{\text{Gradient}}$: A gradient check to ensure the flag bypass is only activated if the ramp signal is smooth, mitigating the risk of passing true CRs.

### 3. Detailed Implementation Steps

**Step A: Star Detection and Mask Generation ($\mathbf{M}_{PSF}$)**
1.  Identify all stellar cores in the frame (using standard PSF fitting/detection algorithms).
2.  For each star center $S_i(x, y)$, generate a local mask $\mathbf{M}_i$. This mask defines an elliptical or circular region centered on $S_i$, truncated at a maximum radius $R_{\text{max}}$ (e.g., 50 pixels) and constrained by the observed PSF angular extent.
3.  Combine these into the global star-local mask: $\mathbf{M}_{PSF} = \bigcup \mathbf{M}_i$.

**Step B: Smoothness Validation ($\mathbf{M}_{\text{Gradient}}$)**
To prevent restoring CRs, any pixel flagged for exemption must pass a local smoothness test. We analyze the flux gradient $G(x, y)$ along the ramp direction (the dither pattern).
1.  Calculate the rate of change in accumulated flux $\Delta F / \Delta N$ across neighboring pixels within the potential spike/wing structure.
2.  If the signal segment is highly monotonic and changes smoothly, indicating real PSF wing structure, grant a high confidence score $C_{smooth}$. If the change is abrupt (characteristic step function), reject the exemption proposal.

**Step C: Flag Application Logic**
The final flag assignment for pixel $(x, y)$ must adhere to:
$$ \text{New\_DQ}_{(x,y)} = \text{AND}(\text{Original Flags}, \text{Mask}_{\text{PSF}}) \lor (\text{PASS} \land (x, y) \in \mathbf{M}_{PSF} \land C_{smooth}) $$

This implementation treats the region as "unflagged" only if it is confirmed to be within a star-local PSF structure *and* exhibits smooth signal behavior. Otherwise, the original flag status ($\text{OUTLIER}$/$\text{JUMP\_DET}$) remains binding.

***

## III. Pseudo-Code Implementation Mockup (Python/Astropy Ecosystem)

This mockup demonstrates the required logic flow, assuming standard Astropy/JWST pipeline utilities are available.

```python
import numpy as np
from astropy.stats import sigma_clipped_skimage
from jwst.datamodels import dqflags 

def recover_psf_signal_flagging(crf_data: np.ndarray, dq_raw: np.ndarray, star_centers: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies localized flag recovery around bright stars without impacting global flags.

    Args:
        crf_data: The primary CRFs data array (flux).
        dq_raw: Raw DQ integer array.
        star_centers: List of (x, y) coordinates for detected star cores.

    Returns:
        Tuple containing the modified DQ array and a boolean mask 
        identifying which pixels were successfully recovered.
    """
    
    # Initialize the output quality flag array (copying raw flags initially)
    dq_modified = dq_raw.copy()
    recovery_mask = np.zeros(dq_raw.shape, dtype=bool)

    # --- Global Mask for Stars ---
    psf_local_mask = np.zeros(dq_raw.shape, dtype=bool)
    for (x_center, y_center) in star_centers:
        # Define a local bounding box/kernel around the star core
        # R_max determines how far out we look for signal remnants (e.g., 50 pixels)
        bbox = np.ogrid[y_center - 60:y_center + 60, x_center - 100:x_center + 100]
        local_mask = bbox[0, :, None] * bbox[1, None, :]
        # Simple approximation of the local mask (replace with proper distance calculation)
        psf_local_mask |= np.clip(np.sqrt((bbox[0][:, None] - y_center)**2 + 
                                        (bbox[1][None, :] - x_center)**2), 
                                      a_min=0, a_max=1) > 0.8

    # --- Apply Conditional Recovery Logic ---
    for i in range(dq_raw.shape[0]):
        for j in range(dq_raw.shape[1]):
            
            pixel = (i, j)
            current_flag = dq