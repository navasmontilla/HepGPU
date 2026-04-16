# HepGPU Usage Examples

Collection of simple examples showcasing the capabilities of `HepGPU`.

## Examples

### Step 1D

- **1D Chronic vs Recovery**  
  Comparison between chronic infection and recovery regimes.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/demos/Step2_1D/1D_cases.ipynb)

- **1D Heterogeneous Domain**  
  Simulation with a spatial barrier blocking transport.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/demos/Step2_1D/1D_heterogeneous_domain.ipynb)

### Step 2D

- **2D No Barrier Case**  
  Wave propagation and immune transport in a basic domain.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/demos/Step3_2D/2D_no_barrier.ipynb)

- **2D Barrier Case**  
  Wave propagation and immune transport in a domain with obstacles.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/demos/Step3_2D/2D_barrier.ipynb)

### 3D Liver Model

- **3D Segmented Liver**  
  Simulation on a realistic liver geometry using medical segmentation data.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/demos/3D_liver_model/3D_liver_segmented.ipynb)


## Notes

- Each example is provided as:
  - `.ipynb` → for interactive execution in Google Colab  
  - `.py` → for local or GPU-based execution
- Simulations can be run on CPU or GPU (e.g., via Google Colab), although GPU support may have some limitations.
- Results are exported to VTK and PNG.
- In the **segmented liver case**, the `.nrrd` file is located in the `data/` folder.
- Custom liver segmentations can be generated from CT scans (TAC) using tools such as **3D Slicer**.
