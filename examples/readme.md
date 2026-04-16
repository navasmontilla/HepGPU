# HepGPU Usage Examples

Collection of examples showcasing the capabilities of `HepGPU` for simulating hepatitis B dynamics.

Each example is provided in two formats:
- `.ipynb` → interactive execution in Google Colab  
- `.py` → execution in local environments or GPU systems  

Simulations can be run on CPU or GPU (e.g., via Google Colab), although GPU support may have some limitations.  
Results are exported to VTK and PNG for visualization.

In the **segmented liver case**, the required `.nrrd` file is located in the `data/` folder.  
Custom liver segmentations can also be generated from CT scans (TAC) using tools such as **3D Slicer**.

## 📁 Step2_1D

- 1D Chronic vs Recovery: Comparison between chronic infection and recovery regimes.  
  - Notebook: [Open in Colab](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/examples/Step2_1D/1D_cases.ipynb)  
  - Script: `Step2_1D/1D_cases.py`
- 1D Heterogeneous Domain: Simulation with a spatial barrier blocking transport.  
  - Notebook: [Open in Colab](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/examples/Step2_1D/1D_heterogeneous_domain.ipynb)  
  - Script: `Step2_1D/1D_heterogeneous_domain.py`

## 📁 Step3_2D

- 2D No Barrier Case: Wave propagation and immune transport in a basic domain.  
  - Notebook: [Open in Colab](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/examples/Step3_2D/2D_no_barrier.ipynb)  
  - Script: `Step3_2D/2D_no_barrier.py`
- 2D Barrier Case: Wave propagation and immune transport in a domain with obstacles.  
  - Notebook: [Open in Colab](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/examples/Step3_2D/2D_barrier.ipynb)  
  - Script: `Step3_2D/2D_barrier.py`

## 📁 3D_liver_model

- 3D Segmented Liver: Simulation on a realistic liver geometry using medical segmentation data.  
  - Notebook: [Open in Colab](https://colab.research.google.com/github/navasmontilla/HepGPU/blob/main/examples/3D_liver_model/3D_liver_segmented.ipynb)  
  - Script: `3D_liver_model/3D_liver_segmented.py`
