# 🧪 HepGPU Demos

Collection of simple examples to showcase the capabilities of **HepGPU**.

---

## 📘 Examples

- **1D Chronic vs Recovery**  
  Comparison between chronic infection and recovery regimes.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/TU_REPO/blob/main/demos/01_1D_cases_clean.ipynb)

---

- **1D Heterogeneous Domain**  
  Simulation with a spatial barrier blocking transport.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/TU_REPO/blob/main/demos/02_1D_heterogeneous_domain_clean.ipynb)

---

- **2D Barrier Case**  
  Wave propagation and immune transport in a domain with obstacles.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/TU_REPO/blob/main/demos/03_2D_barrier.ipynb)

---

- **3D Basic Simulation**  
  Full 3D simulation in a simplified domain.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/TU_REPO/blob/main/demos/04_3D_basic.ipynb)

---

- **3D Segmented Liver**  
  Simulation on a realistic liver geometry using medical segmentation data.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/TU_REPO/blob/main/demos/05_3D_segmented_liver.ipynb)

---

## ⚙️ Notes

- Simulations can run on CPU or GPU.
- Results are exported to VTK and PNG.
- In the **segmented liver case**, the `.nrrd` file is located in the `data/` folder.
- You can use other liver segmentations generated from CT scans (TAC) using tools such as **3D Slicer**.
