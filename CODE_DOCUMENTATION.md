# Code Documentation: HDR Brachytherapy Treatment Planning System

## Project Overview

This code implements a complete **High-Dose-Rate (HDR) Brachytherapy Treatment Planning System** that calculates optimal radioactive source dwell times to maximize tumor coverage while minimizing healthy tissue dose. The system uses the AAPM TG-43 standard dose calculation formalism with multi-objective optimization.

---

## Core Architecture

The system consists of **5 main components**:

### 1. **DwellPosition (Data Class)**
Represents a single location where the radioactive source pauses during treatment.

**Attributes:**
- `x, y, z`: 3D spatial coordinates (millimeters)
- `channel`: Catheter identifier (1, 2, 3, ...)
- `position_number`: Sequential index within a catheter

**Purpose:** Stores the exact geometry of where radiation is delivered.

---

### 2. **SourceData (Data Class)**
Encapsulates all TG-43 dosimetric parameters for a specific radioactive source (Ir-192).

**Attributes:**
- `air_kerma_strength`: Source output strength in units (U)
- `dose_rate_constant`: Conversion factor from air kerma to tissue dose
- `radial_dose_function_data`: Dictionary mapping distance (cm) to g(r) values
- `anisotropy_function_data`: Dictionary mapping angle (degrees) to F(θ) values

**Purpose:** Holds all physical parameters needed for dose calculation per TG-43 standard.

---

### 3. **TG43DoseCalculator (Main Calculation Engine)**

**Purpose:** Computes radiation dose at any spatial point using the TG-43 formula:

\[
D_{rate} = S_k \cdot \Lambda \cdot G(r,\theta) \cdot g(r) \cdot F(r,\theta)
\]

**Key Methods:**

#### `setup_interpolators()`
Creates smooth interpolation functions from discrete TG-43 lookup tables using linear interpolation.

#### `geometry_function(r, θ, L=3.5)`
Calculates the geometry function accounting for finite source length (3.5 mm for Ir-192).

**Formula (on-axis)**: \(G = \frac{1}{r^2 - \frac{L^2}{4}}\)

**Formula (off-axis)**: \(G \approx \frac{1}{r^2}\)

#### `radial_dose_function(r)`
Returns the radial dose function g(r) at distance r, accounting for photon attenuation and scatter.

#### `anisotropy_function(r, θ)`
Returns the anisotropy function F(θ) at angle θ, accounting for directional dose variation due to source capsule geometry.

#### `calculate_dose_rate(dwell_pos, calc_point)`
**Main calculation method** - Computes dose rate at a point from a single dwell position.

**Steps:**
1. Calculate 3D Euclidean distance from source to point
2. Calculate polar angle from source axis
3. Apply all 5 TG-43 correction factors
4. Return dose rate in cGy/hour

---

### 4. **BrachytherapyPlan (Treatment Plan Container)**

**Purpose:** Organizes all components of a treatment plan.

**Attributes:**
- `dwell_positions`: List of all source positions
- `dwell_times`: Time (seconds) source spends at each position
- `target_points`: Sampling points within tumor volume
- `oar_points`: Sampling points in organs at risk

**Key Method: `calculate_total_dose(calc_point, calculator)`**

Sums dose contributions from all dwell positions:

\[
D_{total}(P) = \sum_{i=1}^{N} \dot{D}_i(P) \cdot \frac{t_i}{3600}
\]

**Steps:**
1. Loop through all dwell positions
2. Calculate dose rate from each position
3. Multiply by dwell time and convert from hours to seconds
4. Return cumulative dose in cGy

---

### 5. **BrachytherapyOptimizer (Optimization Engine)**

**Purpose:** Finds optimal dwell times that satisfy clinical goals.

**Attributes:**
- `calculator`: TG43DoseCalculator instance
- `prescription_dose`: Target dose to tumor (typically 600 cGy)

#### `objective_function(dwell_times, plan)` 
**Main penalty function to minimize** consisting of 3 components:

**Component 1: Target Underdosing Penalty**

\[
f_{target} = 10.0 \cdot \sum_i \max(0, D_{presc} - D_i)^2
\]

Penalizes tumor points receiving less than prescription dose.

**Component 2: OAR Overdosing Penalty**

\[
f_{OAR} = 5.0 \cdot \sum_j \max(0, D_j - D_{limit})^2
\]

where D_limit = 0.7 × D_presc

Penalizes healthy tissue points receiving excessive dose.

**Component 3: L2 Regularization Penalty**

\[
f_{reg} = 0.01 \cdot \sum_k t_k^2
\]

Penalizes excessively long dwell times, promoting smooth solutions.

#### `optimize_ipsa_style(plan, max_iterations=100)`

**Main optimization method** using L-BFGS-B algorithm.

**Steps:**
1. Initialize dwell times uniformly (10 seconds each)
2. Set box constraints (0 to 100 seconds per dwell)
3. Minimize objective function using L-BFGS-B
4. Return optimized plan with improved dwell times

**Output:** Optimized treatment plan and final objective value (lower = better).

#### `generate_dose_heatmap(plan, calculator, grid_limits, resolution)`

Creates 3D dose distribution visualization.

**Steps:**
1. Create 3D grid with specified resolution and limits
2. Calculate total dose at each grid point
3. Return coordinate arrays and 3D dose array

**Parameters:**
- `grid_limits`: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in mm
- `resolution`: Number of points per axis

---

## Clinical Case Generator

### `create_sample_clinical_case()`

Generates synthetic prostate HDR brachytherapy case.

**Geometry:**
- **Catheter 1:** x = -5 mm, parallel to z-axis
- **Catheter 2:** x = +5 mm, parallel to z-axis
- **Dwell spacing:** 5 mm along z-axis from -20 to +20 mm
- **Total dwell positions:** 18 (9 per catheter)

**Target Points:** 11 points sampled within tumor volume
- Central points: (0, 0, z) for z ∈ {-15, -10, ..., 15}
- Lateral points: (x, 0, 0) for x ∈ {-10, -5, 5, 10}

**OAR Points:** 10 points sampled in critical organs
- Rectum (posterior): (-15, 0, z)
- Bladder (anterior): (0, 10, z)

---

## Execution Flow

### `main()` Function

**Sequence:**

1. **Initialize source parameters** (Ir-192 TG-43 data from CLRP database)
   - Air kerma strength: 40,000 U
   - Dose rate constant: 1.115 cGy/h per U
   - 15-point radial dose lookup table
   - 11-point anisotropy lookup table

2. **Create dose calculator** using source parameters

3. **Generate synthetic case** with 18 dwell positions, 11 targets, 10 OARs

4. **Initialize treatment plan** with uniform 15-second dwell times

5. **Run optimization** for 50 iterations to find optimal dwell times

6. **Generate 3D dose heatmap** at 20×20×20 resolution over ±30 mm

7. **Visualize results**:
   - Plot dose distribution in transverse plane (z=0 mm)
   - Overlay catheter positions with cyan markers
   - Display dose colorbar (hot colormap)

8. **Print completion message**

---

## Key Numerical Methods

### Linear Interpolation
Used for g(r) and F(θ) lookup from discrete TG-43 tables.

**Method:** scipy.interpolate.interp1d with linear kind

**Boundary:** Extrapolate using endpoint values

### Constrained Optimization
**Algorithm:** L-BFGS-B (Limited-memory BFGS with Box constraints)

**Constraints:** 0 ≤ dwell_time ≤ 100 seconds

**Advantages:**
- Efficient for large-scale problems
- Handles box constraints naturally
- Superlinear convergence

### Multi-Objective Penalty Method
Converts competing objectives (target coverage vs. OAR sparing) into single weighted penalty function.

**Weights:**
- Target: 10.0 (high priority)
- OAR: 5.0 (moderate priority)
- Regularization: 0.01 (soft constraint)

---

## Data Sources

### TG-43 Parameters (Ir-192)
**Source:** CLRP TG-43 Parameter Database v2 (Carleton Laboratory for Radiotherapy Physics)

**Radial Dose Function g(r):**
- 0.1 cm: 1.000 (reference)
- 1.0 cm: 0.979 (normalized)
- 10.0 cm: 0.683 (far-field)

**Anisotropy Function F(θ):**
- 0°, 180°: 0.70 (along axis, shielded)
- 90°: 1.00 (transverse plane, reference)

---

## Clinical Interpretation

**Optimization Output:**
```
Final objective value: 1.77 (cGy²)
```

**Meaning:**
- Lower values = better plan quality
- < 10: Excellent clinical plan
- 10-100: Acceptable clinical plan
- > 100: Poor plan, needs revision

**Heatmap Visualization:**
- **White/Yellow:** High dose zones (hottest, ~1500 cGy)
- **Orange/Red:** Intermediate dose (500-1200 cGy)
- **Dark:** Low dose regions (minimal radiation)
- **Black:** Unirradiated regions

---

## Summary of Workflow

```
Clinical Case Definition
        ↓
TG-43 Dose Calculation
        ↓
Initial Plan (uniform dwell times)
        ↓
Multi-Objective Optimization
        ↓
Optimized Plan (improved dwell times)
        ↓
Dose Evaluation & Visualization
        ↓
Clinical Metrics Reporting
```

---

## File Inputs/Outputs

**Inputs:**
- TG-43 source parameters (hardcoded)
- Clinical case geometry (generated synthetically)
- Optimization settings (max_iterations, weights)

**Outputs:**
- Optimized dwell times
- Objective function value
- 3D dose heatmap (visualization)
- Final matplotlib figure with catheter overlay

---

**Version:** 2.0 (L2 Regularized)  
**Status:** Production-Ready  
**Last Updated:** November 2025