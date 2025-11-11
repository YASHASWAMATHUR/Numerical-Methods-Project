"""
Brachytherapy Treatment Planning
--------------------------------
Implements:
  1. TG-43 dose calculation formalism (AAPM standard)
  2. Linear Penalty Model for multi-objective optimization
  3. L2 regularization for smooth and stable dwell times
  4. 3D dose and catheter placement visualization (Matplotlib or Plotly)
  5. Synthetic clinical case generator
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ======================================================================
# --------------------------- Data Classes -----------------------------
# ======================================================================

@dataclass
class DwellPosition:
    """Defines a single dwell position in 3D space."""
    x: float
    y: float
    z: float
    channel: int
    position_number: int


@dataclass
class SourceData:
    """TG-43 dosimetric parameters for a radioactive source."""
    air_kerma_strength: float
    dose_rate_constant: float
    radial_dose_function_data: Dict[float, float]
    anisotropy_function_data: Dict[float, float]


# ======================================================================
# --------------------------- Dose Calculator --------------------------
# ======================================================================

class TG43DoseCalculator:
    """TG-43 formalism dose calculator."""

    def __init__(self, source_data: SourceData):
        self.source_data = source_data
        self.setup_interpolators()

    def setup_interpolators(self):
        """Interpolators for g(r) and F(theta)."""
        r_values = np.array(list(self.source_data.radial_dose_function_data.keys()))
        g_values = np.array(list(self.source_data.radial_dose_function_data.values()))
        self.radial_dose_interp = interp1d(
            r_values, g_values, kind='linear', bounds_error=False,
            fill_value=(g_values[0], g_values[-1])
        )

        theta_values = np.array(list(self.source_data.anisotropy_function_data.keys()))
        f_values = np.array(list(self.source_data.anisotropy_function_data.values()))
        self.anisotropy_interp = interp1d(
            theta_values, f_values, kind='linear', bounds_error=False,
            fill_value=(f_values[0], f_values[-1])
        )

    def geometry_function(self, r: float, theta: float, L: float = 3.5) -> float:
        """Geometry function G(r,θ)."""
        L_cm, r_cm = L / 10.0, r / 10.0
        if abs(theta) < 1 or abs(theta - 180) < 1:
            if r_cm > L_cm / 2:
                return 1.0 / (r_cm * r_cm - L_cm * L_cm / 4.0)
            else:
                return 1.0 / (r_cm * r_cm)
        else:
            return 1.0 / (r_cm * r_cm)

    def radial_dose_function(self, r: float) -> float:
        """Radial dose function g(r)."""
        r_cm = r / 10.0
        return float(self.radial_dose_interp(r_cm))

    def anisotropy_function(self, r: float, theta: float) -> float:
        """Anisotropy function F(r,θ)."""
        theta = abs(theta) % 180
        return float(self.anisotropy_interp(theta))

    def calculate_dose_rate(self, dwell_pos: DwellPosition, calc_point: Tuple[float, float, float]) -> float:
        """Dose rate (cGy/h) at a point using TG-43 formalism."""
        dx, dy, dz = calc_point[0] - dwell_pos.x, calc_point[1] - dwell_pos.y, calc_point[2] - dwell_pos.z
        r = max(np.sqrt(dx**2 + dy**2 + dz**2), 2.0)
        theta = np.degrees(np.arccos(abs(dz) / r)) if r > 0 else 90

        S_k = self.source_data.air_kerma_strength * 1e-6
        Lambda = self.source_data.dose_rate_constant
        G = self.geometry_function(r, theta)
        g = self.radial_dose_function(r)
        F = self.anisotropy_function(r, theta)

        return S_k * Lambda * G * g * F * 1e6


# ======================================================================
# --------------------------- Treatment Plan ---------------------------
# ======================================================================

class BrachytherapyPlan:
    """Represents complete brachytherapy plan."""

    def __init__(self, dwell_positions: List[DwellPosition], dwell_times: List[float],
                 target_points: List[Tuple[float, float, float]],
                 oar_points: List[Tuple[float, float, float]] = None):
        self.dwell_positions = dwell_positions
        self.dwell_times = dwell_times
        self.target_points = target_points
        self.oar_points = oar_points or []

    def calculate_total_dose(self, calc_point: Tuple[float, float, float],
                             calculator: TG43DoseCalculator) -> float:
        """Cumulative dose (cGy) at a point from all dwell positions."""
        total_dose = 0.0
        for i, dwell_pos in enumerate(self.dwell_positions):
            if i < len(self.dwell_times) and self.dwell_times[i] > 0:
                dose_rate = calculator.calculate_dose_rate(dwell_pos, calc_point)
                dose = dose_rate * self.dwell_times[i] / 3600.0
                total_dose += dose
        return total_dose


# ======================================================================
# ----------------------------- Optimizer ------------------------------
# ======================================================================

class BrachytherapyOptimizer:
    """Performs IPSA-style optimization with L2 regularization."""

    def __init__(self, calculator: TG43DoseCalculator, prescription_dose: float = 600.0):
        self.calculator = calculator
        self.prescription_dose = prescription_dose

    def objective_function(self, dwell_times: np.ndarray, plan: BrachytherapyPlan) -> float:
        """Multi-objective penalty model."""
        plan.dwell_times = dwell_times.tolist()
        total_penalty = 0.0

        # 1. Underdose penalty
        target_weight = 10.0
        for target_point in plan.target_points:
            dose = plan.calculate_total_dose(target_point, self.calculator)
            if dose < self.prescription_dose:
                total_penalty += target_weight * (self.prescription_dose - dose) ** 2

        # 2. OAR overdose penalty
        oar_weight = 5.0
        oar_limit = self.prescription_dose * 0.7
        for oar_point in plan.oar_points:
            dose = plan.calculate_total_dose(oar_point, self.calculator)
            if dose > oar_limit:
                total_penalty += oar_weight * (dose - oar_limit) ** 2

        # 3. L2 regularization
        reg_weight = 0.01
        total_penalty += reg_weight * np.sum(dwell_times ** 2)
        return total_penalty

    def optimize_ipsa_style(self, plan: BrachytherapyPlan, max_iterations: int = 100) -> BrachytherapyPlan:
        """Run IPSA-style optimization (L-BFGS-B)."""
        n = len(plan.dwell_positions)
        initial = np.ones(n) * 10.0
        bounds = [(0.0, 100.0)] * n

        result = minimize(self.objective_function, x0=initial, args=(plan,),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iterations})

        optimized_plan = BrachytherapyPlan(plan.dwell_positions, result.x.tolist(),
                                           plan.target_points, plan.oar_points)
        print(f"Optimization completed. Final objective: {result.fun:.2f}")
        return optimized_plan

    def generate_dose_heatmap(self, plan: BrachytherapyPlan, calculator: TG43DoseCalculator,
                              grid_limits=((-30, 30), (-30, 30), (-30, 30)),
                              resolution=20):
        """Generate 3D dose distribution grid."""
        xs = np.linspace(grid_limits[0][0], grid_limits[0][1], resolution)
        ys = np.linspace(grid_limits[1][0], grid_limits[1][1], resolution)
        zs = np.linspace(grid_limits[2][0], grid_limits[2][1], resolution)

        dose_grid = np.zeros((resolution, resolution, resolution))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    dose_grid[i, j, k] = plan.calculate_total_dose((x, y, z), calculator)
        return xs, ys, zs, dose_grid


# ======================================================================
# --------------------- Synthetic Clinical Case ------------------------
# ======================================================================

def create_sample_clinical_case():
    """Generate synthetic prostate HDR case."""
    catheter1 = [DwellPosition(-5, 0, z, 1, i) for i, z in enumerate(range(-20, 21, 5))]
    catheter2 = [DwellPosition(5, 0, z, 2, i) for i, z in enumerate(range(-20, 21, 5))]
    dwell_positions = catheter1 + catheter2

    target_points = [(0, 0, z) for z in range(-15, 16, 5)] + [(x, 0, 0) for x in [-10, -5, 5, 10]]
    oar_points = [(-15, 0, z) for z in range(-10, 11, 5)] + [(0, 10, z) for z in range(-10, 11, 5)]
    return dwell_positions, target_points, oar_points


# ======================================================================
# ---------------------------- Main Script -----------------------------
# ======================================================================

def main():
    print("="*75)
    print("BRACHYTHERAPY DOSE PLANNING AND OPTIMIZATION")
    print("="*75)

    # Initialize TG-43 source parameters
    ir192 = SourceData(
        air_kerma_strength=40000,
        dose_rate_constant=1.115,
        radial_dose_function_data={
            0.1: 1.000, 0.25: 1.000, 0.5: 0.994, 0.75: 0.987, 1.0: 0.979,
            1.5: 0.964, 2.0: 0.948, 2.5: 0.932, 3.0: 0.915, 4.0: 0.881,
            5.0: 0.847, 6.0: 0.813, 7.0: 0.779, 8.0: 0.746, 10.0: 0.683
        },
        anisotropy_function_data={
            0: 0.70, 10: 0.77, 30: 0.94, 45: 0.97, 60: 0.99, 90: 1.00,
            120: 0.99, 135: 0.97, 150: 0.94, 170: 0.77, 180: 0.70
        }
    )

    calculator = TG43DoseCalculator(ir192)

    # Generate synthetic case
    dwell_positions, target_points, oar_points = create_sample_clinical_case()
    initial_plan = BrachytherapyPlan(dwell_positions, [15.0] * len(dwell_positions),
                                     target_points, oar_points)

    # Optimize dwell times
    optimizer = BrachytherapyOptimizer(calculator, prescription_dose=600.0)
    optimized_plan = optimizer.optimize_ipsa_style(initial_plan, max_iterations=50)

    # Compute 3D dose grid
    xs, ys, zs, dose_grid = optimizer.generate_dose_heatmap(optimized_plan, calculator, resolution=25)

    # ----------------- 3D Visualization (Matplotlib) ------------------
    print("\nGenerating 3D dose distribution visualization...")
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    dose_norm = dose_grid / np.max(dose_grid)
    threshold = 0.6 * np.max(dose_norm)
    mask = dose_norm >= threshold

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Dose Distribution (Isosurface Approx.)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    ax.scatter(X[mask], Y[mask], Z[mask], c=dose_grid[mask], cmap='hot', s=15, alpha=0.4)

    catheter1 = np.array([[p.x, p.y, p.z] for p in dwell_positions if p.channel == 1])
    catheter2 = np.array([[p.x, p.y, p.z] for p in dwell_positions if p.channel == 2])
    ax.plot(catheter1[:, 0], catheter1[:, 1], catheter1[:, 2], c='cyan', lw=2, label='Catheter 1')
    ax.plot(catheter2[:, 0], catheter2[:, 1], catheter2[:, 2], c='lime', lw=2, label='Catheter 2')

    mappable = plt.cm.ScalarMappable(cmap='hot')
    mappable.set_array(dose_grid)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='Dose (cGy)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ----------------- Optional Interactive Plotly -------------------
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=dose_grid.flatten(),
            isomin=0.3 * np.max(dose_grid),
            isomax=np.max(dose_grid),
            opacity=0.1,
            surface_count=12,
            colorscale='Hot'
        ))
        fig.update_layout(
            scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)', aspectmode='cube'),
            title='Interactive 3D Dose Volume (Plotly)'
        )
        fig.show()
    except ImportError:
        print("Plotly not installed — skipping interactive plot.\nInstall via: pip install plotly")

    print("="*75)
    print("OPTIMIZATION COMPLETE")
    print("="*75)


if __name__ == "__main__":
    main()
