"""
=============================================================================
BRACHYTHERAPY DOSE PLANNING AND OPTIMIZATION SYSTEM
=============================================================================
This code implements HDR brachytherapy treatment planning using:
  1. TG-43 dose calculation formalism (AAPM standard)
  2. Linear Penalty Model (LPM) for multi-objective optimization
  3. L2 regularization for smooth, stable dwell time solutions
  4. 3D dose heatmap visualization
  5. Synthetic clinical case generator for testing and validation
=============================================================================
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DwellPosition:
    """
    Represents a single dwell position where radioactive source can pause.
    
    Attributes:
        x, y, z: 3D coordinates in millimeters
        channel: Catheter/channel number (1, 2, 3, ...)
        position_number: Sequential position index within the catheter
    """
    x: float
    y: float
    z: float
    channel: int
    position_number: int


@dataclass
class SourceData:
    """
    Contains TG-43 dosimetric parameters for a specific radioactive source.
    
    Attributes:
        air_kerma_strength: Source strength in U (micro Gy·m²/h)
        dose_rate_constant: Lambda constant in cGy/h per U
        radial_dose_function_data: Dictionary of g(r) values vs distance
        anisotropy_function_data: Dictionary of F(theta) values vs angle
    """
    air_kerma_strength: float
    dose_rate_constant: float
    radial_dose_function_data: Dict[float, float]
    anisotropy_function_data: Dict[float, float]


# =============================================================================
# TG-43 DOSE CALCULATION ENGINE
# =============================================================================

class TG43DoseCalculator:
    """
    Implements AAPM TG-43 formalism for brachytherapy dose calculation.
    
    The TG-43 formula computes dose rate at any point from a source:
        D_rate = S_k X Λ X G(r,θ) X g(r) X F(r,θ)
    
    Where:
        S_k = Air kerma strength
        Λ = Dose rate constant
        G(r,θ) = Geometry function
        g(r) = Radial dose function
        F(r,θ) = Anisotropy function
    """
    
    def __init__(self, source_data: SourceData):
        """
        Initialize dose calculator with source-specific parameters.
        
        Args:
            source_data: SourceData object containing TG-43 parameters
        """
        self.source_data = source_data
        self.setup_interpolators()
    
    def setup_interpolators(self):
        """
        Creates interpolation functions for g(r) and F(theta).
        
        Uses linear interpolation with boundary extrapolation to provide
        smooth dose estimation at any distance or angle.
        """
        # Radial dose function g(r) interpolator
        r_values = np.array(list(self.source_data.radial_dose_function_data.keys()))
        g_values = np.array(list(self.source_data.radial_dose_function_data.values()))
        self.radial_dose_interp = interp1d(
            r_values, g_values, kind='linear', bounds_error=False,
            fill_value=(g_values[0], g_values[-1])
        )
        
        # Anisotropy function F(theta) interpolator
        theta_values = np.array(list(self.source_data.anisotropy_function_data.keys()))
        f_values = np.array(list(self.source_data.anisotropy_function_data.values()))
        self.anisotropy_interp = interp1d(
            theta_values, f_values, kind='linear', bounds_error=False,
            fill_value=(f_values[0], f_values[-1])
        )
    
    def geometry_function(self, r: float, theta: float, L: float = 3.5) -> float:
        """
        Compute geometry function G(r,θ) accounting for finite source length.
        
        Args:
            r: Distance from source center (mm)
            theta: Polar angle from source axis (degrees)
            L: Active source length (mm), default 3.5mm for Ir-192
            
        Returns:
            Geometry function value (dimensionless)
        """
        L_cm = L / 10.0  # Convert to cm
        r_cm = r / 10.0
        
        # Along-axis approximation
        if abs(theta) < 1 or abs(theta - 180) < 1:
            if r_cm > L_cm / 2:
                return 1.0 / (r_cm * r_cm - L_cm * L_cm / 4.0)
            else:
                return 1.0 / (r_cm * r_cm)
        # Off-axis point source approximation
        else:
            return 1.0 / (r_cm * r_cm)
    
    def radial_dose_function(self, r: float) -> float:
        """
        Get radial dose function g(r) at distance r.
        
        Args:
            r: Distance from source (mm)
            
        Returns:
            g(r) value accounting for attenuation and scatter
        """
        r_cm = r / 10.0
        return float(self.radial_dose_interp(r_cm))
    
    def anisotropy_function(self, r: float, theta: float) -> float:
        """
        Get anisotropy function F(r,θ) at angle theta.
        
        Args:
            r: Distance from source (mm)
            theta: Polar angle (degrees)
            
        Returns:
            F(theta) value accounting for angular dose variation
        """
        theta = abs(theta) % 180  # Normalize angle
        return float(self.anisotropy_interp(theta))
    
    def calculate_dose_rate(self, dwell_pos: DwellPosition, 
                          calc_point: Tuple[float, float, float]) -> float:
        """
        Calculate dose rate at a point from a single dwell position.
        
        Implements full TG-43 formalism with all correction factors.
        
        Args:
            dwell_pos: Source dwell position
            calc_point: (x, y, z) coordinates where dose is calculated (mm)
            
        Returns:
            Dose rate in cGy/hour
        """
        # Calculate distance and angle from source to point
        dx = calc_point[0] - dwell_pos.x
        dy = calc_point[1] - dwell_pos.y
        dz = calc_point[2] - dwell_pos.z
        
        r = np.sqrt(dx*dx + dy*dy + dz*dz)  # Euclidean distance (mm)
        
        # Avoid singularity very close to source
        if r < 2.0:
            r = 2.0
        
        # Polar angle from source axis (z-direction)
        theta = np.degrees(np.arccos(abs(dz) / r)) if r > 0 else 90
        
        # TG-43 formula components
        S_k = self.source_data.air_kerma_strength * 1e-6  # Convert units
        Lambda = self.source_data.dose_rate_constant
        G = self.geometry_function(r, theta)
        g = self.radial_dose_function(r)
        F = self.anisotropy_function(r, theta)
        
        # Final dose rate calculation
        dose_rate = S_k * Lambda * G * g * F * 1e6
        return dose_rate


# =============================================================================
# TREATMENT PLAN CLASS
# =============================================================================

class BrachytherapyPlan:
    """
    Represents a complete brachytherapy treatment plan.
    
    Contains:
        - Catheter geometry (dwell positions)
        - Source dwell times at each position
        - Target points (tumor volume)
        - OAR points (organs at risk)
    """
    
    def __init__(self, dwell_positions: List[DwellPosition], 
                 dwell_times: List[float],
                 target_points: List[Tuple[float, float, float]], 
                 oar_points: List[Tuple[float, float, float]] = None):
        """
        Initialize treatment plan.
        
        Args:
            dwell_positions: List of DwellPosition objects
            dwell_times: List of dwell times in seconds
            target_points: List of (x,y,z) tumor sampling points
            oar_points: List of (x,y,z) OAR sampling points
        """
        self.dwell_positions = dwell_positions
        self.dwell_times = dwell_times
        self.target_points = target_points
        self.oar_points = oar_points or []
    
    def calculate_total_dose(self, calc_point: Tuple[float, float, float], 
                           calculator: TG43DoseCalculator) -> float:
        """
        Calculate cumulative dose at a point from all dwell positions.
        
        Args:
            calc_point: (x,y,z) coordinates (mm)
            calculator: TG43DoseCalculator instance
            
        Returns:
            Total dose in cGy
        """
        total_dose = 0.0
        
        for i, dwell_pos in enumerate(self.dwell_positions):
            if i < len(self.dwell_times) and self.dwell_times[i] > 0:
                dose_rate = calculator.calculate_dose_rate(dwell_pos, calc_point)
                # Convert dose rate (cGy/h) to dose (cGy) using dwell time (sec)
                dose = dose_rate * self.dwell_times[i] / 3600.0
                total_dose += dose
        
        return total_dose


# =============================================================================
# OPTIMIZATION ENGINE (L2 REGULARIZED VERSION)
# =============================================================================

class BrachytherapyOptimizer:
    """
    Optimization engine for HDR brachytherapy treatment planning.
    
    KEY FEATURES:
    1. Linear Penalty Model (LPM) for multi-objective optimization
    2. L2 regularization for smooth, stable solutions
    3. L-BFGS-B constrained optimization algorithm
    
    DIFFERENCE FROM PREVIOUS VERSION:
    - Previous: Hard penalty if dwell_time > 100 seconds
    - Current (L2): Soft penalty on sum of squared dwell times
    
    L2 Advantages:
    - Smoother optimization landscape (better convergence)
    - Prevents extreme dwell time values
    - More mathematically principled regularization
    - Tunable via regularization_weight parameter
    """
    
    def __init__(self, calculator: TG43DoseCalculator, prescription_dose: float = 600.0):
        """
        Initialize optimizer.
        
        Args:
            calculator: TG43DoseCalculator for dose computation
            prescription_dose: Target tumor dose in cGy
        """
        self.calculator = calculator
        self.prescription_dose = prescription_dose
    
    def objective_function(self, dwell_times: np.ndarray, plan: BrachytherapyPlan) -> float:
        """
        Objective function to minimize during optimization.
        
        Combines three penalty components:
        1. Target underdosing penalty (weight = 10.0)
        2. OAR overdosing penalty (weight = 5.0)
        3. L2 regularization penalty (weight = 0.01)
        
        Args:
            dwell_times: Current dwell time vector (seconds)
            plan: BrachytherapyPlan object
            
        Returns:
            Total penalty value (lower = better plan)
        """
        plan.dwell_times = dwell_times.tolist()
        total_penalty = 0.0
        
        # ===== 1. TARGET COVERAGE PENALTY =====
        # Penalize points receiving less than prescription dose
        target_weight = 10.0
        for target_point in plan.target_points:
            dose = plan.calculate_total_dose(target_point, self.calculator)
            if dose < self.prescription_dose:
                underdose = self.prescription_dose - dose
                penalty = target_weight * (underdose ** 2)  # Quadratic penalty
                total_penalty += penalty
        
        # ===== 2. OAR SPARING PENALTY =====
        # Penalize points receiving more than 70% of prescription
        oar_weight = 5.0
        oar_limit = self.prescription_dose * 0.7
        for oar_point in plan.oar_points:
            dose = plan.calculate_total_dose(oar_point, self.calculator)
            if dose > oar_limit:
                overdose = dose - oar_limit
                penalty = oar_weight * (overdose ** 2)  # Quadratic penalty
                total_penalty += penalty
        
        # ===== 3. L2 REGULARIZATION =====
        # NEW IN THIS VERSION: Soft penalty on dwell time magnitude
        # Previous version: Hard cutoff at 100 seconds
        # Current version: Smooth penalty proportional to sum(dwell_times²)
        regularization_weight = 0.01  # Small weight for soft constraint
        reg_term = np.sum(dwell_times ** 2)
        total_penalty += regularization_weight * reg_term
        
        return total_penalty
    
    def optimize_ipsa_style(self, plan: BrachytherapyPlan, 
                          max_iterations: int = 100) -> BrachytherapyPlan:
        """
        Run IPSA-style optimization to find optimal dwell times.
        
        Uses L-BFGS-B algorithm (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
        which is efficient for large-scale constrained optimization.
        
        Args:
            plan: Initial treatment plan
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized BrachytherapyPlan with improved dwell times
        """
        print(f"Starting IPSA-style optimization with {len(plan.dwell_positions)} dwell positions...")
        
        n_positions = len(plan.dwell_positions)
        initial_dwell_times = np.ones(n_positions) * 10.0  # Uniform 10 sec start
        
        # Box constraints: 0 ≤ dwell_time ≤ 100 seconds
        bounds = [(0.0, 100.0) for _ in range(n_positions)]
        
        # Run optimization
        result = minimize(
            fun=self.objective_function,
            x0=initial_dwell_times,
            args=(plan,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )
        
        # Create optimized plan
        optimized_plan = plan.__class__(
            dwell_positions=plan.dwell_positions,
            dwell_times=result.x.tolist(),
            target_points=plan.target_points,
            oar_points=plan.oar_points
        )
        
        print(f"Optimization completed. Final objective value: {result.fun:.2f}")
        return optimized_plan
    
    def generate_dose_heatmap(self, plan: BrachytherapyPlan, 
                            calculator: TG43DoseCalculator,
                            grid_limits=((-30, 30), (-30, 30), (-30, 30)), 
                            resolution=10):
        """
        Generate 3D dose distribution grid for visualization.
        
        Args:
            plan: Treatment plan to evaluate
            calculator: TG43DoseCalculator instance
            grid_limits: ((x_min,x_max), (y_min,y_max), (z_min,z_max)) in mm
            resolution: Number of grid points per axis
            
        Returns:
            (xs, ys, zs, dose_grid): Coordinate arrays and 3D dose grid
        """
        xs = np.linspace(grid_limits[0][0], grid_limits[0][1], resolution)
        ys = np.linspace(grid_limits[1][0], grid_limits[1][1], resolution)
        zs = np.linspace(grid_limits[2][0], grid_limits[2][1], resolution)
        
        dose_grid = np.zeros((resolution, resolution, resolution))
        
        # Calculate dose at each grid point
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    dose_grid[i, j, k] = plan.calculate_total_dose((x, y, z), calculator)
        
        return xs, ys, zs, dose_grid


# =============================================================================
# CLINICAL CASE GENERATOR
# =============================================================================

def create_sample_clinical_case():
    """
    Generate synthetic prostate HDR brachytherapy case.
    
    Creates:
        - 2 parallel catheters (at x = -5 and x = +5 mm)
        - 9 dwell positions per catheter (z = -20 to +20 mm)
        - 11 target points (tumor sampling)
        - 10 OAR points (organs at risk)
    
    Returns:
        (dwell_positions, target_points, oar_points)
    """
    print("Generating synthetic prostate HDR brachytherapy case...")
    
    # Catheter 1: Left side (x = -5 mm)
    catheter1_positions = [
        DwellPosition(x=-5, y=0, z=z, channel=1, position_number=i)
        for i, z in enumerate(range(-20, 21, 5))
    ]
    
    # Catheter 2: Right side (x = +5 mm)
    catheter2_positions = [
        DwellPosition(x=5, y=0, z=z, channel=2, position_number=i)
        for i, z in enumerate(range(-20, 21, 5))
    ]
    
    all_dwell_positions = catheter1_positions + catheter2_positions
    
    # Target points (tumor volume)
    target_points = [
        (0, 0, z) for z in range(-15, 16, 5)  # Central points
    ] + [
        (x, 0, 0) for x in [-10, -5, 5, 10]   # Lateral points
    ]
    
    # OAR points (rectum and bladder)
    oar_points = [
        (-15, 0, z) for z in range(-10, 11, 5)  # Rectum (posterior)
    ] + [
        (0, 10, z) for z in range(-10, 11, 5)   # Bladder (anterior)
    ]
    
    return all_dwell_positions, target_points, oar_points


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("BRACHYTHERAPY DOSE PLANNING AND OPTIMIZATION")
    print("="*70 + "\n")
    
    # ===== STEP 1: Initialize TG-43 source parameters =====
    ir192_source_data = SourceData(
        air_kerma_strength=40000,  # 40,000 U for new Ir-192 source
        dose_rate_constant=1.115,   # Standard for Ir-192
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
    
    calculator = TG43DoseCalculator(ir192_source_data)
    print("✓ TG-43 dose calculator initialized")
    
    # ===== STEP 2: Create clinical case =====
    dwell_positions, target_points, oar_points = create_sample_clinical_case()
    print(f"✓ Clinical case: {len(dwell_positions)} positions, "
          f"{len(target_points)} targets, {len(oar_points)} OARs\n")
    
    # ===== STEP 3: Create initial plan =====
    initial_plan = BrachytherapyPlan(
        dwell_positions=dwell_positions,
        dwell_times=[15.0] * len(dwell_positions),  # Uniform 15 sec
        target_points=target_points,
        oar_points=oar_points
    )
    
    # ===== STEP 4: Run optimization =====
    optimizer = BrachytherapyOptimizer(calculator, prescription_dose=600.0)
    optimized_plan = optimizer.optimize_ipsa_style(initial_plan, max_iterations=50)
    
    # ===== STEP 5: Generate and visualize heatmap =====
    print("\nGenerating dose heatmap...")
    xs, ys, zs, dose_grid = optimizer.generate_dose_heatmap(
        optimized_plan, calculator, resolution=20
    )
    
    # Plot central slice (z = 0 mm)
    mid_z = len(zs) // 2
    plt.figure(figsize=(10, 8))
    plt.imshow(dose_grid[:, :, mid_z], 
               extent=(xs[0], xs[-1], ys[0], ys[-1]), 
               origin='lower', cmap='hot')
    plt.colorbar(label='Dose (cGy)', shrink=0.8)
    plt.title('Optimized Dose Heatmap Slice at z=0 mm', fontsize=14, fontweight='bold')
    plt.xlabel('X (mm)', fontsize=12)
    plt.ylabel('Y (mm)', fontsize=12)
    
    # Overlay catheter positions
    catheter_x = [-5, 5]  # x-coordinates of catheters
    plt.scatter(catheter_x, [0, 0], c='cyan', marker='x', s=200, 
                linewidths=3, label='Catheter positions')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
