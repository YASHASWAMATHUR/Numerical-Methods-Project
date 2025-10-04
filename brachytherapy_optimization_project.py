import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class DwellPosition:
    """datacontainer for dwell position in a catheter"""
    x: float  # mm units
    y: float  # mm units
    z: float  # mm units
    channel: int
    position_number: int

@dataclass
class SourceData:
    """datacontainer for TG-43 source specification data for Ir-192"""
    air_kerma_strength: float  # U (micro Gy m^2/h) units
    dose_rate_constant: float  # Lambda (cGy/h/U) units
    radial_dose_function_data: Dict  # g(r) vs r units
    anisotropy_function_data: Dict   # F(r,theta) vs r,theta units

class TG43DoseCalculator:
    """Implementation of AAPM TG-43 dose calculation formalism"""

    def __init__(self, source_data: SourceData):
        #setting up self data
        self.source_data = source_data
        self.setup_interpolators()

    def setup_interpolators(self):
        """Setup interpolation functions for TG-43 parameters"""
        r_values = np.array(list(self.source_data.radial_dose_function_data.keys()))
        g_values = np.array(list(self.source_data.radial_dose_function_data.values()))
        #setting up linear interpolator functions for self
        self.radial_dose_interp = interp1d(r_values, g_values, kind='linear', bounds_error=False, fill_value=(g_values[0], g_values[-1]))

        theta_values = np.array(list(self.source_data.anisotropy_function_data.keys()))
        f_values = np.array(list(self.source_data.anisotropy_function_data.values()))
        self.anisotropy_interp = interp1d(theta_values, f_values, kind='linear', bounds_error=False, fill_value=(f_values[0], f_values[-1]))
    
    def geometry_function(self, r: float, theta: float, L: float = 3.5) -> float:
        #Calculate geometry function G(r,theta) for line source
        #L: active length of source (mm) - typical 3.5mm for Ir-192
        #convert into cm 
        L_cm = L / 10.0
        r_cm = r / 10.0

        if abs(theta) < 1 or abs(theta - 180) < 1: 
            if r_cm > L_cm/2 :
                return 1.0 / (r_cm * r_cm - L_cm * L_cm / 4.0)
            else :
                return 1.0 / (r_cm * r_cm)
        else:
            return 1.0 / (r_cm * r_cm)

    def radial_dose_function(self, r: float) -> float:
        #to obtain radial dose function g(r) valuem for a specified r value
        r_cm = r / 10.0
        return float(self.radial_dose_interp(r_cm))

    def anisotropy_function(self, r: float, theta: float) -> float:
        #Get anisotropy function F(r,theta) value
        theta = abs(theta) % 180
        return float(self.anisotropy_interp(theta))

    def calculate_dose_rate(self, dwell_pos: DwellPosition, calc_point: Tuple[float, float, float]) -> float:
        #final calculation of dose rate at calculation point from single dwell position using TG-43 formalism function : D_rate = S_k * Lambda * G(r,theta) * g(r) * F(r,theta)

        #defining dwell position 
        dx = calc_point[0] - dwell_pos.x
        dy = calc_point[1] - dwell_pos.y
        dz = calc_point[2] - dwell_pos.z

        #defining radial value wrt current dwell position
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        if r < 2.0: #bounding the max value for the dwell position
            r = 2.0

        #calculating the theta angle, (from the given dwell position)
        if r > 0 :
            theta = np.degrees(np.arccos(abs(dz) / r)) 
        else :
            theta = 90

        #finally defining all the function constants and values at the given dwell position, using the calculated r and theta values
        S_k = self.source_data.air_kerma_strength * 1e-6  
        Lambda = self.source_data.dose_rate_constant  #from used dataset
        G = self.geometry_function(r, theta)
        g = self.radial_dose_function(r) 
        F = self.anisotropy_function(r, theta)  

        #final dose rate value
        dose_rate = S_k * Lambda * G * g * F * 1e6  

        return dose_rate

#the code above completes the discussion about the dosage calculation at one specified dwell position
#to calculate dosage at every point we develop an entire treatment plan to go at various dwell positions for given dwell times

class BrachytherapyPlan:
    #Represents the complete current brachytherapy treatment plan

    def __init__(self, dwell_positions: List[DwellPosition], dwell_times: List[float], target_points: List[Tuple[float, float, float]], oar_points: List[Tuple[float, float, float]] = None):
        self.dwell_positions = dwell_positions
        self.dwell_times = dwell_times 
        self.target_points = target_points 
        self.oar_points = oar_points or []  

    def calculate_total_dose(self, calc_point: Tuple[float, float, float], calculator: TG43DoseCalculator) -> float:
        """Calculate total dose at a point from all dwell positions"""
        total_dose = 0.0

        for i, dwell_pos in enumerate(self.dwell_positions):
            if i < len(self.dwell_times) and self.dwell_times[i] > 0:
                dose_rate = calculator.calculate_dose_rate(dwell_pos, calc_point)
                dose = dose_rate * self.dwell_times[i] / 3600.0 
                total_dose += dose

        return total_dose

class BrachytherapyOptimizer:
    #Optimization algorithms for brachytherapy treatment planning

    def __init__(self, calculator: TG43DoseCalculator, prescription_dose: float = 600.0):
        self.calculator = calculator
        self.prescription_dose = prescription_dose 

    def objective_function(self, dwell_times: np.ndarray, plan: BrachytherapyPlan) -> float:
        """
        Objective function for optimization (to be minimized)
        Implements a simplified Linear Penalty Model (LPM)
        """
        plan.dwell_times = dwell_times.tolist()

        total_penalty = 0.0

        target_weight = 10.0
        for target_point in plan.target_points:
            dose = plan.calculate_total_dose(target_point, self.calculator)
            if dose < self.prescription_dose:
                penalty = target_weight * (self.prescription_dose - dose) ** 2
                total_penalty += penalty
        oar_weight = 5.0
        oar_limit = self.prescription_dose * 0.7
        for oar_point in plan.oar_points:
            dose = plan.calculate_total_dose(oar_point, self.calculator)
            if dose > oar_limit:
                penalty = oar_weight * (dose - oar_limit) ** 2
                total_penalty += penalty

        max_dwell_time = 100.0
        regularization_weight = 0.1
        for dt in dwell_times:
            if dt > max_dwell_time:
                penalty = regularization_weight * (dt - max_dwell_time) ** 2
                total_penalty += penalty

        return total_penalty

    def optimize_ipsa_style(self, plan: BrachytherapyPlan, max_iterations: int = 100) -> BrachytherapyPlan:
        """
        Simplified IPSA-style optimization using scipy's optimization
        (Inverse Planning Simulated Annealing approach)
        """
        print(f"Starting IPSA-style optimization with {len(plan.dwell_positions)} dwell positions...")

        n_positions = len(plan.dwell_positions)
        initial_dwell_times = np.ones(n_positions) * 10.0 

        bounds = [(0.0, 100.0) for _ in range(n_positions)]

        result = minimize(
            fun=self.objective_function,
            x0=initial_dwell_times,
            args=(plan,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'disp': False}
        )

        optimized_plan = BrachytherapyPlan(
            dwell_positions=plan.dwell_positions,
            dwell_times=result.x.tolist(),
            target_points=plan.target_points,
            oar_points=plan.oar_points
        )

        print(f"Optimization completed. Final objective value: {result.fun:.2f}")
        return optimized_plan

def calculate_clinical_metrics(target_doses, oar_doses, prescription_dose=600.0):
    """Calculate standard clinical metrics for brachytherapy"""
    metrics = {}

    metrics['D90'] = np.percentile(target_doses, 10)  # Minimum dose to 90% of target
    metrics['D95'] = np.percentile(target_doses, 5)   # Minimum dose to 95% of target
    metrics['V100'] = np.sum(np.array(target_doses) >= prescription_dose) / len(target_doses) * 100
    metrics['V150'] = np.sum(np.array(target_doses) >= 1.5*prescription_dose) / len(target_doses) * 100

    # OAR metrics
    metrics['OAR_D2cc'] = np.percentile(oar_doses, 80) 
    metrics['OAR_mean'] = np.mean(oar_doses)

    return metrics

def create_sample_clinical_case():
    """Create a sample clinical case for demonstration"""
    print("Generating sample prostate HDR brachytherapy case...")

    catheter1_positions = [
        DwellPosition(x=-5, y=0, z=z, channel=1, position_number=i)
        for i, z in enumerate(range(-20, 21, 5))
    ]

    catheter2_positions = [
        DwellPosition(x=5, y=0, z=z, channel=2, position_number=i)
        for i, z in enumerate(range(-20, 21, 5))
    ]

    all_dwell_positions = catheter1_positions + catheter2_positions

    target_points = [ (0, 0, z) for z in range(-15, 16, 5) ] + [ (x, 0, 0) for x in [-10, -5, 5, 10] ]
    oar_points = [ (-15, 0, z) for z in range(-10, 11, 5) ] + [ (0, 10, z) for z in range(-10, 11, 5) ]

    return all_dwell_positions, target_points, oar_points

def main():
    """Main function to execute the brachytherapy optimization project"""

    print("=== BRACHYTHERAPY DOSE PLANNING AND OPTIMIZATION MINI PROJECT ===")

    ir192_source_data = SourceData(
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

    
    calculator = TG43DoseCalculator(ir192_source_data)
    print("✓ TG-43 dose calculator initialized")

    dwell_positions, target_points, oar_points = create_sample_clinical_case()
    print(f"✓ Clinical case created: {len(dwell_positions)} positions, {len(target_points)} targets, {len(oar_points)} OARs")

    initial_plan = BrachytherapyPlan(
        dwell_positions=dwell_positions,
        dwell_times=[15.0] * len(dwell_positions),
        target_points=target_points,
        oar_points=oar_points
    )

    optimizer = BrachytherapyOptimizer(calculator, prescription_dose=600.0)
    optimized_plan = optimizer.optimize_ipsa_style(initial_plan, max_iterations=50)

    print("\n=== RESULTS ANALYSIS ===")

    initial_target_doses = [initial_plan.calculate_total_dose(p, calculator) for p in target_points]
    initial_oar_doses = [initial_plan.calculate_total_dose(p, calculator) for p in oar_points]

    optimized_target_doses = [optimized_plan.calculate_total_dose(p, calculator) for p in target_points]  
    optimized_oar_doses = [optimized_plan.calculate_total_dose(p, calculator) for p in oar_points]

    initial_metrics = calculate_clinical_metrics(initial_target_doses, initial_oar_doses)
    optimized_metrics = calculate_clinical_metrics(optimized_target_doses, optimized_oar_doses)

    print("\nOptimization Results:")
    print(f"Target V100 coverage: {optimized_metrics['V100']:.1f}%")
    print(f"OAR dose reduction: {(initial_metrics['OAR_mean'] - optimized_metrics['OAR_mean'])/initial_metrics['OAR_mean']*100:.1f}%")
    print(f"Treatment time: {sum(optimized_plan.dwell_times):.1f} seconds")

    print("\n PROJECT COMPLETED SUCCESSFULLY! ")

    return {
        'calculator': calculator,
        'initial_plan': initial_plan,
        'optimized_plan': optimized_plan,
        'initial_metrics': initial_metrics,
        'optimized_metrics': optimized_metrics
    }

if __name__ == "__main__":
    results = main()


