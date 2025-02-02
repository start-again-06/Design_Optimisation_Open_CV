import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def get_contour_points(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, x1,x2, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    contour = max(contours, key=cv2.contourArea)
    points = contour.squeeze()  # Convert to Nx2 array
    return points

def fit_spline(points, smoothing=0.1):
    tck, u = splprep([points[:,0], points[:,1]], s=smoothing)
    return tck, u


def objective(params, tck, target_points):
    new_tck = (params.reshape(2, -1), tck[1], tck[2])
    new_curve = np.array(splev(np.linspace(0, 1, len(target_points)), new_tck)).T
    return np.sum(np.linalg.norm(new_curve - target_points, axis=1))


def optimize_contour(points):
    tck, _ = fit_spline(points)
    initial_params = np.array(tck[0]).flatten()

    res = minimize(objective, initial_params, args=(tck, points), method='BFGS')
    
    optimized_tck = (res.x.reshape(2, -1), tck[1], tck[2])
    optimized_curve = np.array(splev(np.linspace(0, 1, len(points)), optimized_tck)).T
    return optimized_curve


image_path = 'model_contour.png'  # Provide path to your image
points = get_contour_points(image_path)
optimized_curve = optimize_contour(points)


plt.figure(figsize=(6,6))
plt.plot(points[:,0], points[:,1], 'ro-', label='Original Contour')
plt.plot(optimized_curve[:,0], optimized_curve[:,1], 'b-', label='Optimized Contour')
plt.legend()
plt.show()
