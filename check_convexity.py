import sympy as sp
import numpy as np

def verify_full_convexity():
    """Verificación completa de convexidad"""
    
    x, y = sp.symbols('x y')
    f = (x**2 + 1)**(y**2 + 1) + sp.exp(y) + sp.exp(-y)
    
    # 1. Calcular Hessiana completa
    H = sp.hessian(f, (x, y))
    
    print("Hessiana completa de f(x,y):")
    sp.pprint(H)
    
    # 2. Convertir a función numérica
    H_func = sp.lambdify((x, y), H, 'numpy')
    
    # 3. Evaluar en grid amplio
    x_vals = np.linspace(-3, 3, 15)
    y_vals = np.linspace(-3, 3, 15)
    
    all_convex = True
    problematic_points = []
    
    for x_val in x_vals:
        for y_val in y_vals:
            H_val = H_func(x_val, y_val)
            eigvals = np.linalg.eigvals(H_val)
            
            # Verificar semidefinida positiva
            min_eigval = np.min(eigvals)
            if min_eigval < -1e-8:
                all_convex = False
                problematic_points.append({
                    'point': (x_val, y_val),
                    'eigvals': eigvals,
                    'min_eigval': min_eigval
                })
    
    if all_convex:
        print("\n✅ La función parece convexa en la región evaluada")
    else:
        print(f"\n❌ Encontrados {len(problematic_points)} puntos no convexos")
        for p in problematic_points[:3]:  # Mostrar solo primeros 3
            print(f"  Punto {p['point']}: min autovalor = {p['min_eigval']:.2e}")
    
    return all_convex, H

# Ejecutar verificación
is_convex, H_matrix = verify_full_convexity()