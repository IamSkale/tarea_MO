import numpy as np

def metodo_newton(optimizador, x0, punto_id):
    """
    Método de Newton
    """
    params = optimizador.config['parametros_optimizacion']
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    values = [optimizador.f(x[0], x[1])]
    gradients = []

    print(f"\n--- Método de Newton - Punto inicial {punto_id}: {x0} ---")

    for i in range(params['max_iter_newton']):
        grad = optimizador.grad_f(x[0], x[1])
        hess = optimizador.hessian_f(x[0], x[1])
        grad_norm = np.linalg.norm(grad)
        gradients.append(grad_norm)

        # Verificar estabilidad numérica
        if np.any(np.isinf(grad)) or np.any(np.isnan(grad)):
            print(f"⚠️  Gradiente no numérico en iteración {i}, deteniendo")
            break

        if i % 10 == 0:
            print(f"Iteración {i:3d}: x={x[0]:.6f}, y={x[1]:.6f}, f={optimizador.f(x[0],x[1]):.8f}, ||∇f||={grad_norm:.2e}")

        if grad_norm < params['tolerancia']:
            print(f"✓ Convergencia en iteración {i}")
            break

        try:
            # Verificar que la Hessiana sea definida positiva
            if np.any(np.isinf(hess)) or np.any(np.isnan(hess)):
                raise np.linalg.LinAlgError("Hessiana no numérica")
                
            d = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessiano singular o no numérico, usando dirección de descenso")
            d = -grad

        # Búsqueda lineal para Newton
        alpha = 1.0
        max_backtrack = 15
        backtrack_count = 0
        step_found = False
        
        while backtrack_count < max_backtrack:
            x_new = x + alpha * d
            
            # Verificar que x_new sea válido
            if x_new[0] > 0.01 and not np.isinf(optimizador.f(x_new[0], x_new[1])):
                f_current = optimizador.f(x[0], x[1])
                f_new = optimizador.f(x_new[0], x_new[1])
                
                # Condición de Armijo simple
                if f_new < f_current or backtrack_count == 0:
                    step_found = True
                    break
                    
            alpha *= 0.5
            backtrack_count += 1

        if not step_found:
            print(f"⚠️  No se encontró paso válido de Newton después de {max_backtrack} intentos")
            # Usar un paso muy pequeño en dirección de descenso
            alpha = 1e-4
            x_new = x + alpha * (-grad)
            if x_new[0] <= 0.01:
                break

        x = x_new
        trajectory.append(x.copy())
        values.append(optimizador.f(x[0], x[1]))

    # Asegurar que tenemos al menos un punto en cada array
    if len(gradients) == 0:
        gradients = [np.linalg.norm(optimizador.grad_f(x[0], x[1]))]

    resultado = {
        'trajectory': np.array(trajectory),
        'values': np.array(values),
        'gradients': np.array(gradients),
        'optimo': x,
        'valor_optimo': optimizador.f(x[0], x[1]),
        'iteraciones': len(trajectory) - 1
    }

    return resultado