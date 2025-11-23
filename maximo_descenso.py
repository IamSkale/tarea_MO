import numpy as np

def maximo_descenso(optimizador, x0, punto_id):
    """
    Método de Máximo Descenso
    """
    params = optimizador.config['parametros_optimizacion']
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    values = [optimizador.f(x[0], x[1])]
    gradients = []

    print(f"\n--- Máximo Descenso - Punto inicial {punto_id}: {x0} ---")

    for i in range(params['max_iter_md']):
        grad = optimizador.grad_f(x[0], x[1])
        grad_norm = np.linalg.norm(grad)
        gradients.append(grad_norm)

        # Verificar si el gradiente es numéricamente estable
        if np.any(np.isinf(grad)) or np.any(np.isnan(grad)):
            print(f"⚠️  Gradiente no numérico en iteración {i}, deteniendo")
            break

        if grad_norm < params['tolerancia']:
            print(f"✓ Convergencia en iteración {i}")
            break

        # Dirección de descenso
        d = -grad

        # Búsqueda lineal con verificación de dominio
        alpha_current = params['alpha_md']
        max_backtrack = 20
        backtrack_count = 0
        
        while backtrack_count < max_backtrack:
            x_new = x + alpha_current * d
            
            # Verificar que x_new sea válido
            if x_new[0] > 0.01 and not np.isinf(optimizador.f(x_new[0], x_new[1])):
                # Verificar descenso suficiente
                f_current = optimizador.f(x[0], x[1])
                f_new = optimizador.f(x_new[0], x_new[1])
                
                if f_new < f_current or backtrack_count == 0:
                    break
                    
            alpha_current *= 0.5
            backtrack_count += 1
        else:
            print(f"⚠️  No se encontró paso válido después de {max_backtrack} intentos")
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