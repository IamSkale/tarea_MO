import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

class OptimizadorFuncion:
    def __init__(self, config_path='config.json'):
        """
        Inicializa el optimizador cargando la configuración desde JSON
        """
        self.cargar_configuracion(config_path)
        self.resultados = {}

    def cargar_configuracion(self, config_path):
        """
        Carga la configuración desde archivo JSON
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("✓ Configuración cargada exitosamente")
        except FileNotFoundError:
            print(f"❌ Error: Archivo {config_path} no encontrado")
            raise

    def f(self, x, y):
        """
        Función objetivo: f(x,y) = x^(2*(y^2 + 1)) + e^y + e^-y
        """
        return x**(2*(y**2 + 1)) + np.exp(y) + np.exp(-y)

    def grad_f(self, x, y):
        """
        Gradiente de la función
        """
        if x <= 0:
            return np.array([np.inf, np.inf])

        exponent = 2 * (y**2 + 1)
        df_dx = exponent * x**(exponent - 1)
        df_dy = 2 * x**(2*(y**2 + 1)) * 2 * y * np.log(x) + np.exp(y) - np.exp(-y)
        return np.array([df_dx, df_dy])

    def hessian_f(self, x, y):
        """
        Hessiana de la función
        """
        if x <= 0:
            return np.array([[np.inf, np.inf], [np.inf, np.inf]])

        exponent = 2 * (y**2 + 1)
        log_x = np.log(x)

        d2f_dx2 = exponent * (exponent - 1) * x**(exponent - 2)
        d2f_dy2 = (4 * x**(exponent) * (2 * y * log_x)**2 +
                   4 * x**(exponent) * log_x +
                   np.exp(y) + np.exp(-y))
        d2f_dxdy = (4 * y * x**(exponent - 1) *
                    (exponent * log_x + 1))

        return np.array([[d2f_dx2, d2f_dxdy],
                        [d2f_dxdy, d2f_dy2]])

    def maximo_descenso(self, x0, punto_id):
        """
        Método de Máximo Descenso
        """
        params = self.config['parametros_optimizacion']
        x = np.array(x0, dtype=float)
        trajectory = [x.copy()]
        values = [self.f(x[0], x[1])]
        gradients = []

        print(f"\n--- Máximo Descenso - Punto inicial {punto_id}: {x0} ---")

        for i in range(params['max_iter_md']):
            grad = self.grad_f(x[0], x[1])
            grad_norm = np.linalg.norm(grad)
            gradients.append(grad_norm)

            if grad_norm < params['tolerancia']:
                print(f"✓ Convergencia en iteración {i}")
                break

            # Dirección de descenso
            d = -grad

            # Búsqueda lineal con verificación de dominio
            alpha_current = params['alpha_md']
            x_new = x + alpha_current * d

            # Asegurar que x > 0
            while x_new[0] <= 0.01:
                alpha_current *= 0.5
                x_new = x + alpha_current * d
                if alpha_current < 1e-10:
                    break

            x = x_new
            trajectory.append(x.copy())
            values.append(self.f(x[0], x[1]))

        resultado = {
            'trajectory': np.array(trajectory),
            'values': np.array(values),
            'gradients': np.array(gradients),
            'optimo': x,
            'valor_optimo': self.f(x[0], x[1]),
            'iteraciones': len(trajectory) - 1
        }

        return resultado

    def metodo_newton(self, x0, punto_id):
        """
        Método de Newton
        """
        params = self.config['parametros_optimizacion']
        x = np.array(x0, dtype=float)
        trajectory = [x.copy()]
        values = [self.f(x[0], x[1])]
        gradients = []

        print(f"\n--- Método de Newton - Punto inicial {punto_id}: {x0} ---")

        for i in range(params['max_iter_newton']):
            grad = self.grad_f(x[0], x[1])
            hess = self.hessian_f(x[0], x[1])
            grad_norm = np.linalg.norm(grad)
            gradients.append(grad_norm)

            if i % 10 == 0:
                print(f"Iteración {i:3d}: x={x[0]:.6f}, y={x[1]:.6f}, f={self.f(x[0],x[1]):.8f}, ||∇f||={grad_norm:.2e}")

            if grad_norm < params['tolerancia']:
                print(f"✓ Convergencia en iteración {i}")
                break

            try:
                d = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                print("Hessiano singular, usando dirección de descenso")
                d = -grad

            # Paso completo de Newton
            x_new = x + d

            # Asegurar que x > 0
            if x_new[0] <= 0.01:
                alpha = 0.5
                while alpha > 1e-10:
                    x_temp = x + alpha * d
                    if x_temp[0] > 0.01 and self.f(x_temp[0], x_temp[1]) < self.f(x[0], x[1]):
                        x_new = x_temp
                        break
                    alpha *= 0.5
                else:
                    print("No se pudo encontrar paso válido")
                    break

            x = x_new
            trajectory.append(x.copy())
            values.append(self.f(x[0], x[1]))

        resultado = {
            'trajectory': np.array(trajectory),
            'values': np.array(values),
            'gradients': np.array(gradients),
            'optimo': x,
            'valor_optimo': self.f(x[0], x[1]),
            'iteraciones': len(trajectory) - 1
        }

        return resultado

    def ejecutar_optimizaciones(self):
        """
        Ejecuta todos los métodos de optimización para todos los puntos iniciales
        """
        puntos = self.config['puntos_iniciales']
        self.resultados = {'maximo_descenso': {}, 'newton': {}}

        print("="*60)
        print("EJECUTANDO OPTIMIZACIONES")
        print("="*60)

        # Método de Máximo Descenso
        print("\n" + "MÉTODO DE MÁXIMO DESCENSO".center(60))
        for i, punto in enumerate(puntos):
            resultado = self.maximo_descenso(punto, f"P{i+1}")
            self.resultados['maximo_descenso'][f'P{i+1}'] = resultado

        # Método de Newton
        print("\n" + "MÉTODO DE NEWTON".center(60))
        for i, punto in enumerate(puntos):
            resultado = self.metodo_newton(punto, f"P{i+1}")
            self.resultados['newton'][f'P{i+1}'] = resultado

    def visualizar_resultados(self):
        """
        Genera visualizaciones de los resultados
        """
        vis_config = self.config['visualizacion']
        x_min, x_max = vis_config['rango_x']
        y_min, y_max = vis_config['rango_y']

        x = np.linspace(x_min, x_max, vis_config['puntos_grid'])
        y = np.linspace(y_min, y_max, vis_config['puntos_grid'])
        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análisis de Optimización: $f(x,y) = x^{2(y^2 + 1)} + e^y + e^{-y}$',
                    fontsize=16, fontweight='bold')

        # Gráfico 1: Superficie 3D
        ax1 = fig.add_subplot(231, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        ax1.set_title('Superficie 3D de la función')
        plt.colorbar(surf, ax=ax1)

        # Gráfico 2: Trayectorias Máximo Descenso
        ax2 = axes[0, 1]
        contour2 = ax2.contour(X, Y, Z, levels=vis_config['niveles_contorno'], cmap='viridis', alpha=0.6)
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, (key, resultado) in enumerate(self.resultados['maximo_descenso'].items()):
            traj = resultado['trajectory']
            ax2.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i],
                    markersize=3, linewidth=1, label=f'MD {key}')

        min_esperado = self.config['minimo_esperado']
        ax2.plot(min_esperado[0], min_esperado[1], 'k*', markersize=15,
                label='Mínimo exacto (1,0)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Trayectorias - Máximo Descenso')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Trayectorias Newton
        ax3 = axes[0, 2]
        contour3 = ax3.contour(X, Y, Z, levels=vis_config['niveles_contorno'], cmap='viridis', alpha=0.6)

        for i, (key, resultado) in enumerate(self.resultados['newton'].items()):
            traj = resultado['trajectory']
            ax3.plot(traj[:, 0], traj[:, 1], 's-', color=colors[i],
                    markersize=3, linewidth=1, label=f'Newton {key}')

        ax3.plot(min_esperado[0], min_esperado[1], 'k*', markersize=15,
                label='Mínimo exacto (1,0)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Trayectorias - Método de Newton')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Gráfico 4: Convergencia de valores
        ax4 = axes[1, 0]
        for i, (key, resultado) in enumerate(self.resultados['maximo_descenso'].items()):
            valores = resultado['values'] - 2  # Restar valor mínimo
            ax4.semilogy(valores, color=colors[i], linewidth=2, label=f'MD {key}')

        for i, (key, resultado) in enumerate(self.resultados['newton'].items()):
            valores = resultado['values'] - 2
            ax4.semilogy(valores, color=colors[i], linestyle='--',
                        linewidth=2, label=f'Newton {key}')

        ax4.set_xlabel('Iteración')
        ax4.set_ylabel('f(x) - f(x*) (escala log)')
        ax4.set_title('Convergencia del Valor de la Función')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Gráfico 5: Convergencia de gradientes
        ax5 = axes[1, 1]
        for i, (key, resultado) in enumerate(self.resultados['maximo_descenso'].items()):
            if len(resultado['gradients']) > 0:
                ax5.semilogy(resultado['gradients'], color=colors[i],
                           linewidth=2, label=f'MD {key}')

        for i, (key, resultado) in enumerate(self.resultados['newton'].items()):
            if len(resultado['gradients']) > 0:
                ax5.semilogy(resultado['gradients'], color=colors[i],
                           linestyle='--', linewidth=2, label=f'Newton {key}')

        ax5.set_xlabel('Iteración')
        ax5.set_ylabel('||∇f(x)|| (escala log)')
        ax5.set_title('Convergencia de la Norma del Gradiente')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Gráfico 6: Resumen de resultados
        ax6 = axes[1, 2]
        ax6.axis('off')

        # Crear tabla de resultados
        tabla_datos = []
        metodos = ['maximo_descenso', 'newton']

        for metodo in metodos:
            for key, resultado in self.resultados[metodo].items():
                x_opt = resultado['optimo']
                tabla_datos.append([
                    f"{metodo[:2].upper()} {key}",
                    f"{x_opt[0]:.6f}",
                    f"{x_opt[1]:.6f}",
                    f"{resultado['valor_optimo']:.8f}",
                    f"{resultado['iteraciones']}"
                ])

        # Agregar mínimo exacto
        tabla_datos.append([
            "Exacto", "1.000000", "0.000000",
            f"{self.f(1,0):.8f}", "-"
        ])

        tabla = ax6.table(cellText=tabla_datos,
                         colLabels=['Método', 'x*', 'y*', 'f(x*)', 'Iter'],
                         loc='center',
                         cellLoc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1, 1.5)
        ax6.set_title('Resumen de Resultados')

        plt.tight_layout()
        plt.show()

    def generar_reporte(self):
        """
        Genera un reporte detallado de los resultados
        """
        print("\n" + "="*80)
        print("REPORTE FINAL DE OPTIMIZACIÓN")
        print("="*80)

        min_esperado = self.config['minimo_esperado']
        f_min_esperado = self.f(min_esperado[0], min_esperado[1])

        print(f"Mínimo esperado: ({min_esperado[0]}, {min_esperado[1]})")
        print(f"Valor en mínimo esperado: {f_min_esperado:.8f}")
        print(f"Gradiente en mínimo esperado: {self.grad_f(min_esperado[0], min_esperado[1])}")
        print()

        # Resultados por método
        for metodo_nombre, resultados in self.resultados.items():
            print(f"\n{metodo_nombre.upper().replace('_', ' ')}:")
            print("-" * 60)
            print(f"{'Punto':<8} {'x*':<12} {'y*':<12} {'f(x*)':<15} {'Iter':<8} {'Error f':<12}")

            for key, resultado in resultados.items():
                x_opt = resultado['optimo']
                error_f = abs(resultado['valor_optimo'] - f_min_esperado)
                print(f"{key:<8} {x_opt[0]:<12.6f} {x_opt[1]:<12.6f} "
                      f"{resultado['valor_optimo']:<15.8f} {resultado['iteraciones']:<8} "
                      f"{error_f:<12.2e}")

        # Estadísticas generales
        print("\n" + "ESTADÍSTICAS GENERALES:")
        print("-" * 40)

        total_iter_md = sum(r['iteraciones'] for r in self.resultados['maximo_descenso'].values())
        total_iter_newton = sum(r['iteraciones'] for r in self.resultados['newton'].values())

        print(f"Total iteraciones Máximo Descenso: {total_iter_md}")
        print(f"Total iteraciones Newton: {total_iter_newton}")
        print(f"Promedio iteraciones por punto - MD: {total_iter_md/len(self.resultados['maximo_descenso']):.1f}")
        print(f"Promedio iteraciones por punto - Newton: {total_iter_newton/len(self.resultados['newton']):.1f}")


if __name__ == "__main__":
    optimizador = OptimizadorFuncion('config.json')

    optimizador.ejecutar_optimizaciones()

    optimizador.visualizar_resultados()

    optimizador.generar_reporte()

    # Guardar resultados en JSON
    with open('resultados_detallados.json', 'w') as f:
        # Convertir arrays de numpy a listas para JSON
        resultados_serializables = {}
        for metodo, puntos in optimizador.resultados.items():
            resultados_serializables[metodo] = {}
            for punto_id, resultado in puntos.items():
                resultados_serializables[metodo][punto_id] = {
                    'optimo': resultado['optimo'].tolist(),
                    'valor_optimo': float(resultado['valor_optimo']),
                    'iteraciones': resultado['iteraciones'],
                    'trajectory_length': len(resultado['trajectory']),
                    'gradients_length': len(resultado['gradients'])
                }

        json.dump(resultados_serializables, f, indent=2)
