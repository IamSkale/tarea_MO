import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from maximo_descenso import maximo_descenso
from newton import metodo_newton

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
        Función objetivo
        """
        return (x**2 + 1)**(y**2 + 1) + np.exp(y) + np.exp(-y)

    def grad_f(self, x, y):
        """
        Gradiente de la función
        """
        if x**2 + 1 <= 0:
            return np.array([np.inf, np.inf])
        
        term_base = x**2 + 1
        exponent = y**2 + 1
        log_term = np.log(term_base)
        
        df_dx = exponent * (term_base)**(exponent - 1) * 2 * x
        
        df_dy = (term_base)**exponent * log_term * 2 * y + np.exp(y) - np.exp(-y)
        
        return np.array([df_dx, df_dy])

    
    def hessian_f(self, x, y):
        """
        Hessiana de la función
        """
        
        term_base = x**2 + 1
        exponent = y**2 + 1
        log_term = np.log(term_base)
        

        term1 = (term_base)**(exponent - 1)
        term2 = 2 * x * exponent * term1
        
        d2f_dx2 = (2 * exponent * term1 + 
                2 * x * exponent * (exponent - 1) * (term_base)**(exponent - 2) * 2 * x)
        
        term3 = (term_base)**exponent * log_term
        d2f_dy2 = (2 * term3 + 
                2 * y * (term_base)**exponent * log_term * 2 * y * log_term +
                np.exp(y) + np.exp(-y))
        
        d2f_dxdy = (2 * x * 2 * y * (term_base)**(exponent - 1) +
                    2 * x * exponent * (term_base)**(exponent - 1) * log_term * 2 * y)
        
        return np.array([[d2f_dx2, d2f_dxdy],
                        [d2f_dxdy, d2f_dy2]])
    
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
            try:
                resultado = maximo_descenso(self, punto, f"P{i+1}")
                self.resultados['maximo_descenso'][f'P{i+1}'] = resultado
            except Exception as e:
                print(f"❌ Error en Máximo Descenso P{i+1}: {e}")
                # Crear resultado por defecto para evitar errores en visualización
                self.resultados['maximo_descenso'][f'P{i+1}'] = {
                    'trajectory': np.array([punto]),
                    'values': np.array([self.f(punto[0], punto[1])]),
                    'gradients': np.array([np.linalg.norm(self.grad_f(punto[0], punto[1]))]),
                    'optimo': np.array(punto),
                    'valor_optimo': self.f(punto[0], punto[1]),
                    'iteraciones': 0
                }

        # Método de Newton
        print("\n" + "MÉTODO DE NEWTON".center(60))
        for i, punto in enumerate(puntos):
            try:
                resultado = metodo_newton(self, punto, f"P{i+1}")
                self.resultados['newton'][f'P{i+1}'] = resultado
            except Exception as e:
                print(f"❌ Error en Newton P{i+1}: {e}")
                # Crear resultado por defecto para evitar errores en visualización
                self.resultados['newton'][f'P{i+1}'] = {
                    'trajectory': np.array([punto]),
                    'values': np.array([self.f(punto[0], punto[1])]),
                    'gradients': np.array([np.linalg.norm(self.grad_f(punto[0], punto[1]))]),
                    'optimo': np.array(punto),
                    'valor_optimo': self.f(punto[0], punto[1]),
                    'iteraciones': 0
                }

    def visualizar_resultados(self):
        """
        Genera visualizaciones de los resultados en 3 figuras separadas
        """
        vis_config = self.config['visualizacion']
        x_min, x_max = vis_config['rango_x']
        y_min, y_max = vis_config['rango_y']

        x = np.linspace(x_min, x_max, vis_config['puntos_grid'])
        y = np.linspace(y_min, y_max, vis_config['puntos_grid'])
        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)

        # Paleta de colores extendida para todos los puntos
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.config['puntos_iniciales'])))

        # =========================================================================
        # FIGURA 1: FUNCIÓN Y TRAYECTORIAS
        # =========================================================================
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Gráfico 1.2: Trayectorias - Máximo Descenso
        contour1 = ax1.contour(X, Y, Z, levels=vis_config['niveles_contorno'], 
                          cmap='viridis', alpha=0.6)
    
        # Crear leyenda separada para Máximo Descenso
        from matplotlib.patches import Patch
        legend_elements_md = []
        
        for i, (key, resultado) in enumerate(self.resultados['maximo_descenso'].items()):
            traj = resultado['trajectory']
            if len(traj) > 1:
                ax1.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i],
                        markersize=3, linewidth=1.5, alpha=0.8)
                ax1.plot(traj[0, 0], traj[0, 1], 's', color=colors[i], 
                        markersize=6, markeredgecolor='black', markeredgewidth=0.5)
                
                # Elemento para la leyenda
                legend_elements_md.append(
                    Patch(facecolor=colors[i], label=f'Punto {key}', alpha=0.8)
                )

        min_esperado = self.config['minimo_esperado']
        ax1.plot(min_esperado[0], min_esperado[1], 'k*', markersize=12,
                label='Mínimo (1,0)', markeredgecolor='white', markeredgewidth=1)
        
        ax1.set_xlabel('x', fontweight='bold')
        ax1.set_ylabel('y', fontweight='bold')
        ax1.set_title('1. TRAYECTORIAS - MÉTODO DE MÁXIMO DESCENSO', fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        
        # Leyenda separada a la derecha
        legend1 = ax1.legend(handles=legend_elements_md, 
                            loc='center left', 
                            bbox_to_anchor=(1.02, 0.5),
                            title="Puntos MD",
                            frameon=True,
                            fancybox=True,
                            shadow=True,
                            ncol=1)
        ax1.add_artist(legend1)

        # Gráfico 1.2: Trayectorias - Newton
        contour2 = ax2.contour(X, Y, Z, levels=vis_config['niveles_contorno'], 
                            cmap='viridis', alpha=0.6)

        # Crear leyenda separada para Newton
        legend_elements_nw = []
        
        for i, (key, resultado) in enumerate(self.resultados['newton'].items()):
            traj = resultado['trajectory']
            if len(traj) > 1:
                ax2.plot(traj[:, 0], traj[:, 1], 's-', color=colors[i],
                        markersize=3, linewidth=1.5, alpha=0.8)
                ax2.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], 
                        markersize=6, markeredgecolor='black', markeredgewidth=0.5)
                
                # Elemento para la leyenda
                legend_elements_nw.append(
                    Patch(facecolor=colors[i], label=f'Punto {key}', alpha=0.8)
                )

        ax2.plot(min_esperado[0], min_esperado[1], 'k*', markersize=12,
                label='Mínimo (1,0)', markeredgecolor='white', markeredgewidth=1)
        
        ax2.set_xlabel('x', fontweight='bold')
        ax2.set_ylabel('y', fontweight='bold')
        ax2.set_title('2. TRAYECTORIAS - MÉTODO DE NEWTON', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        
        # Leyenda separada a la derecha
        legend2 = ax2.legend(handles=legend_elements_nw, 
                            loc='center left', 
                            bbox_to_anchor=(1.02, 0.5),
                            title="Puntos Newton",
                            frameon=True,
                            fancybox=True,
                            shadow=True,
                            ncol=1)
        ax2.add_artist(legend2)

        # Ajustar layout para dar espacio a las leyendas
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, right=0.85)  # Más espacio a la derecha
        plt.show()

        # =========================================================================
        # FIGURA 2: CONVERGENCIAS
        # =========================================================================
        fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(14, 12))

        # Gráfico 2.1: Convergencia de valores de la función (AGRUPTADO)
        # Máximo Descenso - una línea con promedio o representativa
        valores_md = []
        labels_md = []
        for i, (key, resultado) in enumerate(self.resultados['maximo_descenso'].items()):
            if len(resultado['values']) > 1:
                valores = resultado['values'] - 2
                # Usar colores más distintos para métodos
                ax4.semilogy(valores, color='blue', linewidth=1.5, alpha=0.4)
                valores_md.append(valores)
                labels_md.append(f'MD {key}')

        # Newton - una línea con promedio o representativa
        valores_nw = []
        labels_nw = []
        for i, (key, resultado) in enumerate(self.resultados['newton'].items()):
            if len(resultado['values']) > 1:
                valores = resultado['values'] - 2
                ax4.semilogy(valores, color='red', linewidth=1.5, alpha=0.4, linestyle='--')
                valores_nw.append(valores)
                labels_nw.append(f'NW {key}')

        # Líneas representativas
        if valores_md:
            # Encontrar la trayectoria más representativa (la del medio)
            longitudes = [len(v) for v in valores_md]
            idx_representativo = longitudes.index(sorted(longitudes)[len(longitudes)//2])
            ax4.semilogy(valores_md[idx_representativo], color='darkblue', 
                        linewidth=3, label='Máximo Descenso (representativo)', alpha=0.8)

        if valores_nw:
            longitudes = [len(v) for v in valores_nw]
            idx_representativo = longitudes.index(sorted(longitudes)[len(longitudes)//2])
            ax4.semilogy(valores_nw[idx_representativo], color='darkred', 
                        linewidth=3, label='Newton (representativo)', linestyle='--', alpha=0.8)

        ax4.set_xlabel('Iteración', fontweight='bold')
        ax4.set_ylabel('f(x) - f(x*) (escala log)', fontweight='bold')
        ax4.set_title('4. CONVERGENCIA DEL VALOR DE LA FUNCIÓN', fontweight='bold', pad=15)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Gráfico 2.2: Convergencia de gradientes (AGRUPTADO)
        # Máximo Descenso
        for i, (key, resultado) in enumerate(self.resultados['maximo_descenso'].items()):
            if len(resultado['gradients']) > 1:
                ax5.semilogy(resultado['gradients'], color='blue', linewidth=1.5, alpha=0.4)

        # Newton
        for i, (key, resultado) in enumerate(self.resultados['newton'].items()):
            if len(resultado['gradients']) > 1:
                ax5.semilogy(resultado['gradients'], color='red', linewidth=1.5, alpha=0.4, linestyle='--')

        # Líneas representativas para gradientes
        if valores_md:
            gradientes_md = []
            for key, resultado in self.resultados['maximo_descenso'].items():
                if len(resultado['gradients']) > 1:
                    gradientes_md.append(resultado['gradients'])
            
            if gradientes_md:
                longitudes = [len(g) for g in gradientes_md]
                idx_representativo = longitudes.index(sorted(longitudes)[len(longitudes)//2])
                ax5.semilogy(gradientes_md[idx_representativo], color='darkblue', 
                            linewidth=3, label='Máximo Descenso (representativo)', alpha=0.8)

        if valores_nw:
            gradientes_nw = []
            for key, resultado in self.resultados['newton'].items():
                if len(resultado['gradients']) > 1:
                    gradientes_nw.append(resultado['gradients'])
            
            if gradientes_nw:
                longitudes = [len(g) for g in gradientes_nw]
                idx_representativo = longitudes.index(sorted(longitudes)[len(longitudes)//2])
                ax5.semilogy(gradientes_nw[idx_representativo], color='darkred', 
                            linewidth=3, label='Newton (representativo)', linestyle='--', alpha=0.8)

        ax5.set_xlabel('Iteración', fontweight='bold')
        ax5.set_ylabel('||∇f(x)|| (escala log)', fontweight='bold')
        ax5.set_title('5. CONVERGENCIA DE LA NORMA DEL GRADIENTE', fontweight='bold', pad=15)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Añadir texto informativo
        ax5.text(0.02, 0.02, f'Total: {len(labels_md)} MD + {len(labels_nw)} Newton\nLíneas claras: todas las trayectorias\nLíneas oscuras: trayectorias representativas', 
                transform=ax5.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()

        # =========================================================================
        # FIGURA 3: RESUMEN DE RESULTADOS
        # =========================================================================
        fig3, ax6 = plt.subplots(1, 1, figsize=(16, 10))
        fig3.suptitle('RESUMEN DE RESULTADOS - COMPARACIÓN DE MÉTODOS', 
                    fontsize=16, fontweight='bold', y=0.95)

        ax6.axis('off')

        # Crear tabla de resultados detallada
        tabla_datos = []
        
        # Encabezados de la tabla
        tabla_datos.append(['Método', 'Punto', 'x₀', 'y₀', 'x*', 'y*', 'f(x*)', 'Iter', '||∇f(x*)||', 'Error f(x*)'])

        f_min_esperado = self.f(1, 0)
        
        for metodo in ['maximo_descenso', 'newton']:
            metodo_corto = 'MD' if metodo == 'maximo_descenso' else 'NW'
            for key, resultado in self.resultados[metodo].items():
                x_opt = resultado['optimo']
                punto_idx = int(key[1:]) - 1
                x0, y0 = self.config['puntos_iniciales'][punto_idx]
                
                grad_norm = np.linalg.norm(self.grad_f(x_opt[0], x_opt[1])) if len(resultado['gradients']) > 0 else np.nan
                error_f = abs(resultado['valor_optimo'] - f_min_esperado)
                
                tabla_datos.append([
                    metodo_corto,
                    key,
                    f"{x0:.2f}",
                    f"{y0:.2f}",
                    f"{x_opt[0]:.6f}",
                    f"{x_opt[1]:.6f}",
                    f"{resultado['valor_optimo']:.8f}",
                    f"{resultado['iteraciones']}",
                    f"{grad_norm:.2e}" if not np.isnan(grad_norm) else "N/A",
                    f"{error_f:.2e}"
                ])

        # Agregar mínimo exacto
        grad_exacto = np.linalg.norm(self.grad_f(1, 0))
        tabla_datos.append([
            "EXACTO", "-", "-", "-", "1.000000", "0.000000",
            f"{f_min_esperado:.8f}", "-", f"{grad_exacto:.2e}", "0.00e+00"
        ])

        # Crear tabla
        tabla = ax6.table(cellText=tabla_datos,
                        loc='center',
                        cellLoc='center',
                        colWidths=[0.08, 0.06, 0.07, 0.07, 0.1, 0.1, 0.12, 0.06, 0.1, 0.1])
        
        # Formatear tabla
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(8)
        tabla.scale(1, 1.5)
        
        # Estilo para la cabecera
        for i in range(len(tabla_datos[0])):
            tabla[(0, i)].set_facecolor('#4B0082')
            tabla[(0, i)].set_text_props(weight='bold', color='white', fontsize=9)
        
        # Resaltar fila del mínimo exacto
        for i in range(len(tabla_datos[0])):
            tabla[(len(tabla_datos)-1, i)].set_facecolor('#2E8B57')
            tabla[(len(tabla_datos)-1, i)].set_text_props(weight='bold', color='white')
        
        # Colorear filas alternas para mejor lectura
        for i in range(1, len(tabla_datos)-1):
            if i % 2 == 1:
                for j in range(len(tabla_datos[0])):
                    tabla[(i, j)].set_facecolor('#f0f0f0')

        # Añadir texto informativo
        ax6.text(0.02, 0.98, 'LEYENDA:\n• MD = Máximo Descenso\n• NW = Método de Newton\n• x₀, y₀ = Punto inicial\n• x*, y* = Punto óptimo encontrado\n• Error f(x*) = |f(x*) - f(1,0)|', 
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
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
