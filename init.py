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
        categorias = self.config['puntos_iniciales']
        self.resultados = {'maximo_descenso': {}, 'newton': {}}
        self.puntos_por_categoria = {}

        # Crear lista plana de puntos con sus categorías
        puntos_planos = []
        for cat in categorias:
            categoria = cat['categoria']
            puntos = cat['puntos']
            self.puntos_por_categoria[categoria] = puntos
            for i, punto in enumerate(puntos):
                puntos_planos.append((categoria, i, punto))

        print("="*60)
        print("EJECUTANDO OPTIMIZACIONES")
        print("="*60)

        # Método de Máximo Descenso
        print("\n" + "MÉTODO DE MÁXIMO DESCENSO".center(60))
        for categoria, idx, punto in puntos_planos:
            punto_id = f"{categoria}_P{idx+1}"
            try:
                resultado = maximo_descenso(self, punto, punto_id)
                self.resultados['maximo_descenso'][punto_id] = resultado
            except Exception as e:
                print(f"❌ Error en Máximo Descenso {punto_id}: {e}")
                # Crear resultado por defecto para evitar errores en visualización
                self.resultados['maximo_descenso'][punto_id] = {
                    'trajectory': np.array([punto]),
                    'values': np.array([self.f(punto[0], punto[1])]),
                    'gradients': np.array([0.0]),  # Default gradient norm
                    'optimo': np.array(punto),
                    'valor_optimo': self.f(punto[0], punto[1]),
                    'iteraciones': 0
                }

        # Método de Newton
        print("\n" + "MÉTODO DE NEWTON".center(60))
        for categoria, idx, punto in puntos_planos:
            punto_id = f"{categoria}_P{idx+1}"
            try:
                resultado = metodo_newton(self, punto, punto_id)
                self.resultados['newton'][punto_id] = resultado
            except Exception as e:
                print(f"❌ Error en Newton {punto_id}: {e}")
                # Crear resultado por defecto para evitar errores en visualización
                self.resultados['newton'][punto_id] = {
                    'trajectory': np.array([punto]),
                    'values': np.array([self.f(punto[0], punto[1])]),
                    'gradients': np.array([0.0]),  # Default gradient norm
                    'optimo': np.array(punto),
                    'valor_optimo': self.f(punto[0], punto[1]),
                    'iteraciones': 0
                }

        plt.show()

    def guardar_resultados_json(self):
        """
        Genera un archivo JSON con los resultados detallados separados por categoría
        """
        min_esperado = self.config['minimo_global']['punto']
        f_min_esperado = self.config['minimo_global']['valor']

        # Estructura para el JSON
        resultados_detallados = {
            'funcion': self.config['funcion'],
            'minimo_global': self.config['minimo_global'],
            'resultados_por_categoria': {},
            'estadisticas_generales': {}
        }

        # Procesar resultados por categoría
        categorias = self.config['puntos_iniciales']
        
        for cat in categorias:
            categoria = cat['categoria']
            descripcion = cat['descripcion']
            puntos = cat['puntos']
            
            resultados_detallados['resultados_por_categoria'][categoria] = {
                'descripcion': descripcion,
                'cantidad_puntos': len(puntos),
                'puntos': [],
                'estadisticas': {
                    'maximo_descenso': {'total_iteraciones': 0, 'puntos_convergentes': 0},
                    'newton': {'total_iteraciones': 0, 'puntos_convergentes': 0}
                }
            }
            
            for i, punto in enumerate(puntos):
                punto_id = f"{categoria}_P{i+1}"
                
                # Resultados para este punto
                punto_resultado = {
                    'punto_inicial': punto,
                    'maximo_descenso': {},
                    'newton': {}
                }
                
                try:
                    valor_inicial = float(self.f(punto[0], punto[1]))
                    punto_resultado['valor_inicial'] = valor_inicial
                except (OverflowError, ValueError):
                    punto_resultado['valor_inicial'] = "overflow"
                
                # Máximo Descenso
                if punto_id in self.resultados['maximo_descenso']:
                    res_md = self.resultados['maximo_descenso'][punto_id]
                    punto_resultado['maximo_descenso'] = {
                        'optimo': res_md['optimo'].tolist(),
                        'valor_optimo': float(res_md['valor_optimo']),
                        'iteraciones': res_md['iteraciones'],
                        'error_absoluto': float(abs(res_md['valor_optimo'] - f_min_esperado)),
                        'convergente': res_md['iteraciones'] > 0 and res_md['iteraciones'] < 1000,
                        'trayectoria_length': len(res_md['trajectory'])
                    }
                    resultados_detallados['resultados_por_categoria'][categoria]['estadisticas']['maximo_descenso']['total_iteraciones'] += res_md['iteraciones']
                    if punto_resultado['maximo_descenso']['convergente']:
                        resultados_detallados['resultados_por_categoria'][categoria]['estadisticas']['maximo_descenso']['puntos_convergentes'] += 1
                else:
                    punto_resultado['maximo_descenso'] = {'error': 'No disponible'}
                
                # Newton
                if punto_id in self.resultados['newton']:
                    res_nw = self.resultados['newton'][punto_id]
                    punto_resultado['newton'] = {
                        'optimo': res_nw['optimo'].tolist(),
                        'valor_optimo': float(res_nw['valor_optimo']),
                        'iteraciones': res_nw['iteraciones'],
                        'error_absoluto': float(abs(res_nw['valor_optimo'] - f_min_esperado)),
                        'convergente': res_nw['iteraciones'] > 0 and res_nw['iteraciones'] < 100,
                        'trayectoria_length': len(res_nw['trajectory'])
                    }
                    resultados_detallados['resultados_por_categoria'][categoria]['estadisticas']['newton']['total_iteraciones'] += res_nw['iteraciones']
                    if punto_resultado['newton']['convergente']:
                        resultados_detallados['resultados_por_categoria'][categoria]['estadisticas']['newton']['puntos_convergentes'] += 1
                else:
                    punto_resultado['newton'] = {'error': 'No disponible'}
                
                resultados_detallados['resultados_por_categoria'][categoria]['puntos'].append(punto_resultado)

        # Estadísticas generales
        total_md = sum(len(cat['puntos']) for cat in categorias)
        total_nw = total_md
        
        resultados_detallados['estadisticas_generales'] = {
            'total_puntos': total_md,
            'maximo_descenso': {
                'total_iteraciones': sum(r['iteraciones'] for r in self.resultados['maximo_descenso'].values()),
                'promedio_iteraciones': sum(r['iteraciones'] for r in self.resultados['maximo_descenso'].values()) / total_md,
                'tasa_convergencia': sum(1 for r in self.resultados['maximo_descenso'].values() if r['iteraciones'] > 0 and r['iteraciones'] < 1000) / total_md
            },
            'newton': {
                'total_iteraciones': sum(r['iteraciones'] for r in self.resultados['newton'].values()),
                'promedio_iteraciones': sum(r['iteraciones'] for r in self.resultados['newton'].values()) / total_nw,
                'tasa_convergencia': sum(1 for r in self.resultados['newton'].values() if r['iteraciones'] > 0 and r['iteraciones'] < 100) / total_nw
            }
        }

        # Guardar en archivo JSON
        with open('resultados_detallados.json', 'w', encoding='utf-8') as f:
            json.dump(resultados_detallados, f, indent=2, ensure_ascii=False)
        
        print("✓ Resultados detallados guardados en 'resultados_detallados.json'")

    def generar_reporte(self):
        """
        Genera un reporte detallado de los resultados
        """
        print("\n" + "="*80)
        print("REPORTE FINAL DE OPTIMIZACIÓN")
        print("="*80)

        min_esperado = self.config['minimo_global']['punto']
        f_min_esperado = self.config['minimo_global']['valor']

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
    optimizador = OptimizadorFuncion('aux_config.json')

    optimizador.ejecutar_optimizaciones()

    optimizador.guardar_resultados_json()

    optimizador.generar_reporte()
