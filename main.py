from init import OptimizadorFuncion

def main():
    print("ANÁLISIS DE OPTIMIZACIÓN - FUNCIÓN NO LINEAL")
    print("="*50)

    try:
        # Inicializar optimizador
        optimizador = OptimizadorFuncion('config.json')

        # Ejecutar análisis completo
        optimizador.ejecutar_optimizaciones()
        optimizador.visualizar_resultados()
        optimizador.generar_reporte()

        print("\n🎯 ANÁLISIS COMPLETADO EXITOSAMENTE")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())