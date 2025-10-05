"""
Script de prueba para subir CSV a Snowflake usando arquitectura de Workflows
Usa el archivo hour.csv de la carpeta test/
"""
import sys
import os
from pathlib import Path

# Agregar el directorio backend al path para importar los módulos
sys.path.insert(0, str(Path(__file__).parent))

from services.workflow_manager import WorkflowManager

def main():
    # Ruta al archivo CSV en la carpeta test
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'test', 'hour.csv')
    csv_path = os.path.abspath(csv_path)
    
    print("=" * 70)
    print("🧪 TEST: Subida de CSV con Arquitectura de Workflow")
    print("=" * 70)
    print(f"📂 Archivo: {csv_path}")
    print(f"📂 Existe: {os.path.exists(csv_path)}")
    
    if not os.path.exists(csv_path):
        print("❌ Error: El archivo no existe")
        return
    
    # Inicializar gestor de workflows
    print("\n🔌 Conectando a Snowflake...")
    try:
        manager = WorkflowManager()
        print("✅ Conexión exitosa")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return
    
    try:
        # Crear un workflow único con UUID
        print("\n📦 Creando workflow con schema aislado...")
        workflow = manager.create_workflow(
            workflow_name="Test Bike Sharing Analysis"
        )
        
        print(f"\n✅ Workflow creado:")
        print(f"   📋 ID: {workflow['workflow_id']}")
        print(f"   📦 Schema: {workflow['schema_name']}")
        print(f"   📝 Nombre: {workflow['workflow_name']}")
        
        # Obtener uploader para este workflow específico
        print(f"\n📤 Iniciando carga en el workflow...")
        uploader = manager.get_workflow_uploader(
            schema_name=workflow['schema_name']
        )
        
        # Subir CSV
        result = uploader.upload_csv(csv_path, 'bike_sharing_hourly')
        
        # Mostrar resultados
        print("\n" + "=" * 70)
        print("✅ CARGA COMPLETADA EXITOSAMENTE")
        print("=" * 70)
        print(f"📊 Método usado:     {result['method']}")
        print(f"📦 Tamaño archivo:   {result['file_size_mb']} MB")
        print(f"📈 Filas cargadas:   {result['rows_loaded']:,}")
        print(f"🗂️  Schema:           {workflow['schema_name']}")
        print(f"📊 Tabla creada:     {result['table_name']}")
        print(f"🔗 Ruta completa:    {workflow['schema_name']}.{result['table_name']}")
        
        # Verificar datos cargados
        print("\n" + "=" * 70)
        print("🔍 VERIFICACIÓN - Primeras 5 filas:")
        print("=" * 70)
        results = uploader.query(f"SELECT * FROM {result['table_name']} LIMIT 5")
        for i, row in enumerate(results, 1):
            print(f"\nFila {i}:")
            print(f"  Fecha: {row[1]}, Hora: {row[5]}")
            print(f"  Usuarios: Casuales={row[14]}, Registrados={row[15]}, Total={row[16]}")
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETADO")
        print("=" * 70)
        print(f"\n💡 Para eliminar este workflow de prueba, ejecuta:")
        print(f"   manager.delete_workflow(schema_name='{workflow['schema_name']}')")
        
        uploader.close()
   
    finally:
        manager.close()
        print("\n🔌 Conexión cerrada")

if __name__ == "__main__":
    main()
