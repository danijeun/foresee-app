"""
Gestor de Workflows para Foresee App
Maneja la creación y gestión de schemas por workflow
"""
import uuid
from datetime import datetime
from .snowflake_ingestion import SnowflakeCSVUploader

class WorkflowManager:
    """
    Gestiona workflows independientes en Snowflake.
    Cada workflow tiene su propio schema con aislamiento completo.
    """
    
    def __init__(self, database=None):
        """
        Inicializa el gestor de workflows
        
        Args:
            database: Nombre de la base de datos (default: usa variable de entorno)
        """
        self.database = database
        # Conexión temporal para operaciones de gestión
        self.uploader = SnowflakeCSVUploader(database=database, schema='PUBLIC')
        
    def create_workflow(self, workflow_name=None):
        """
        Crea un nuevo workflow con su propio schema
        
        Args:
            workflow_name: Nombre descriptivo del workflow (opcional)
            
        Returns:
            dict: Información del workflow creado
        """
        # Generar UUID único para el workflow
        workflow_id = str(uuid.uuid4()).replace('-', '_')
        schema_name = f"WORKFLOW_{workflow_id}"
        
        print(f"🔨 Creando nuevo workflow: {schema_name}")
        
        # Crear el schema del workflow
        self.uploader.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        print(f"✓ Schema '{schema_name}' creado")
        
        # Crear tablas estándar dentro del schema
        self._create_workflow_tables(schema_name)
        
        workflow_info = {
            'workflow_id': workflow_id,
            'schema_name': schema_name,
            'workflow_name': workflow_name or 'Unnamed Workflow',
            'created_at': datetime.now().isoformat()
        }
        
        print(f"✅ Workflow creado exitosamente")
        return workflow_info
    
    def _create_workflow_tables(self, schema_name):
        """
        Crea las tablas estándar dentro del schema del workflow
        """
        # Tabla para metadata del workflow
        metadata_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.workflow_metadata (
            key VARCHAR(255),
            value VARIANT,
            updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        self.uploader.cursor.execute(metadata_sql)
        print(f"  ✓ Tabla 'workflow_metadata' creada")
        
    def get_workflow_uploader(self, workflow_id=None, schema_name=None):
        """
        Obtiene un uploader configurado para un workflow específico
        
        Args:
            workflow_id: ID del workflow
            schema_name: Nombre del schema (alternativo a workflow_id)
            
        Returns:
            SnowflakeCSVUploader: Uploader configurado para el workflow
        """
        if schema_name is None and workflow_id:
            schema_name = f"WORKFLOW_{workflow_id}"
        
        if schema_name is None:
            raise ValueError("Debe proporcionar workflow_id o schema_name")
        
        return SnowflakeCSVUploader(database=self.database, schema=schema_name)
    
    def delete_workflow(self, workflow_id=None, schema_name=None):
        """
        Elimina un workflow completo (CUIDADO: Operación destructiva)
        
        Args:
            workflow_id: ID del workflow
            schema_name: Nombre del schema (alternativo a workflow_id)
        """
        if schema_name is None and workflow_id:
            schema_name = f"WORKFLOW_{workflow_id}"
        
        if schema_name is None:
            raise ValueError("Debe proporcionar workflow_id o schema_name")
        
        print(f"🗑️ Eliminando workflow: {schema_name}")
        
        # Eliminar schema completo con todas sus tablas
        self.uploader.cursor.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        print(f"✓ Workflow eliminado")
    
    def list_workflows(self):
        """
        Lista todos los workflows activos consultando los schemas de Snowflake
        
        Returns:
            list: Lista de workflows
        """
        try:
            self.uploader.cursor.execute("""
                SHOW SCHEMAS LIKE 'WORKFLOW_%'
            """)
            results = self.uploader.cursor.fetchall()
            
            workflows = []
            for row in results:
                schema_name = row[1]  # El nombre del schema está en la columna 1
                # Extraer workflow_id del nombre del schema
                workflow_id = schema_name.replace('WORKFLOW_', '')
                workflows.append({
                    'workflow_id': workflow_id,
                    'schema_name': schema_name,
                    'created_on': row[0]  # Fecha de creación
                })
            return workflows
        except Exception as e:
            print(f"⚠️ No se pudo listar workflows: {e}")
            return []
    
    def close(self):
        """
        Cierra la conexión
        """
        self.uploader.close()


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    # Crear gestor de workflows
    manager = WorkflowManager()
    
    try:
        # 1. Crear un nuevo workflow
        workflow = manager.create_workflow(workflow_name="Análisis de Ventas Q1")
        print(f"\n📋 Workflow Info:")
        print(f"   ID: {workflow['workflow_id']}")
        print(f"   Schema: {workflow['schema_name']}")
        
        # 2. Obtener uploader para ese workflow
        uploader = manager.get_workflow_uploader(schema_name=workflow['schema_name'])
        
        # 3. Subir datos al workflow
        result = uploader.upload_csv('datos.csv', 'raw_data')
        print(f"\n✅ Datos subidos: {result['rows_loaded']:,} filas")
        
        # 4. Listar todos los workflows
        print("\n📊 Workflows activos:")
        workflows = manager.list_workflows()
        for wf in workflows:
            print(f"   - {wf['schema_name']} (ID: {wf['workflow_id']})")
        
        uploader.close()
        
        # 5. Eliminar workflow (opcional)
        # manager.delete_workflow(schema_name=workflow['schema_name'])
        
    finally:
        manager.close()
