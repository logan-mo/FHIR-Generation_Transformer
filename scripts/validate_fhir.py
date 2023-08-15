import json
from jsonschema import validate, ValidationError
def validate_fhir_resource_with_schema_loading(resource, schema_path="/mnt/data/fhir.schema.json"):

    # Load the FHIR schema
    with open(schema_path, "r") as file:
        fhir_schema = json.load(file)

    try:
        validate(instance=resource, schema=fhir_schema)
        return "Resource is valid."
    except ValidationError as e:
        return f"Resource is invalid. Reason: {e.message}"