from converter import Converter

openAPI_json = {
    "swagger": "2.0",
    "info": {
        "title": "Simple API overview",
        "version": "v2"
    },
    "paths": {
        "/": {
            "get": {
                "operationId": "listVersionsv2",
                "summary": "List API versions",
                "produces": ["application/json"],
                "responses": {
                    "200": {
                        "description": "200 300 response",
                        "examples": {
                            "application/json":
                            "{\n    \"versions\": [\n        {\n            \"status\": \"CURRENT\",\n            \"updated\": \"2011-01-21T11:33:21Z\",\n            \"id\": \"v2.0\",\n            \"links\": [\n                {\n                    \"href\": \"http://127.0.0.1:8774/v2/\",\n                    \"rel\": \"self\"\n                }\n            ]\n        },\n        {\n            \"status\": \"EXPERIMENTAL\",\n            \"updated\": \"2013-07-23T11:33:21Z\",\n            \"id\": \"v3.0\",\n            \"links\": [\n                {\n                    \"href\": \"http://127.0.0.1:8774/v3/\",\n                    \"rel\": \"self\"\n                }\n            ]\n        }\n    ]\n}"
                        }
                    },
                    "300": {
                        "description": "200 300 response",
                        "examples": {
                            "application/json":
                            "{\n    \"versions\": [\n        {\n            \"status\": \"CURRENT\",\n            \"updated\": \"2011-01-21T11:33:21Z\",\n            \"id\": \"v2.0\",\n            \"links\": [\n                {\n                    \"href\": \"http://127.0.0.1:8774/v2/\",\n                    \"rel\": \"self\"\n                }\n            ]\n        },\n        {\n            \"status\": \"EXPERIMENTAL\",\n            \"updated\": \"2013-07-23T11:33:21Z\",\n            \"id\": \"v3.0\",\n            \"links\": [\n                {\n                    \"href\": \"http://127.0.0.1:8774/v3/\",\n                    \"rel\": \"self\"\n                }\n            ]\n        }\n    ]\n}"
                        }
                    }
                }
            }
        },
        "/v2": {
            "get": {
                "operationId": "getVersionDetailsv2",
                "summary": "Show API version details",
                "produces": ["application/json"],
                "responses": {
                    "200": {
                        "description": "200 203 response",
                        "examples": {
                            "application/json":
                            "{\n    \"version\": {\n        \"status\": \"CURRENT\",\n        \"updated\": \"2011-01-21T11:33:21Z\",\n        \"media-types\": [\n            {\n                \"base\": \"application/xml\",\n                \"type\": \"application/vnd.openstack.compute+xml;version=2\"\n            },\n            {\n                \"base\": \"application/json\",\n                \"type\": \"application/vnd.openstack.compute+json;version=2\"\n            }\n        ],\n        \"id\": \"v2.0\",\n        \"links\": [\n            {\n                \"href\": \"http://127.0.0.1:8774/v2/\",\n                \"rel\": \"self\"\n            },\n            {\n                \"href\": \"http://docs.openstack.org/api/openstack-compute/2/os-compute-devguide-2.pdf\",\n                \"type\": \"application/pdf\",\n                \"rel\": \"describedby\"\n            },\n            {\n                \"href\": \"http://docs.openstack.org/api/openstack-compute/2/wadl/os-compute-2.wadl\",\n                \"type\": \"application/vnd.sun.wadl+xml\",\n                \"rel\": \"describedby\"\n            },\n            {\n              \"href\": \"http://docs.openstack.org/api/openstack-compute/2/wadl/os-compute-2.wadl\",\n              \"type\": \"application/vnd.sun.wadl+xml\",\n              \"rel\": \"describedby\"\n            }\n        ]\n    }\n}"
                        }
                    },
                    "203": {
                        "description": "200 203 response",
                        "examples": {
                            "application/json":
                            "{\n    \"version\": {\n        \"status\": \"CURRENT\",\n        \"updated\": \"2011-01-21T11:33:21Z\",\n        \"media-types\": [\n            {\n                \"base\": \"application/xml\",\n                \"type\": \"application/vnd.openstack.compute+xml;version=2\"\n            },\n            {\n                \"base\": \"application/json\",\n                \"type\": \"application/vnd.openstack.compute+json;version=2\"\n            }\n        ],\n        \"id\": \"v2.0\",\n        \"links\": [\n            {\n                \"href\": \"http://23.253.228.211:8774/v2/\",\n                \"rel\": \"self\"\n            },\n            {\n                \"href\": \"http://docs.openstack.org/api/openstack-compute/2/os-compute-devguide-2.pdf\",\n                \"type\": \"application/pdf\",\n                \"rel\": \"describedby\"\n            },\n            {\n                \"href\": \"http://docs.openstack.org/api/openstack-compute/2/wadl/os-compute-2.wadl\",\n                \"type\": \"application/vnd.sun.wadl+xml\",\n                \"rel\": \"describedby\"\n            }\n        ]\n    }\n}"
                        }
                    }
                }
            }
        }
    },
    "consumes": ["application/json"]
}

openAPI_json_1 = {
    "swagger": "2.0",
    "info": {
        "version": "1.0.0",
        "title": "Swagger Petstore",
        "description":
        "A sample API that uses a petstore as an example to demonstrate features in the swagger-2.0 specification",
        "termsOfService": "http://swagger.io/terms/",
        "contact": {
            "name": "Swagger API Team",
            "email": "apiteam@swagger.io",
            "url": "http://swagger.io"
        },
        "license": {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
        }
    },
    "host": "petstore.swagger.io",
    "basePath": "/api",
    "schemes": ["http"],
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "paths": {
        "/pets": {
            "get": {
                "description":
                "Returns all pets from the system that the user has access to\nNam sed condimentum est. Maecenas tempor sagittis sapien, nec rhoncus sem sagittis sit amet. Aenean at gravida augue, ac iaculis sem. Curabitur odio lorem, ornare eget elementum nec, cursus id lectus. Duis mi turpis, pulvinar ac eros ac, tincidunt varius justo. In hac habitasse platea dictumst. Integer at adipiscing ante, a sagittis ligula. Aenean pharetra tempor ante molestie imperdiet. Vivamus id aliquam diam. Cras quis velit non tortor eleifend sagittis. Praesent at enim pharetra urna volutpat venenatis eget eget mauris. In eleifend fermentum facilisis. Praesent enim enim, gravida ac sodales sed, placerat id erat. Suspendisse lacus dolor, consectetur non augue vel, vehicula interdum libero. Morbi euismod sagittis libero sed lacinia.\n\nSed tempus felis lobortis leo pulvinar rutrum. Nam mattis velit nisl, eu condimentum ligula luctus nec. Phasellus semper velit eget aliquet faucibus. In a mattis elit. Phasellus vel urna viverra, condimentum lorem id, rhoncus nibh. Ut pellentesque posuere elementum. Sed a varius odio. Morbi rhoncus ligula libero, vel eleifend nunc tristique vitae. Fusce et sem dui. Aenean nec scelerisque tortor. Fusce malesuada accumsan magna vel tempus. Quisque mollis felis eu dolor tristique, sit amet auctor felis gravida. Sed libero lorem, molestie sed nisl in, accumsan tempor nisi. Fusce sollicitudin massa ut lacinia mattis. Sed vel eleifend lorem. Pellentesque vitae felis pretium, pulvinar elit eu, euismod sapien.\n",
                "operationId":
                "findPets",
                "parameters": [{
                    "name": "tags",
                    "in": "query",
                    "description": "tags to filter by",
                    "required": False,
                    "type": "array",
                    "collectionFormat": "csv",
                    "items": {
                        "type": "string"
                    }
                }, {
                    "name": "limit",
                    "in": "query",
                    "description": "maximum number of results to return",
                    "required": False,
                    "type": "integer",
                    "format": "int32"
                }],
                "responses": {
                    "200": {
                        "description": "pet response",
                        "schema": {
                            "type": "array",
                            "items": {
                                "$ref": "#/definitions/Pet"
                            }
                        }
                    },
                    "default": {
                        "description": "unexpected error",
                        "schema": {
                            "$ref": "#/definitions/Error"
                        }
                    }
                }
            },
            "post": {
                "description":
                "Creates a new pet in the store.  Duplicates are allowed",
                "operationId":
                "addPet",
                "parameters": [{
                    "name": "pet",
                    "in": "body",
                    "description": "Pet to add to the store",
                    "required": True,
                    "schema": {
                        "$ref": "#/definitions/NewPet"
                    }
                }],
                "responses": {
                    "200": {
                        "description": "pet response",
                        "schema": {
                            "$ref": "#/definitions/Pet"
                        }
                    },
                    "default": {
                        "description": "unexpected error",
                        "schema": {
                            "$ref": "#/definitions/Error"
                        }
                    }
                }
            }
        },
        "/pets/{id}": {
            "get": {
                "description":
                "Returns a user based on a single ID, if the user does not have access to the pet",
                "operationId":
                "find pet by id",
                "parameters": [{
                    "name": "id",
                    "in": "path",
                    "description": "ID of pet to fetch",
                    "required": True,
                    "type": "integer",
                    "format": "int64"
                }],
                "responses": {
                    "200": {
                        "description": "pet response",
                        "schema": {
                            "$ref": "#/definitions/Pet"
                        }
                    },
                    "default": {
                        "description": "unexpected error",
                        "schema": {
                            "$ref": "#/definitions/Error"
                        }
                    }
                }
            },
            "delete": {
                "description":
                "deletes a single pet based on the ID supplied",
                "operationId":
                "deletePet",
                "parameters": [{
                    "name": "id",
                    "in": "path",
                    "description": "ID of pet to delete",
                    "required": True,
                    "type": "integer",
                    "format": "int64"
                }],
                "responses": {
                    "204": {
                        "description": "pet deleted"
                    },
                    "default": {
                        "description": "unexpected error",
                        "schema": {
                            "$ref": "#/definitions/Error"
                        }
                    }
                }
            }
        }
    },
    "definitions": {
        "Pet": {
            "type":
            "object",
            "allOf": [{
                "$ref": "#/definitions/NewPet"
            }, {
                "required": ["id"],
                "properties": {
                    "id": {
                        "type": "integer",
                        "format": "int64"
                    }
                }
            }]
        },
        "NewPet": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string"
                },
                "tag": {
                    "type": "string"
                }
            }
        },
        "Error": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
                "code": {
                    "type": "integer",
                    "format": "int32"
                },
                "message": {
                    "type": "string"
                }
            }
        }
    }
}


def test_converter():
    s = Converter()
    s.addSpecs(data=openAPI_json_1)


def test_converter_parseSpecs():
    s = Converter.parseSpecs(data=openAPI_json)
    assert len(s) == 2

    assert s[0].name == "listVersionsv2"
    assert s[0].description == "List API versions"
    assert s[1].name == "getVersionDetailsv2"
    assert s[1].description == "Show API version details"

    s = Converter.parseSpecs(data=openAPI_json_1)
    assert len(s) == 4
    assert s[0].name == "findPets"
    assert "Returns all pets from the system that" in s[0].description
    _tags = s[0].parameters.properties.get("tags", {})
    assert _tags.get("_type", "") == "array"
    assert _tags.get("description", "") == "tags to filter by"
