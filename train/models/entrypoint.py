from ai_integration.public_interface import start_loop

from models import eusr, infer

EUSR()

start_loop(inference_function=infer, inputs_schema={
    "image": {
        "type": "image"
    }
} )